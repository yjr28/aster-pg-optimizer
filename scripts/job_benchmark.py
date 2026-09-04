from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from aster.artifacts import load_model
from aster.benchmarks import JobBenchmarkConfig, benchmark_job_workload, capture_benchmark_environment
from aster.candidates import default_candidates, research_candidates
from aster.integration import PsqlExplainRunner
from aster.workloads import build_job_manifest, load_job_queries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_preflight(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"workload_sha256", "dataset_version", "benchmark_input_sha256"}
    missing = sorted(required - payload.keys())
    if missing:
        raise SystemExit(f"preflight manifest missing fields: {', '.join(missing)}")
    return payload


def _resolve_identity(
    output_dir: Path,
    *,
    requested_experiment_id: str | None,
    benchmark_input_sha256: str,
    model_sha256: str,
    environment_sha256: str,
    candidate_set: str,
) -> dict:
    path = output_dir / "benchmark_identity.json"
    if path.exists():
        identity = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "benchmark_input_sha256": benchmark_input_sha256,
            "model_sha256": model_sha256,
            "environment_sha256": environment_sha256,
            "candidate_set": candidate_set,
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise SystemExit(
                    f"benchmark output directory identity mismatch for {key}: "
                    f"existing={identity.get(key)!r} requested={value!r}"
                )
        if requested_experiment_id and identity.get("experiment_id") != requested_experiment_id:
            raise SystemExit("benchmark output directory belongs to another experiment ID")
        return identity

    identity = {
        "schema_version": 2,
        "experiment_id": requested_experiment_id or str(uuid4()),
        "benchmark_input_sha256": benchmark_input_sha256,
        "model_sha256": model_sha256,
        "environment_sha256": environment_sha256,
        "candidate_set": candidate_set,
        "cache_policy": "paired-warm-cache",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return identity


def _persist_environment(output_dir: Path, environment) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "benchmark_environment.json"
    payload = environment.to_jsonable()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("environment_sha256") != environment.environment_sha256:
            raise SystemExit("benchmark output directory was captured under another environment")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run native vs learned vs fallback benchmarks across JOB")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dsn")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--experiment-id")
    parser.add_argument("--candidate-set", choices=("fast", "research"), default="research")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--max-log-std", type=float, default=0.45)
    parser.add_argument("--min-predicted-gain", type=float, default=0.10)
    parser.add_argument("--domain-margin", type=float, default=0.15)
    parser.add_argument("--max-domain-distance", type=float, default=4.0)
    parser.add_argument("--max-outside-features", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-partial-workload", action="store_true")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("ASTER_DSN")
    if not dsn:
        raise SystemExit("database DSN required via --dsn or ASTER_DSN")
    if not args.model.is_file():
        raise SystemExit(f"model artifact not found: {args.model}")

    preflight = _load_preflight(args.preflight)
    strict = not args.allow_partial_workload
    workload = build_job_manifest(args.query_dir, strict=strict)
    if workload.workload_sha256 != preflight["workload_sha256"]:
        raise SystemExit("JOB query checkout no longer matches the preflight workload hash")

    runner = PsqlExplainRunner(dsn, timeout_s=args.timeout)
    environment = capture_benchmark_environment(runner)
    _persist_environment(args.output_dir, environment)

    model_sha256 = _sha256_file(args.model)
    identity = _resolve_identity(
        args.output_dir,
        requested_experiment_id=args.experiment_id,
        benchmark_input_sha256=preflight["benchmark_input_sha256"],
        model_sha256=model_sha256,
        environment_sha256=environment.environment_sha256,
        candidate_set=args.candidate_set,
    )
    model = load_model(args.model)
    specs = research_candidates() if args.candidate_set == "research" else default_candidates()
    config = JobBenchmarkConfig(
        experiment_id=identity["experiment_id"],
        benchmark_input_sha256=preflight["benchmark_input_sha256"],
        dataset_version=preflight["dataset_version"],
        seed=args.seed,
        warmups=args.warmups,
        repetitions=args.repetitions,
        max_log_std=args.max_log_std,
        min_predicted_gain=args.min_predicted_gain,
        domain_margin=args.domain_margin,
        max_domain_distance=args.max_domain_distance,
        max_outside_features=args.max_outside_features,
    )
    result = benchmark_job_workload(
        runner,
        model,
        load_job_queries(args.query_dir, strict=strict),
        specs,
        args.output_dir,
        config=config,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    result["model_sha256"] = model_sha256
    result["environment_sha256"] = environment.environment_sha256
    result["candidate_set"] = args.candidate_set
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed_queries"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
