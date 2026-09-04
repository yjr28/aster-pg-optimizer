from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from aster.benchmarks.environment import capture_benchmark_environment
from aster.candidates import CandidateCollector, default_candidates, research_candidates
from aster.integration import PsqlExplainRunner
from aster.workloads import (
    TpchCollectionConfig,
    build_tpch_workload_manifest,
    collect_tpch_workload,
)


def _revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_preflight(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read preflight manifest {path}: {exc}") from exc
    required = {
        "specification_version",
        "scale_factor",
        "workload_sha256",
        "dataset_version",
        "benchmark_input_sha256",
        "query_count",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise SystemExit(f"preflight manifest missing fields: {', '.join(missing)}")
    return payload


def _bind_environment(output_dir: Path, environment) -> None:
    path = output_dir / "environment.json"
    payload = environment.to_jsonable()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid existing environment manifest {path}: {exc}") from exc
        if existing.get("environment_sha256") != environment.environment_sha256:
            raise SystemExit(
                "collection output directory belongs to another benchmark environment: "
                f"existing={existing.get('environment_sha256')} current={environment.environment_sha256}"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect measured candidate plans for TPC-H")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dsn")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--experiment-id")
    parser.add_argument("--candidate-set", choices=("fast", "research"), default="research")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-partial-workload", action="store_true")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("ASTER_DSN")
    if not dsn:
        raise SystemExit("database DSN required via --dsn or ASTER_DSN")

    preflight = _load_preflight(args.preflight)
    strict = not args.allow_partial_workload
    current_workload = build_tpch_workload_manifest(args.query_dir, strict=strict)
    if current_workload.workload_sha256 != preflight["workload_sha256"]:
        raise SystemExit(
            "TPC-H query checkout no longer matches preflight manifest: "
            f"expected {preflight['workload_sha256'][:12]}, got {current_workload.workload_sha256[:12]}"
        )

    runner = PsqlExplainRunner(dsn, timeout_s=args.timeout)
    environment = capture_benchmark_environment(runner)
    _bind_environment(args.output_dir, environment)
    experiment_id = args.experiment_id or str(uuid4())
    config = TpchCollectionConfig(
        experiment_id=experiment_id,
        dataset_version=preflight["dataset_version"],
        benchmark_input_sha256=preflight["benchmark_input_sha256"],
        workload_sha256=preflight["workload_sha256"],
        specification_version=preflight["specification_version"],
        scale_factor=preflight["scale_factor"],
        run_seed=args.seed,
        code_revision=_revision(),
        environment_sha256=environment.environment_sha256,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    collector = CandidateCollector(runner)
    candidates = research_candidates() if args.candidate_set == "research" else default_candidates()
    summary = collect_tpch_workload(
        collector,
        args.query_dir,
        args.output_dir,
        config=config,
        candidates=candidates,
        strict_workload=strict,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    result = {
        "experiment_id": experiment_id,
        "dataset_version": config.dataset_version,
        "benchmark_input_sha256": config.benchmark_input_sha256,
        "environment_sha256": config.environment_sha256,
        "candidate_set": args.candidate_set,
        "summary": asdict(summary),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if summary.failed_queries == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
