from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from aster.candidates import CandidateCollector, CandidateSpec
from aster.ranking import rank_with_fallback
from aster.workloads.job import JobQuery

from .paired import (
    DistributionSummary,
    PairedBenchmarkResult,
    PairedExecutionSample,
    run_paired_benchmark,
)
from .workload import WorkloadQueryBenchmark, summarize_workload_benchmark


@dataclass(frozen=True)
class JobBenchmarkConfig:
    experiment_id: str
    benchmark_input_sha256: str
    dataset_version: str
    seed: int = 7
    warmups: int = 2
    repetitions: int = 15
    max_log_std: float = 0.45
    min_predicted_gain: float = 0.10
    domain_margin: float = 0.15
    max_domain_distance: float = 4.0
    max_outside_features: int = 4

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if len(self.benchmark_input_sha256) != 64:
            raise ValueError("benchmark_input_sha256 must be a SHA-256 hex string")
        if self.warmups < 0 or self.repetitions < 1:
            raise ValueError("warmups must be >=0 and repetitions >=1")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def native_control_from_paired(
    paired: PairedBenchmarkResult,
    *,
    selection_overhead_ms: float,
) -> PairedBenchmarkResult:
    """Reuse the native control side when fallback selects native.

    The no-fallback paired experiment already executed native once in every repetition.
    Reusing those measurements avoids a second, statistically different native run just
    to represent fallback. The Aster end-to-end side still pays selection overhead.
    """
    samples = tuple(
        PairedExecutionSample(
            repetition=sample.repetition,
            execution_order=("native_shared_control",),
            native_execution_ms=sample.native_execution_ms,
            aster_execution_ms=sample.native_execution_ms,
            native_planning_ms=sample.native_planning_ms,
            aster_planning_ms=sample.native_planning_ms,
        )
        for sample in paired.samples
    )
    end_to_end = [
        (sample.native_execution_ms + sample.native_planning_ms)
        / (sample.native_execution_ms + sample.native_planning_ms + selection_overhead_ms)
        for sample in samples
    ]
    return PairedBenchmarkResult(
        postgres_version=paired.postgres_version,
        native_candidate_id=paired.native_candidate_id,
        selected_candidate_id=paired.native_candidate_id,
        same_physical_plan=True,
        selection_overhead_ms=selection_overhead_ms,
        native_execution=paired.native_execution,
        aster_execution=paired.native_execution,
        native_database_latency=paired.native_database_latency,
        aster_database_latency=paired.native_database_latency,
        execution_speedup_geomean=1.0,
        end_to_end_speedup_geomean=_geomean(end_to_end),
        improved_fraction=0.0,
        regressed_fraction=0.0,
        worst_regression_ratio=1.0,
        samples=samples,
    )


def _distribution_from_dict(payload: dict) -> DistributionSummary:
    return DistributionSummary(**payload)


def _paired_from_dict(payload: dict) -> PairedBenchmarkResult:
    return PairedBenchmarkResult(
        postgres_version=payload["postgres_version"],
        native_candidate_id=payload["native_candidate_id"],
        selected_candidate_id=payload["selected_candidate_id"],
        same_physical_plan=payload["same_physical_plan"],
        selection_overhead_ms=payload["selection_overhead_ms"],
        native_execution=_distribution_from_dict(payload["native_execution"]),
        aster_execution=_distribution_from_dict(payload["aster_execution"]),
        native_database_latency=_distribution_from_dict(payload["native_database_latency"]),
        aster_database_latency=_distribution_from_dict(payload["aster_database_latency"]),
        execution_speedup_geomean=payload["execution_speedup_geomean"],
        end_to_end_speedup_geomean=payload["end_to_end_speedup_geomean"],
        improved_fraction=payload["improved_fraction"],
        regressed_fraction=payload["regressed_fraction"],
        worst_regression_ratio=payload["worst_regression_ratio"],
        samples=tuple(PairedExecutionSample(**sample) for sample in payload["samples"]),
    )


def _entry_from_payload(payload: dict, *, config: JobBenchmarkConfig) -> WorkloadQueryBenchmark:
    if payload.get("experiment_id") != config.experiment_id:
        raise ValueError("benchmark shard belongs to another experiment")
    if payload.get("benchmark_input_sha256") != config.benchmark_input_sha256:
        raise ValueError("benchmark shard belongs to another benchmark input")
    return WorkloadQueryBenchmark(
        query_id=payload["query_id"],
        no_fallback=_paired_from_dict(payload["no_fallback"]),
        fallback=_paired_from_dict(payload["fallback"]),
        fallback_triggered=payload["fallback_triggered"],
        fallback_reason=payload.get("fallback_reason") or "",
        selection_overhead_ms=payload["selection_overhead_ms"],
    )


def benchmark_job_workload(
    runner,
    model,
    queries: tuple[JobQuery, ...] | list[JobQuery],
    candidate_specs: tuple[CandidateSpec, ...] | list[CandidateSpec],
    output_dir: str | Path,
    *,
    config: JobBenchmarkConfig,
    resume: bool = True,
    fail_fast: bool = False,
) -> dict:
    root = Path(output_dir)
    entries: list[WorkloadQueryBenchmark] = []
    failed_ids: list[str] = []

    for query_index, query in enumerate(queries):
        shard = root / "queries" / f"{query.query_id}.json"
        if resume and shard.exists():
            payload = json.loads(shard.read_text(encoding="utf-8"))
            if payload.get("query_id") != query.query_id:
                raise ValueError(f"query id mismatch in benchmark shard {shard}")
            entries.append(_entry_from_payload(payload, config=config))
            continue

        try:
            collector = CandidateCollector(runner)
            started = perf_counter_ns()
            discovery = collector.discover_report(query.sql, candidate_specs)
            decision = rank_with_fallback(
                model,
                discovery.unique_candidates,
                max_log_std=config.max_log_std,
                min_predicted_gain=config.min_predicted_gain,
                domain_margin=config.domain_margin,
                max_domain_distance=config.max_domain_distance,
                max_outside_features=config.max_outside_features,
            )
            selection_overhead_ms = (perf_counter_ns() - started) / 1_000_000
            no_fallback_candidate = decision.ranked[0].candidate
            paired_seed = config.seed + query_index
            no_fallback = run_paired_benchmark(
                runner,
                query.sql,
                decision.native,
                no_fallback_candidate,
                selection_overhead_ms=selection_overhead_ms,
                warmups=config.warmups,
                repetitions=config.repetitions,
                seed=paired_seed,
            )
            if decision.fallback:
                fallback = native_control_from_paired(
                    no_fallback, selection_overhead_ms=selection_overhead_ms
                )
            else:
                fallback = no_fallback

            entry = WorkloadQueryBenchmark(
                query_id=query.query_id,
                no_fallback=no_fallback,
                fallback=fallback,
                fallback_triggered=decision.fallback,
                fallback_reason=decision.reason or "",
                selection_overhead_ms=selection_overhead_ms,
            )
            payload = {
                "schema_version": 1,
                "experiment_id": config.experiment_id,
                "benchmark_input_sha256": config.benchmark_input_sha256,
                "dataset_version": config.dataset_version,
                "query_id": query.query_id,
                "family": query.family,
                "variant": query.variant,
                "candidate_discovery": {
                    "attempted_interventions": discovery.attempted_interventions,
                    "unique_plan_count": discovery.unique_plan_count,
                    "duplicate_interventions": discovery.duplicate_interventions,
                    "uniqueness_ratio": discovery.uniqueness_ratio,
                },
                "selected_no_fallback": no_fallback_candidate.spec.candidate_id,
                "selected_fallback": decision.selected.spec.candidate_id,
                "fallback_triggered": decision.fallback,
                "fallback_reason": decision.reason or "",
                "selection_overhead_ms": selection_overhead_ms,
                "ranking_only_overhead_ms": decision.decision_overhead_ms,
                "no_fallback": no_fallback.to_jsonable(),
                "fallback": fallback.to_jsonable(),
                "predictions": [
                    {
                        "candidate_id": ranked.candidate.spec.candidate_id,
                        "runtime_ms": ranked.prediction.runtime_ms,
                        "log_std": ranked.prediction.log_std,
                        "domain_distance": ranked.prediction.domain_distance,
                        "outside_training_range_count": ranked.prediction.outside_training_range_count,
                        "unseen_structural_features": list(ranked.prediction.unseen_structural_features),
                    }
                    for ranked in decision.ranked
                ],
            }
            _atomic_write(shard, json.dumps(payload, indent=2, sort_keys=True) + "\n")
            failure_path = root / "failures" / f"{query.query_id}.json"
            if failure_path.exists():
                failure_path.unlink()
            entries.append(entry)
        except Exception as exc:
            failed_ids.append(query.query_id)
            failure = {
                "query_id": query.query_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(
                root / "failures" / f"{query.query_id}.json",
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
            )
            if fail_fast:
                raise

    result = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "benchmark_input_sha256": config.benchmark_input_sha256,
        "dataset_version": config.dataset_version,
        "expected_queries": len(queries),
        "completed_queries": len(entries),
        "failed_queries": len(failed_ids),
        "failure_query_ids": failed_ids,
        "config": asdict(config),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not failed_ids and len(entries) == len(queries):
        result["summary"] = summarize_workload_benchmark(entries).to_jsonable()
    _atomic_write(root / "benchmark_manifest.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
