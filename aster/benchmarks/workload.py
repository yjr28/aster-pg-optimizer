from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from .paired import DistributionSummary, PairedBenchmarkResult


@dataclass(frozen=True)
class WorkloadQueryBenchmark:
    query_id: str
    no_fallback: PairedBenchmarkResult
    fallback: PairedBenchmarkResult
    fallback_triggered: bool
    fallback_reason: str
    selection_overhead_ms: float


@dataclass(frozen=True)
class WorkloadVariantSummary:
    geometric_mean_speedup_vs_native: float
    median_speedup_vs_native: float
    execution_latency_ms: DistributionSummary
    end_to_end_latency_ms: DistributionSummary
    improved_fraction: float
    regressed_fraction: float
    worst_regression_ratio: float
    maximum_speedup: float


@dataclass(frozen=True)
class WorkloadBenchmarkSummary:
    query_count: int
    native_execution_latency_ms: DistributionSummary
    no_fallback: WorkloadVariantSummary
    fallback: WorkloadVariantSummary
    fallback_fraction: float
    selection_overhead_ms: DistributionSummary

    def to_jsonable(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty workload")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> DistributionSummary:
    return DistributionSummary(
        median_ms=float(statistics.median(values)),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
        minimum_ms=min(values),
        maximum_ms=max(values),
    )


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _variant_summary(
    results: list[PairedBenchmarkResult],
    selection_overheads: list[float],
) -> WorkloadVariantSummary:
    selected_execution = [result.aster_execution.median_ms for result in results]
    selected_end_to_end = [
        result.aster_database_latency.median_ms + overhead
        for result, overhead in zip(results, selection_overheads)
    ]
    speedups = [
        result.native_execution.median_ms / result.aster_execution.median_ms
        for result in results
    ]
    regressions = [1.0 / speedup for speedup in speedups]
    return WorkloadVariantSummary(
        geometric_mean_speedup_vs_native=_geomean(speedups),
        median_speedup_vs_native=float(statistics.median(speedups)),
        execution_latency_ms=_distribution(selected_execution),
        end_to_end_latency_ms=_distribution(selected_end_to_end),
        improved_fraction=sum(speedup > 1.0 for speedup in speedups) / len(speedups),
        regressed_fraction=sum(speedup < 1.0 for speedup in speedups) / len(speedups),
        worst_regression_ratio=max(regressions),
        maximum_speedup=max(speedups),
    )


def summarize_workload_benchmark(
    queries: list[WorkloadQueryBenchmark] | tuple[WorkloadQueryBenchmark, ...],
) -> WorkloadBenchmarkSummary:
    if not queries:
        raise ValueError("at least one query benchmark is required")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("workload benchmark contains duplicate query IDs")
    if any(query.selection_overhead_ms < 0 or not math.isfinite(query.selection_overhead_ms) for query in queries):
        raise ValueError("selection overhead must be finite and non-negative")

    overheads = [query.selection_overhead_ms for query in queries]
    no_fallback = [query.no_fallback for query in queries]
    fallback = [query.fallback for query in queries]
    native_medians = [result.native_execution.median_ms for result in fallback]

    return WorkloadBenchmarkSummary(
        query_count=len(queries),
        native_execution_latency_ms=_distribution(native_medians),
        no_fallback=_variant_summary(no_fallback, overheads),
        fallback=_variant_summary(fallback, overheads),
        fallback_fraction=sum(query.fallback_triggered for query in queries) / len(queries),
        selection_overhead_ms=_distribution(overheads),
    )
