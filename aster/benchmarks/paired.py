from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from aster.candidates import DiscoveredCandidate
from aster.integration.psql import ExplainRunner
from aster.plans import parse_explain_json, plan_fingerprint


@dataclass(frozen=True)
class PairedExecutionSample:
    repetition: int
    execution_order: tuple[str, ...]
    native_execution_ms: float
    aster_execution_ms: float
    native_planning_ms: float
    aster_planning_ms: float

    @property
    def execution_speedup(self) -> float:
        return self.native_execution_ms / self.aster_execution_ms

    @property
    def execution_regression_ratio(self) -> float:
        return self.aster_execution_ms / self.native_execution_ms


@dataclass(frozen=True)
class DistributionSummary:
    median_ms: float
    p95_ms: float
    p99_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class PairedBenchmarkResult:
    postgres_version: str
    native_candidate_id: str
    selected_candidate_id: str
    same_physical_plan: bool
    selection_overhead_ms: float
    native_execution: DistributionSummary
    aster_execution: DistributionSummary
    native_database_latency: DistributionSummary
    aster_database_latency: DistributionSummary
    execution_speedup_geomean: float
    end_to_end_speedup_geomean: float
    improved_fraction: float
    regressed_fraction: float
    worst_regression_ratio: float
    samples: tuple[PairedExecutionSample, ...]

    def to_jsonable(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> DistributionSummary:
    return DistributionSummary(
        median_ms=_percentile(values, 0.50),
        p95_ms=_percentile(values, 0.95),
        p99_ms=_percentile(values, 0.99),
        minimum_ms=min(values),
        maximum_ms=max(values),
    )


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _execute_checked(
    runner: ExplainRunner,
    query: str,
    candidate: DiscoveredCandidate,
) -> tuple[float, float]:
    raw = runner.explain(query, candidate.spec.settings, analyze=True)
    plan = parse_explain_json(raw)
    if plan.execution_time_ms is None:
        raise RuntimeError("EXPLAIN ANALYZE did not report Execution Time")
    measured_fingerprint = plan_fingerprint(plan)
    if measured_fingerprint != candidate.fingerprint:
        raise RuntimeError(
            "candidate plan changed during paired benchmark; "
            f"expected {candidate.fingerprint[:12]}, got {measured_fingerprint[:12]}"
        )
    planning_ms = plan.planning_time_ms or 0.0
    return plan.execution_time_ms, planning_ms


def run_paired_benchmark(
    runner: ExplainRunner,
    query: str,
    native: DiscoveredCandidate,
    selected: DiscoveredCandidate,
    *,
    selection_overhead_ms: float,
    warmups: int = 1,
    repetitions: int = 15,
    seed: int = 7,
) -> PairedBenchmarkResult:
    """Benchmark native PostgreSQL against Aster's selected candidate.

    If Aster selected the native physical plan, each repetition executes it once and
    shares that measurement across both sides. Otherwise native and Aster are run as
    a pair in randomized order on every repetition to reduce monotonic cache/thermal
    drift from systematically favoring one side.

    `selection_overhead_ms` is supplied by the caller and should cover candidate
    discovery, feature extraction, model inference, and fallback policy. It is kept
    separate from PostgreSQL execution latency and only added to the Aster side for
    the end-to-end speedup calculation.
    """
    if warmups < 0:
        raise ValueError("warmups must be >= 0")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if selection_overhead_ms < 0.0 or not math.isfinite(selection_overhead_ms):
        raise ValueError("selection_overhead_ms must be finite and >= 0")

    same_plan = native.fingerprint == selected.fingerprint
    rng = random.Random(seed)

    for warmup in range(warmups):
        if same_plan:
            _execute_checked(runner, query, native)
            continue
        order = [("native", native), ("aster", selected)]
        if warmup % 2:
            order.reverse()
        for _, candidate in order:
            _execute_checked(runner, query, candidate)

    samples: list[PairedExecutionSample] = []
    for repetition in range(repetitions):
        if same_plan:
            execution_ms, planning_ms = _execute_checked(runner, query, native)
            samples.append(
                PairedExecutionSample(
                    repetition=repetition,
                    execution_order=("native_shared",),
                    native_execution_ms=execution_ms,
                    aster_execution_ms=execution_ms,
                    native_planning_ms=planning_ms,
                    aster_planning_ms=planning_ms,
                )
            )
            continue

        pair = [("native", native), ("aster", selected)]
        rng.shuffle(pair)
        measured: dict[str, tuple[float, float]] = {}
        for label, candidate in pair:
            measured[label] = _execute_checked(runner, query, candidate)
        native_execution, native_planning = measured["native"]
        aster_execution, aster_planning = measured["aster"]
        samples.append(
            PairedExecutionSample(
                repetition=repetition,
                execution_order=tuple(label for label, _ in pair),
                native_execution_ms=native_execution,
                aster_execution_ms=aster_execution,
                native_planning_ms=native_planning,
                aster_planning_ms=aster_planning,
            )
        )

    native_exec = [sample.native_execution_ms for sample in samples]
    aster_exec = [sample.aster_execution_ms for sample in samples]
    native_db = [sample.native_execution_ms + sample.native_planning_ms for sample in samples]
    aster_db = [sample.aster_execution_ms + sample.aster_planning_ms for sample in samples]
    execution_speedups = [n / a for n, a in zip(native_exec, aster_exec)]
    end_to_end_speedups = [
        n / (a + selection_overhead_ms)
        for n, a in zip(native_db, aster_db)
    ]

    improved = sum(a < n for n, a in zip(native_exec, aster_exec)) / len(samples)
    regressed = sum(a > n for n, a in zip(native_exec, aster_exec)) / len(samples)

    return PairedBenchmarkResult(
        postgres_version=runner.postgres_version(),
        native_candidate_id=native.spec.candidate_id,
        selected_candidate_id=selected.spec.candidate_id,
        same_physical_plan=same_plan,
        selection_overhead_ms=selection_overhead_ms,
        native_execution=_summary(native_exec),
        aster_execution=_summary(aster_exec),
        native_database_latency=_summary(native_db),
        aster_database_latency=_summary(aster_db),
        execution_speedup_geomean=_geometric_mean(execution_speedups),
        end_to_end_speedup_geomean=_geometric_mean(end_to_end_speedups),
        improved_fraction=improved,
        regressed_fraction=regressed,
        worst_regression_ratio=max(a / n for n, a in zip(native_exec, aster_exec)),
        samples=tuple(samples),
    )
