from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from aster.models import RuntimeEnsemble, TrainingExample


@dataclass(frozen=True)
class RankingMetrics:
    queries: int
    geometric_mean_speedup_vs_native: float
    median_speedup_vs_native: float
    improved_fraction: float
    regressed_fraction: float
    worst_regression_ratio: float
    oracle_geometric_mean_speedup: float


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def evaluate_runtime_ranking(model: RuntimeEnsemble, examples: list[TrainingExample]) -> RankingMetrics:
    """Evaluate selected-plan *measured runtime* against native PostgreSQL runtime.

    Prediction error is intentionally absent from these database-performance metrics.
    """
    by_query: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_query[example.query_id].append(example)

    speedups: list[float] = []
    oracle_speedups: list[float] = []
    regression_ratios: list[float] = []
    for query_id, candidates in by_query.items():
        native = next((e for e in candidates if e.candidate_id == "native"), None)
        if native is None:
            raise ValueError(f"query {query_id} has no native candidate")
        selected = min(candidates, key=lambda e: model.predict(e.plan).runtime_ms)
        actual_speedup = native.runtime_ms / selected.runtime_ms
        speedups.append(actual_speedup)
        oracle = min(c.runtime_ms for c in candidates)
        oracle_speedups.append(native.runtime_ms / oracle)
        regression_ratios.append(selected.runtime_ms / native.runtime_ms)

    if not speedups:
        raise ValueError("no queries to evaluate")
    return RankingMetrics(
        queries=len(speedups),
        geometric_mean_speedup_vs_native=_geomean(speedups),
        median_speedup_vs_native=float(statistics.median(speedups)),
        improved_fraction=sum(s > 1.0 for s in speedups) / len(speedups),
        regressed_fraction=sum(s < 1.0 for s in speedups) / len(speedups),
        worst_regression_ratio=max(regression_ratios),
        oracle_geometric_mean_speedup=_geomean(oracle_speedups),
    )
