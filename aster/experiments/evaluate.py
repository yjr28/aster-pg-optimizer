from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from aster.models import PlanScorer, RuntimeEnsemble, TrainingExample
from aster.ranking.risk import fallback_reason


@dataclass(frozen=True)
class RankingMetrics:
    queries: int
    geometric_mean_speedup_vs_native: float
    median_speedup_vs_native: float
    improved_fraction: float
    regressed_fraction: float
    worst_regression_ratio: float
    oracle_geometric_mean_speedup: float


@dataclass(frozen=True)
class FallbackMetrics:
    queries: int
    geometric_mean_speedup_vs_native: float
    median_speedup_vs_native: float
    improved_fraction: float
    regressed_fraction: float
    worst_regression_ratio: float
    fallback_fraction: float
    oracle_geometric_mean_speedup: float


@dataclass(frozen=True)
class FallbackCurvePoint:
    max_log_std: float
    min_predicted_gain: float
    metrics: FallbackMetrics


def _geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _group_candidates(examples):
    grouped = defaultdict(list)
    for example in examples:
        if example.runtime_ms <= 0: raise ValueError("measured runtime must be positive")
        grouped[(
            example.environment_sha256 or "",
            example.dataset_version or "",
            example.workload or "",
            example.query_id,
        )].append(example)
    return grouped


def _metrics(speedups, oracle, regressions):
    if not speedups: raise ValueError("no queries to evaluate")
    return RankingMetrics(len(speedups), _geomean(speedups), float(statistics.median(speedups)),
                          sum(s > 1 for s in speedups) / len(speedups),
                          sum(s < 1 for s in speedups) / len(speedups), max(regressions), _geomean(oracle))


def evaluate_ranking(model: PlanScorer, examples: list[TrainingExample]) -> RankingMetrics:
    speedups, oracle, regressions = [], [], []
    for (_, _, _, query_id), candidates in _group_candidates(examples).items():
        native = next((e for e in candidates if e.candidate_id == "native"), None)
        if native is None: raise ValueError(f"query {query_id} has no native candidate")
        selected = min(candidates, key=lambda e: model.score(e.plan))
        speedups.append(native.runtime_ms / selected.runtime_ms)
        oracle.append(native.runtime_ms / min(c.runtime_ms for c in candidates))
        regressions.append(selected.runtime_ms / native.runtime_ms)
    return _metrics(speedups, oracle, regressions)


def evaluate_runtime_ranking(model: PlanScorer, examples: list[TrainingExample]) -> RankingMetrics:
    return evaluate_ranking(model, examples)


def evaluate_fallback_policy(model: RuntimeEnsemble, examples: list[TrainingExample], *,
                             max_log_std: float = 0.45, min_predicted_gain: float = 0.10,
                             domain_margin: float = 0.15, max_domain_distance: float = 4.0,
                             max_outside_features: int = 4) -> FallbackMetrics:
    speedups, oracle, regressions = [], [], []; fallback_count = 0
    for (_, _, _, query_id), candidates in _group_candidates(examples).items():
        native = next((e for e in candidates if e.candidate_id == "native"), None)
        if native is None: raise ValueError(f"query {query_id} has no native candidate")
        predictions = [(c, model.predict(c.plan)) for c in candidates]
        best, best_prediction = min(predictions, key=lambda item: item[1].runtime_ms)
        native_prediction = next(pred for cand, pred in predictions if cand is native)
        reason = fallback_reason(model, best_prediction=best_prediction, native_prediction=native_prediction,
                                 best_is_native=best is native, max_log_std=max_log_std,
                                 min_predicted_gain=min_predicted_gain, domain_margin=domain_margin,
                                 max_domain_distance=max_domain_distance, max_outside_features=max_outside_features)
        selected = native if reason is not None else best; fallback_count += int(reason is not None)
        speedups.append(native.runtime_ms / selected.runtime_ms)
        oracle.append(native.runtime_ms / min(c.runtime_ms for c in candidates))
        regressions.append(selected.runtime_ms / native.runtime_ms)
    base = _metrics(speedups, oracle, regressions)
    return FallbackMetrics(base.queries, base.geometric_mean_speedup_vs_native, base.median_speedup_vs_native,
                           base.improved_fraction, base.regressed_fraction, base.worst_regression_ratio,
                           fallback_count / base.queries, base.oracle_geometric_mean_speedup)


def fallback_pareto_sweep(model: RuntimeEnsemble, examples: list[TrainingExample], *,
                          max_log_stds=(0.10,0.20,0.35,0.50,0.75), min_predicted_gains=(0.00,0.05,0.10,0.20),
                          domain_margin: float = 0.15, max_domain_distance: float = 4.0,
                          max_outside_features: int = 4) -> tuple[FallbackCurvePoint, ...]:
    points = []
    for max_std in max_log_stds:
        if max_std < 0: raise ValueError("max_log_std values must be non-negative")
        for min_gain in min_predicted_gains:
            if not 0 <= min_gain < 1: raise ValueError("min_predicted_gain values must be in [0, 1)")
            points.append(FallbackCurvePoint(max_std, min_gain, evaluate_fallback_policy(
                model, examples, max_log_std=max_std, min_predicted_gain=min_gain,
                domain_margin=domain_margin, max_domain_distance=max_domain_distance,
                max_outside_features=max_outside_features)))
    return tuple(points)
