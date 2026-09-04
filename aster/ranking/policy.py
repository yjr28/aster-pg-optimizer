from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from aster.candidates.collect import DiscoveredCandidate
from aster.models.ensemble import RuntimeEnsemble, RuntimePrediction
from aster.plans import parse_explain_json


@dataclass(frozen=True)
class RankedCandidate:
    candidate: DiscoveredCandidate
    prediction: RuntimePrediction


@dataclass(frozen=True)
class RankingDecision:
    selected: DiscoveredCandidate
    native: DiscoveredCandidate
    ranked: tuple[RankedCandidate, ...]
    fallback: bool
    reason: str
    decision_overhead_ms: float


def rank_with_fallback(
    model: RuntimeEnsemble,
    candidates: tuple[DiscoveredCandidate, ...] | list[DiscoveredCandidate],
    *,
    max_log_std: float = 0.45,
    min_predicted_gain: float = 0.10,
    domain_margin: float = 0.15,
) -> RankingDecision:
    started = perf_counter_ns()
    if not candidates:
        raise ValueError("at least one candidate is required")
    native = next((c for c in candidates if c.spec.candidate_id == "native"), None)
    if native is None:
        raise ValueError("candidate set must include native PostgreSQL plan")

    ranked = tuple(
        sorted(
            (
                RankedCandidate(c, model.predict(parse_explain_json(c.plan_json)))
                for c in candidates
            ),
            key=lambda item: item.prediction.runtime_ms,
        )
    )
    best = ranked[0]
    native_ranked = next(item for item in ranked if item.candidate is native)

    fallback = False
    reason = "learned_candidate"
    if best.candidate is native:
        fallback, reason = True, "native_already_best"
    elif not model.in_training_cost_domain(best.prediction, margin=domain_margin):
        fallback, reason = True, "outside_training_cost_domain"
    elif best.prediction.log_std > max_log_std:
        fallback, reason = True, "uncertainty_too_high"
    else:
        gain = 1.0 - (best.prediction.runtime_ms / native_ranked.prediction.runtime_ms)
        if gain < min_predicted_gain:
            fallback, reason = True, "predicted_gain_too_small"

    selected = native if fallback else best.candidate
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    return RankingDecision(
        selected=selected,
        native=native,
        ranked=ranked,
        fallback=fallback,
        reason=reason,
        decision_overhead_ms=elapsed_ms,
    )
