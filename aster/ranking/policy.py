from dataclasses import dataclass
from time import perf_counter_ns

from aster.candidates.collect import DiscoveredCandidate
from aster.models.ensemble import RuntimeEnsemble, RuntimePrediction
from aster.plans import parse_explain_json
from .risk import fallback_reason


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


def rank_with_fallback(model: RuntimeEnsemble, candidates, *, max_log_std: float = 0.45,
                       min_predicted_gain: float = 0.10, domain_margin: float = 0.15,
                       max_domain_distance: float = 4.0, max_outside_features: int = 4) -> RankingDecision:
    started = perf_counter_ns()
    if not candidates: raise ValueError("at least one candidate is required")
    native = next((c for c in candidates if c.spec.candidate_id == "native"), None)
    if native is None: raise ValueError("candidate set must include native PostgreSQL plan")
    ranked = tuple(sorted((RankedCandidate(c, model.predict(parse_explain_json(c.plan_json))) for c in candidates),
                          key=lambda item: item.prediction.runtime_ms))
    best = ranked[0]; native_ranked = next(item for item in ranked if item.candidate is native)
    reason_or_none = fallback_reason(model, best_prediction=best.prediction,
                                     native_prediction=native_ranked.prediction,
                                     best_is_native=best.candidate is native,
                                     max_log_std=max_log_std, min_predicted_gain=min_predicted_gain,
                                     domain_margin=domain_margin, max_domain_distance=max_domain_distance,
                                     max_outside_features=max_outside_features)
    fallback = reason_or_none is not None
    return RankingDecision(native if fallback else best.candidate, native, ranked, fallback,
                           reason_or_none or "learned_candidate", (perf_counter_ns() - started) / 1_000_000)
