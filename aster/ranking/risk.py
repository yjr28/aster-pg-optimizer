from aster.models.ensemble import RuntimeEnsemble, RuntimePrediction


def fallback_reason(model: RuntimeEnsemble, *, best_prediction: RuntimePrediction,
                    native_prediction: RuntimePrediction, best_is_native: bool,
                    max_log_std: float, min_predicted_gain: float, domain_margin: float,
                    max_domain_distance: float = 4.0, max_outside_features: int = 4) -> str | None:
    if best_is_native: return "native_already_best"
    if best_prediction.unseen_structural_features: return "unseen_plan_structure"
    if best_prediction.domain_distance > max_domain_distance: return "outside_training_feature_domain"
    if best_prediction.outside_training_range_count > max_outside_features: return "too_many_out_of_range_features"
    if not model.in_training_cost_domain(best_prediction, margin=domain_margin): return "outside_training_cost_domain"
    if best_prediction.log_std > max_log_std: return "uncertainty_too_high"
    gain = 1.0 - best_prediction.runtime_ms / native_prediction.runtime_ms
    if gain < min_predicted_gain: return "predicted_gain_too_small"
    return None
