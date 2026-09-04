from aster.models import RuntimePrediction
from aster.ranking.risk import fallback_reason


class InDomainModel:
    def in_training_cost_domain(self, prediction, *, margin):
        return True


def prediction(runtime, *, lower=None, upper=None):
    return RuntimePrediction(
        runtime_ms=runtime,
        log_std=0.05,
        root_cost_log=1.0,
        domain_distance=0.0,
        outside_training_range_count=0,
        unseen_structural_features=(),
        interval_lower_ms=lower,
        interval_upper_ms=upper,
        calibrated_log_radius=0.1 if lower is not None else None,
    )


def test_calibrated_bounds_can_reject_point_estimate_gain():
    reason=fallback_reason(
        InDomainModel(),
        best_prediction=prediction(50.0,lower=35.0,upper=90.0),
        native_prediction=prediction(100.0,lower=80.0,upper=130.0),
        best_is_native=False,
        max_log_std=1.0,
        min_predicted_gain=0.10,
        domain_margin=0.15,
    )
    assert reason == "calibrated_gain_too_small"


def test_uncalibrated_predictions_keep_point_estimate_policy():
    reason=fallback_reason(
        InDomainModel(),
        best_prediction=prediction(50.0),
        native_prediction=prediction(100.0),
        best_is_native=False,
        max_log_std=1.0,
        min_predicted_gain=0.10,
        domain_margin=0.15,
    )
    assert reason is None
