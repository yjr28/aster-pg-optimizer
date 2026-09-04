from __future__ import annotations

from dataclasses import asdict, dataclass

from aster.models import (
    PairwiseLogisticRanker,
    PostgresCostRanker,
    QueryNormalizedRidgeModel,
    RidgeRuntimeModel,
    RuntimeEnsemble,
    TrainingExample,
)

from .evaluate import evaluate_fallback_policy, evaluate_ranking, fallback_pareto_sweep
from .split import parameter_holdout, query_holdout, template_holdout, workload_holdout


@dataclass(frozen=True)
class TrainingProtocol:
    split_regime: str = "template"
    test_fraction: float = 0.2
    calibration_fraction: float = 0.15
    conformal_alpha: float = 0.10
    min_log_scale: float = 0.05
    seed: int = 7
    trees: int = 128
    min_samples_leaf: int = 2
    ridge_alpha: float = 1.0
    pairwise_c: float = 1.0

    def __post_init__(self) -> None:
        if self.split_regime not in {"template", "parameter", "workload"}:
            raise ValueError(f"unsupported split regime: {self.split_regime}")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if not 0 <= self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal_alpha must be between 0 and 1")
        if self.min_log_scale <= 0:
            raise ValueError("min_log_scale must be positive")


@dataclass(frozen=True)
class TrainingExperimentResult:
    split_regime: str
    primary_train_examples: int
    fit_examples: int
    calibration_examples: int
    test_examples: int
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    calibration_query_groups: tuple[str, ...]
    calibration: dict
    test_intervals: dict
    objective_metadata: dict
    baseline_metrics: dict
    fallback_metrics: dict
    fallback_pareto: tuple[dict, ...]

    def to_jsonable(self) -> dict:
        return asdict(self)


def _primary_split(examples: list[TrainingExample], protocol: TrainingProtocol):
    splitters = {
        "template": template_holdout,
        "parameter": parameter_holdout,
        "workload": workload_holdout,
    }
    return splitters[protocol.split_regime](
        examples,
        test_fraction=protocol.test_fraction,
        seed=protocol.seed,
    )


def _interval_diagnostics(model: RuntimeEnsemble, examples: list[TrainingExample]) -> dict:
    if not examples or model.calibrator is None:
        return {"enabled": False, "examples": len(examples)}
    covered = 0
    relative_widths: list[float] = []
    for example in examples:
        prediction = model.predict(example.plan)
        if prediction.interval_lower_ms is None or prediction.interval_upper_ms is None:
            raise RuntimeError("calibrated model did not emit prediction interval")
        covered += int(
            prediction.interval_lower_ms
            <= example.runtime_ms
            <= prediction.interval_upper_ms
        )
        relative_widths.append(
            (prediction.interval_upper_ms - prediction.interval_lower_ms)
            / example.runtime_ms
        )
    return {
        "enabled": True,
        "examples": len(examples),
        "coverage": covered / len(examples),
        "mean_relative_interval_width": sum(relative_widths) / len(relative_widths),
    }


def run_training_experiment(
    examples: list[TrainingExample],
    protocol: TrainingProtocol,
) -> tuple[RuntimeEnsemble, TrainingExperimentResult]:
    if not examples:
        raise ValueError("training dataset is empty")
    split = _primary_split(examples, protocol)
    primary_train = list(split.train)
    test = list(split.test)

    if protocol.calibration_fraction > 0:
        calibration_split = query_holdout(
            primary_train,
            test_fraction=protocol.calibration_fraction,
            seed=protocol.seed + 101,
        )
        fit_examples = list(calibration_split.train)
        calibration = list(calibration_split.test)
        calibration_groups = tuple(sorted(calibration_split.test_groups))
    else:
        fit_examples = primary_train
        calibration = []
        calibration_groups = ()

    postgres_cost = PostgresCostRanker()
    ridge = RidgeRuntimeModel(alpha=protocol.ridge_alpha).fit(fit_examples)
    normalized = QueryNormalizedRidgeModel(alpha=protocol.ridge_alpha).fit(fit_examples)
    pairwise = PairwiseLogisticRanker(c=protocol.pairwise_c, seed=protocol.seed).fit(fit_examples)
    model = RuntimeEnsemble(
        trees=protocol.trees,
        seed=protocol.seed,
        min_samples_leaf=protocol.min_samples_leaf,
    ).fit(fit_examples)
    if calibration:
        model.calibrate(
            calibration,
            alpha=protocol.conformal_alpha,
            min_log_scale=protocol.min_log_scale,
        )

    if model.calibrator is None:
        calibration_metadata = {"enabled": False, "examples": 0}
    else:
        calibration_metadata = {
            "enabled": True,
            "examples": len(calibration),
            "alpha": model.calibrator.alpha,
            "target_coverage": model.calibrator.target_coverage,
            "quantile": model.calibrator.quantile,
            "min_log_scale": model.calibrator.min_log_scale,
            **{
                f"calibration_{key}": value
                for key, value in _interval_diagnostics(model, calibration).items()
                if key not in {"enabled", "examples"}
            },
        }

    baseline_metrics = {
        "postgres_estimated_cost": asdict(evaluate_ranking(postgres_cost, test)),
        "ridge_log_runtime": asdict(evaluate_ranking(ridge, test)),
        "ridge_query_normalized_runtime": asdict(evaluate_ranking(normalized, test)),
        "pairwise_logistic_ranking": asdict(evaluate_ranking(pairwise, test)),
        "random_forest_runtime": asdict(evaluate_ranking(model, test)),
    }
    fallback_metrics = asdict(evaluate_fallback_policy(model, test))
    pareto = tuple(
        {
            "max_log_std": point.max_log_std,
            "min_predicted_gain": point.min_predicted_gain,
            "metrics": asdict(point.metrics),
        }
        for point in fallback_pareto_sweep(model, test)
    )
    result = TrainingExperimentResult(
        split_regime=protocol.split_regime,
        primary_train_examples=len(primary_train),
        fit_examples=len(fit_examples),
        calibration_examples=len(calibration),
        test_examples=len(test),
        train_groups=tuple(sorted(split.train_groups)),
        test_groups=tuple(sorted(split.test_groups)),
        calibration_query_groups=calibration_groups,
        calibration=calibration_metadata,
        test_intervals=_interval_diagnostics(model, test),
        objective_metadata={"pairwise_training_pairs": pairwise.training_pairs},
        baseline_metrics=baseline_metrics,
        fallback_metrics=fallback_metrics,
        fallback_pareto=pareto,
    )
    return model, result
