from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from aster.features.baseline import baseline_feature_dict
from aster.plans.types import PlanDocument
from aster.uncertainty import ConformalLogCalibrator, FeatureDomain


@dataclass(frozen=True)
class TrainingExample:
    plan: PlanDocument
    runtime_ms: float
    query_id: str
    candidate_id: str = "unknown"
    query_template: str | None = None
    parameter_key: str | None = None
    workload: str | None = None
    dataset_version: str | None = None
    environment_sha256: str | None = None


@dataclass(frozen=True)
class RuntimePrediction:
    runtime_ms: float
    log_std: float
    root_cost_log: float
    domain_distance: float
    outside_training_range_count: int
    unseen_structural_features: tuple[str, ...]
    interval_lower_ms: float | None = None
    interval_upper_ms: float | None = None
    calibrated_log_radius: float | None = None


class RuntimeEnsemble:
    def __init__(self, *, trees: int = 64, seed: int = 7, min_samples_leaf: int = 2):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = RandomForestRegressor(n_estimators=trees, random_state=seed,
                                           min_samples_leaf=min_samples_leaf, n_jobs=1)
        self._fitted = False
        self._root_cost_range = None
        self._feature_domain = None
        self._calibrator: ConformalLogCalibrator | None = None

    def fit(self, examples: list[TrainingExample]) -> "RuntimeEnsemble":
        if len(examples) < 4: raise ValueError("at least 4 training examples are required")
        if any(e.runtime_ms <= 0 for e in examples): raise ValueError("runtime labels must be positive")
        rows = [baseline_feature_dict(e.plan) for e in examples]
        self.model.fit(self.vectorizer.fit_transform(rows), np.log1p([e.runtime_ms for e in examples]))
        costs = [float(row["root_total_cost_log"]) for row in rows]
        self._root_cost_range = (min(costs), max(costs))
        self._feature_domain = FeatureDomain.fit(rows)
        self._calibrator = None
        self._fitted = True
        return self

    @property
    def calibrator(self) -> ConformalLogCalibrator | None:
        return self._calibrator

    def calibrate(
        self,
        examples: list[TrainingExample],
        *,
        alpha: float = 0.10,
        min_log_scale: float = 0.05,
    ) -> "RuntimeEnsemble":
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        if len(examples) < 2:
            raise ValueError("at least two calibration examples are required")
        self._calibrator = None
        predictions = [self.predict(example.plan) for example in examples]
        self._calibrator = ConformalLogCalibrator.fit(
            [prediction.runtime_ms for prediction in predictions],
            [prediction.log_std for prediction in predictions],
            [example.runtime_ms for example in examples],
            alpha=alpha,
            min_log_scale=min_log_scale,
        )
        return self

    def score(self, plan: PlanDocument) -> float: return self.predict(plan).runtime_ms

    def predict(self, plan: PlanDocument) -> RuntimePrediction:
        if not self._fitted or self._feature_domain is None: raise RuntimeError("model is not fitted")
        features = baseline_feature_dict(plan); x = self.vectorizer.transform([features])
        predictions = np.array([tree.predict(x)[0] for tree in self.model.estimators_])
        runtime_ms = max(1e-9, math.expm1(float(predictions.mean())))
        log_std = float(predictions.std(ddof=0))
        domain = self._feature_domain.assess(features)
        lower = upper = radius = None
        if self._calibrator is not None:
            interval = self._calibrator.interval(runtime_ms, log_std)
            lower = interval.lower_ms
            upper = interval.upper_ms
            radius = interval.log_radius
        return RuntimePrediction(
            runtime_ms,
            log_std,
            float(features["root_total_cost_log"]),
            domain.rms_z_distance,
            domain.outside_training_range_count,
            domain.unseen_structural_features,
            lower,
            upper,
            radius,
        )

    def in_training_cost_domain(self, prediction: RuntimePrediction, *, margin: float = 0.15) -> bool:
        if self._root_cost_range is None: raise RuntimeError("model is not fitted")
        low, high = self._root_cost_range; span = max(1e-9, high - low)
        return low - margin * span <= prediction.root_cost_log <= high + margin * span
