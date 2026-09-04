from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from aster.features.baseline import baseline_feature_dict
from aster.plans.types import PlanDocument


@dataclass(frozen=True)
class TrainingExample:
    plan: PlanDocument
    runtime_ms: float
    query_id: str
    candidate_id: str = "unknown"
    query_template: str | None = None
    parameter_key: str | None = None


@dataclass(frozen=True)
class RuntimePrediction:
    runtime_ms: float
    log_std: float
    root_cost_log: float


class RuntimeEnsemble:
    """First learned baseline: random-forest runtime model over plan-summary features.

    It is intentionally not the final Aster model. The graph representation exists
    separately and a graph model must outperform this baseline before replacing it.
    """

    def __init__(self, *, trees: int = 64, seed: int = 7, min_samples_leaf: int = 2):
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = RandomForestRegressor(
            n_estimators=trees,
            random_state=seed,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
        )
        self._fitted = False
        self._root_cost_range: tuple[float, float] | None = None

    def fit(self, examples: list[TrainingExample]) -> "RuntimeEnsemble":
        if len(examples) < 4:
            raise ValueError("at least 4 training examples are required")
        if any(example.runtime_ms <= 0 for example in examples):
            raise ValueError("runtime labels must be positive")
        dicts = [baseline_feature_dict(example.plan) for example in examples]
        x = self.vectorizer.fit_transform(dicts)
        y = np.log1p([example.runtime_ms for example in examples])
        self.model.fit(x, y)
        costs = [float(d["root_total_cost_log"]) for d in dicts]
        self._root_cost_range = (min(costs), max(costs))
        self._fitted = True
        return self

    def predict(self, plan: PlanDocument) -> RuntimePrediction:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        features = baseline_feature_dict(plan)
        x = self.vectorizer.transform([features])
        tree_predictions = np.array([tree.predict(x)[0] for tree in self.model.estimators_])
        log_mean = float(tree_predictions.mean())
        log_std = float(tree_predictions.std(ddof=0))
        return RuntimePrediction(
            runtime_ms=max(1e-9, math.expm1(log_mean)),
            log_std=log_std,
            root_cost_log=float(features["root_total_cost_log"]),
        )

    def in_training_cost_domain(self, prediction: RuntimePrediction, *, margin: float = 0.15) -> bool:
        if self._root_cost_range is None:
            raise RuntimeError("model is not fitted")
        low, high = self._root_cost_range
        span = max(1e-9, high - low)
        return low - margin * span <= prediction.root_cost_log <= high + margin * span
