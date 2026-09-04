from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from aster.features.baseline import baseline_feature_dict
from aster.plans.types import PlanDocument

from .ensemble import TrainingExample


class PlanScorer(Protocol):
    """Lower scores are preferred when ranking candidate plans."""

    def score(self, plan: PlanDocument) -> float: ...


class PostgresCostRanker:
    """Use PostgreSQL's own root estimated cost as the ranking baseline."""

    name = "postgres_estimated_cost"

    def score(self, plan: PlanDocument) -> float:
        return float(plan.root.total_cost)


class RidgeRuntimeModel:
    """Linear learned baseline over the same leak-free plan-summary features."""

    name = "ridge_log_runtime"

    def __init__(self, *, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, examples: list[TrainingExample]) -> "RidgeRuntimeModel":
        if len(examples) < 2:
            raise ValueError("at least 2 training examples are required")
        if any(example.runtime_ms <= 0 for example in examples):
            raise ValueError("runtime labels must be positive")
        x = self.vectorizer.fit_transform(
            [baseline_feature_dict(example.plan) for example in examples]
        )
        y = np.log1p([example.runtime_ms for example in examples])
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict_runtime_ms(self, plan: PlanDocument) -> float:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        x = self.vectorizer.transform([baseline_feature_dict(plan)])
        log_runtime = float(self.model.predict(x)[0])
        return max(1e-9, math.expm1(log_runtime))

    def score(self, plan: PlanDocument) -> float:
        return self.predict_runtime_ms(plan)
