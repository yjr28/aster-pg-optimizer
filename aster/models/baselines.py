from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Protocol

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

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


def _query_key(example: TrainingExample) -> tuple[str, str, str]:
    return (example.dataset_version or "", example.workload or "", example.query_id)


class QueryNormalizedRidgeModel:
    """Predict log runtime relative to the measured native plan for each training query.

    The normalization target removes much of the absolute query-difficulty scale and
    asks the model a ranking-focused question: how expensive is this candidate relative
    to PostgreSQL's own choice for the same query? Native runtime is used only to build
    training labels, never as an inference feature.
    """

    name = "ridge_query_normalized_runtime"

    def __init__(self, *, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, examples: list[TrainingExample]) -> "QueryNormalizedRidgeModel":
        if len(examples) < 2:
            raise ValueError("at least 2 training examples are required")
        grouped: dict[tuple[str, str, str], list[TrainingExample]] = defaultdict(list)
        for example in examples:
            if example.runtime_ms <= 0:
                raise ValueError("runtime labels must be positive")
            grouped[_query_key(example)].append(example)

        targets: list[float] = []
        rows: list[dict] = []
        for key, candidates in grouped.items():
            native = next((item for item in candidates if item.candidate_id == "native"), None)
            if native is None:
                raise ValueError(f"training query {key[2]} has no native candidate")
            for candidate in candidates:
                rows.append(baseline_feature_dict(candidate.plan))
                targets.append(math.log(candidate.runtime_ms / native.runtime_ms))
        self.model.fit(self.vectorizer.fit_transform(rows), np.asarray(targets))
        self._fitted = True
        return self

    def score(self, plan: PlanDocument) -> float:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        x = self.vectorizer.transform([baseline_feature_dict(plan)])
        return float(self.model.predict(x)[0])


class PairwiseLogisticRanker:
    """Bradley-Terry-style linear ranker trained on within-query plan differences.

    Each non-tied pair contributes both orientations. A positive logistic decision means
    the first plan is predicted slower, so the learned per-plan linear utility is a
    lower-is-better score compatible with the common Aster ranking evaluator.
    """

    name = "pairwise_logistic_ranking"

    def __init__(self, *, c: float = 1.0, seed: int = 7):
        if c <= 0:
            raise ValueError("c must be positive")
        self.c = c
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = LogisticRegression(C=c, random_state=seed, max_iter=2000)
        self._fitted = False
        self.training_pairs = 0

    def fit(self, examples: list[TrainingExample]) -> "PairwiseLogisticRanker":
        grouped: dict[tuple[str, str, str], list[TrainingExample]] = defaultdict(list)
        for example in examples:
            if example.runtime_ms <= 0:
                raise ValueError("runtime labels must be positive")
            grouped[_query_key(example)].append(example)
        if not grouped:
            raise ValueError("training examples are required")

        all_rows = [baseline_feature_dict(example.plan) for example in examples]
        transformed = self.vectorizer.fit_transform(all_rows)
        vectors = {id(example): transformed[index] for index, example in enumerate(examples)}
        x_pairs: list[np.ndarray] = []
        y_pairs: list[int] = []
        logical_pairs = 0
        for candidates in grouped.values():
            for left, right in combinations(candidates, 2):
                if math.isclose(left.runtime_ms, right.runtime_ms, rel_tol=1e-12, abs_tol=1e-12):
                    continue
                diff = vectors[id(left)] - vectors[id(right)]
                label = int(left.runtime_ms > right.runtime_ms)
                x_pairs.extend((diff, -diff))
                y_pairs.extend((label, 1 - label))
                logical_pairs += 1
        if logical_pairs < 1:
            raise ValueError("at least one non-tied within-query plan pair is required")
        self.model.fit(np.asarray(x_pairs), np.asarray(y_pairs))
        self.training_pairs = logical_pairs
        self._fitted = True
        return self

    def score(self, plan: PlanDocument) -> float:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        x = self.vectorizer.transform([baseline_feature_dict(plan)])
        return float(self.model.decision_function(x)[0])
