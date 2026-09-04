from __future__ import annotations

import random
from dataclasses import dataclass

from aster.models import TrainingExample


@dataclass(frozen=True)
class HoldoutSplit:
    train: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]
    train_groups: frozenset[str]
    test_groups: frozenset[str]


def template_holdout(
    examples: list[TrainingExample],
    *,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> HoldoutSplit:
    """Split entire query templates, never individual plans, across train/test."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.query_template]
    if missing:
        raise ValueError(f"query_template required for template holdout; missing for {missing[:3]}")
    groups = sorted({e.query_template for e in examples if e.query_template})
    if len(groups) < 2:
        raise ValueError("at least two query templates are required")
    rng = random.Random(seed)
    rng.shuffle(groups)
    test_count = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
    test_groups = frozenset(groups[:test_count])
    train_groups = frozenset(groups[test_count:])
    train = tuple(e for e in examples if e.query_template in train_groups)
    test = tuple(e for e in examples if e.query_template in test_groups)
    return HoldoutSplit(train, test, train_groups, test_groups)
