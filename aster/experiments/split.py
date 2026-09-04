from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from aster.models import TrainingExample


@dataclass(frozen=True)
class HoldoutSplit:
    train: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]
    train_groups: frozenset[str]
    test_groups: frozenset[str]


def _test_count(group_count: int, test_fraction: float) -> int:
    return max(1, min(group_count - 1, round(group_count * test_fraction)))


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
    test_count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:test_count])
    train_groups = frozenset(groups[test_count:])
    train = tuple(e for e in examples if e.query_template in train_groups)
    test = tuple(e for e in examples if e.query_template in test_groups)
    return HoldoutSplit(train, test, train_groups, test_groups)


def parameter_holdout(
    examples: list[TrainingExample],
    *,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> HoldoutSplit:
    """Hold out parameterizations while keeping every template represented in train.

    The split unit is (query_template, parameter_key), so every candidate plan for one
    parameterization stays together. Every template must expose at least two distinct
    parameter keys; otherwise a within-template generalization test is undefined.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    missing_template = [e.query_id for e in examples if not e.query_template]
    missing_parameter = [e.query_id for e in examples if not e.parameter_key]
    if missing_template:
        raise ValueError(f"query_template required for parameter holdout; missing for {missing_template[:3]}")
    if missing_parameter:
        raise ValueError(f"parameter_key required for parameter holdout; missing for {missing_parameter[:3]}")

    by_template: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        by_template[str(example.query_template)].add(str(example.parameter_key))
    insufficient = sorted(template for template, keys in by_template.items() if len(keys) < 2)
    if insufficient:
        raise ValueError(
            "parameter holdout requires at least two parameter keys per template; "
            f"insufficient={insufficient[:3]}"
        )

    rng = random.Random(seed)
    test_pairs: set[tuple[str, str]] = set()
    train_pairs: set[tuple[str, str]] = set()
    for template in sorted(by_template):
        keys = sorted(by_template[template])
        rng.shuffle(keys)
        count = _test_count(len(keys), test_fraction)
        test_pairs.update((template, key) for key in keys[:count])
        train_pairs.update((template, key) for key in keys[count:])

    def pair(example: TrainingExample) -> tuple[str, str]:
        return str(example.query_template), str(example.parameter_key)

    train = tuple(example for example in examples if pair(example) in train_pairs)
    test = tuple(example for example in examples if pair(example) in test_pairs)
    train_groups = frozenset(f"{template}::{key}" for template, key in train_pairs)
    test_groups = frozenset(f"{template}::{key}" for template, key in test_pairs)
    return HoldoutSplit(train, test, train_groups, test_groups)


def workload_holdout(
    examples: list[TrainingExample],
    *,
    test_fraction: float = 0.2,
    seed: int = 7,
) -> HoldoutSplit:
    """Hold out entire named workloads to measure cross-workload generalization."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.workload]
    if missing:
        raise ValueError(f"workload required for workload holdout; missing for {missing[:3]}")
    groups = sorted({str(e.workload) for e in examples if e.workload})
    if len(groups) < 2:
        raise ValueError("at least two workloads are required for workload holdout")
    rng = random.Random(seed)
    rng.shuffle(groups)
    count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count])
    train_groups = frozenset(groups[count:])
    train = tuple(e for e in examples if str(e.workload) in train_groups)
    test = tuple(e for e in examples if str(e.workload) in test_groups)
    return HoldoutSplit(train, test, train_groups, test_groups)
