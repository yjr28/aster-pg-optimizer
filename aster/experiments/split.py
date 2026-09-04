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
    notes: tuple[str, ...] = ()


def _test_count(group_count: int, test_fraction: float) -> int:
    return max(1, min(group_count - 1, round(group_count * test_fraction)))


def _query_key(example: TrainingExample) -> str:
    return (
        f"{example.environment_sha256 or ''}::{example.dataset_version or ''}::"
        f"{example.workload or ''}::{example.query_id}"
    )


def template_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Split entire query templates, never individual plans, across train/test."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.query_template]
    if missing: raise ValueError(f"query_template required for template holdout; missing for {missing[:3]}")
    groups = sorted({e.query_template for e in examples if e.query_template})
    if len(groups) < 2: raise ValueError("at least two query templates are required")
    rng = random.Random(seed); rng.shuffle(groups)
    count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count]); train_groups = frozenset(groups[count:])
    return HoldoutSplit(
        tuple(e for e in examples if e.query_template in train_groups),
        tuple(e for e in examples if e.query_template in test_groups),
        train_groups, test_groups,
    )


def parameter_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Hold out parameterizations while keeping every template represented in train."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    missing_template = [e.query_id for e in examples if not e.query_template]
    missing_parameter = [e.query_id for e in examples if not e.parameter_key]
    if missing_template: raise ValueError(f"query_template required for parameter holdout; missing for {missing_template[:3]}")
    if missing_parameter: raise ValueError(f"parameter_key required for parameter holdout; missing for {missing_parameter[:3]}")
    by_template: dict[str, set[str]] = defaultdict(set)
    for example in examples: by_template[str(example.query_template)].add(str(example.parameter_key))
    insufficient = sorted(template for template, keys in by_template.items() if len(keys) < 2)
    if insufficient:
        raise ValueError("parameter holdout requires at least two parameter keys per template; " f"insufficient={insufficient[:3]}")
    rng = random.Random(seed); test_pairs: set[tuple[str, str]] = set(); train_pairs: set[tuple[str, str]] = set()
    for template in sorted(by_template):
        keys = sorted(by_template[template]); rng.shuffle(keys); count = _test_count(len(keys), test_fraction)
        test_pairs.update((template, key) for key in keys[:count]); train_pairs.update((template, key) for key in keys[count:])
    pair = lambda e: (str(e.query_template), str(e.parameter_key))
    train_groups = frozenset(f"{template}::{key}" for template, key in train_pairs)
    test_groups = frozenset(f"{template}::{key}" for template, key in test_pairs)
    return HoldoutSplit(
        tuple(e for e in examples if pair(e) in train_pairs),
        tuple(e for e in examples if pair(e) in test_pairs),
        train_groups, test_groups,
    )


def workload_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Hold out entire named workloads to measure cross-workload generalization."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.workload]
    if missing: raise ValueError(f"workload required for workload holdout; missing for {missing[:3]}")
    groups = sorted({str(e.workload) for e in examples if e.workload})
    if len(groups) < 2: raise ValueError("at least two workloads are required for workload holdout")
    rng = random.Random(seed); rng.shuffle(groups); count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count]); train_groups = frozenset(groups[count:])
    return HoldoutSplit(
        tuple(e for e in examples if str(e.workload) in train_groups),
        tuple(e for e in examples if str(e.workload) in test_groups),
        train_groups, test_groups,
    )


def dataset_version_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Hold out exact dataset snapshots/scales to measure data-version shift."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.dataset_version]
    if missing: raise ValueError(f"dataset_version required for dataset holdout; missing for {missing[:3]}")
    groups = sorted({str(e.dataset_version) for e in examples if e.dataset_version})
    if len(groups) < 2: raise ValueError("at least two dataset versions are required for dataset holdout")
    rng = random.Random(seed); rng.shuffle(groups); count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count]); train_groups = frozenset(groups[count:])
    return HoldoutSplit(
        tuple(e for e in examples if str(e.dataset_version) in train_groups),
        tuple(e for e in examples if str(e.dataset_version) in test_groups),
        train_groups, test_groups,
    )


def environment_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Hold out entire benchmark database/runtime snapshots.

    `environment_sha256` fingerprints PostgreSQL settings, relations/indexes/statistics
    state and host properties captured before collection. All examples measured under
    one fingerprint stay together, allowing clean stale-statistics/index/config/hardware
    shift studies without averaging labels across states.
    """
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    missing = [e.query_id for e in examples if not e.environment_sha256]
    if missing:
        raise ValueError(f"environment_sha256 required for environment holdout; missing for {missing[:3]}")
    groups = sorted({str(e.environment_sha256) for e in examples if e.environment_sha256})
    if len(groups) < 2:
        raise ValueError("at least two benchmark environments are required for environment holdout")
    rng = random.Random(seed); rng.shuffle(groups); count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count]); train_groups = frozenset(groups[count:])
    return HoldoutSplit(
        tuple(e for e in examples if str(e.environment_sha256) in train_groups),
        tuple(e for e in examples if str(e.environment_sha256) in test_groups),
        train_groups, test_groups,
    )


def _relations(example: TrainingExample) -> frozenset[str]:
    return frozenset(
        node.relation_name
        for _depth, node in example.plan.root.walk()
        if node.relation_name
    )


def relation_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.2, seed: int = 7) -> HoldoutSplit:
    """Hold out query groups containing one relation never seen in training plans.

    Candidate plans are first grouped by query. Relation membership for a query is the
    union across its candidates. A deterministic relation is chosen whose induced test
    fraction is closest to `test_fraction`, subject to non-empty train/test sets. Every
    training query is guaranteed not to reference the held-out relation.
    """
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    by_query: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples: by_query[_query_key(example)].append(example)
    if len(by_query) < 2: raise ValueError("at least two query groups are required for relation holdout")
    query_relations = {
        key: frozenset().union(*(_relations(example) for example in candidates))
        for key, candidates in by_query.items()
    }
    relation_names = sorted(set().union(*query_relations.values()))
    if not relation_names: raise ValueError("relation holdout requires plans with relation names")
    rng = random.Random(seed); rng.shuffle(relation_names)
    candidates: list[tuple[float, int, str, frozenset[str], frozenset[str]]] = []
    total = len(by_query)
    for order, relation in enumerate(relation_names):
        test_groups = frozenset(key for key, rels in query_relations.items() if relation in rels)
        train_groups = frozenset(key for key in by_query if key not in test_groups)
        if not train_groups or not test_groups: continue
        deviation = abs(len(test_groups) / total - test_fraction)
        candidates.append((deviation, order, relation, train_groups, test_groups))
    if not candidates:
        raise ValueError("no relation can be held out while leaving non-empty train and test query groups")
    _deviation, _order, relation, train_groups, test_groups = min(candidates)
    return HoldoutSplit(
        tuple(e for e in examples if _query_key(e) in train_groups),
        tuple(e for e in examples if _query_key(e) in test_groups),
        train_groups,
        test_groups,
        notes=(f"heldout_relation={relation}",),
    )


def query_holdout(examples: list[TrainingExample], *, test_fraction: float = 0.15, seed: int = 17) -> HoldoutSplit:
    """Split whole query candidate sets for calibration or secondary evaluation."""
    if not 0 < test_fraction < 1: raise ValueError("test_fraction must be between 0 and 1")
    groups = sorted({_query_key(e) for e in examples})
    if len(groups) < 2: raise ValueError("at least two query groups are required")
    rng = random.Random(seed); rng.shuffle(groups); count = _test_count(len(groups), test_fraction)
    test_groups = frozenset(groups[:count]); train_groups = frozenset(groups[count:])
    return HoldoutSplit(
        tuple(example for example in examples if _query_key(example) in train_groups),
        tuple(example for example in examples if _query_key(example) in test_groups),
        train_groups, test_groups,
    )
