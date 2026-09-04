from __future__ import annotations

import hashlib
import json
from typing import Any

from .types import PlanDocument, PlanNode


def _canonical_node(node: PlanNode) -> dict[str, Any]:
    """Return structural plan identity, deliberately excluding estimates/runtimes.

    Cost/cardinality estimates are model features but not plan identity: the same
    physical plan under changed statistics should deduplicate structurally while its
    feature vector can still change.
    """
    return {
        "node_type": node.node_type,
        "relation": node.relation_name,
        "index": node.index_name,
        "join_type": node.join_type,
        "parent_relationship": node.parent_relationship,
        "sort_key": list(node.sort_key),
        "children": [_canonical_node(child) for child in node.children],
    }


def canonical_plan(plan: PlanDocument | PlanNode) -> dict[str, Any]:
    root = plan.root if isinstance(plan, PlanDocument) else plan
    return _canonical_node(root)


def plan_fingerprint(plan: PlanDocument | PlanNode) -> str:
    encoded = json.dumps(canonical_plan(plan), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
