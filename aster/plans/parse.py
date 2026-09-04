from __future__ import annotations

import json
from typing import Any

from .types import PlanDocument, PlanNode


def _f(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _i(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _node(obj: dict[str, Any]) -> PlanNode:
    children = tuple(_node(child) for child in obj.get("Plans", []))
    known = {
        "Node Type", "Startup Cost", "Total Cost", "Plan Rows", "Plan Width",
        "Actual Rows", "Actual Total Time", "Actual Loops", "Relation Name", "Alias",
        "Index Name", "Join Type", "Parent Relationship", "Filter", "Index Cond",
        "Hash Cond", "Merge Cond", "Sort Key", "Shared Hit Blocks", "Shared Read Blocks",
        "Temp Read Blocks", "Temp Written Blocks", "Plans",
    }
    return PlanNode(
        node_type=str(obj.get("Node Type", "Unknown")),
        startup_cost=float(obj.get("Startup Cost", 0.0)),
        total_cost=float(obj.get("Total Cost", 0.0)),
        plan_rows=float(obj.get("Plan Rows", 0.0)),
        plan_width=float(obj.get("Plan Width", 0.0)),
        actual_rows=_f(obj.get("Actual Rows")),
        actual_total_time_ms=_f(obj.get("Actual Total Time")),
        actual_loops=_f(obj.get("Actual Loops")),
        relation_name=obj.get("Relation Name"),
        alias=obj.get("Alias"),
        index_name=obj.get("Index Name"),
        join_type=obj.get("Join Type"),
        parent_relationship=obj.get("Parent Relationship"),
        filter_text=obj.get("Filter"),
        index_cond=obj.get("Index Cond"),
        hash_cond=obj.get("Hash Cond"),
        merge_cond=obj.get("Merge Cond"),
        sort_key=tuple(str(x) for x in obj.get("Sort Key", [])),
        shared_hit_blocks=_i(obj.get("Shared Hit Blocks")),
        shared_read_blocks=_i(obj.get("Shared Read Blocks")),
        temp_read_blocks=_i(obj.get("Temp Read Blocks")),
        temp_written_blocks=_i(obj.get("Temp Written Blocks")),
        children=children,
        extra={k: v for k, v in obj.items() if k not in known},
    )


def parse_explain_json(payload: str | bytes | list[Any] | dict[str, Any]) -> PlanDocument:
    """Parse PostgreSQL EXPLAIN (FORMAT JSON) output.

    PostgreSQL normally returns a one-element JSON array. Aster also accepts the inner
    object to make persisted artifacts easier to manipulate.
    """
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("expected PostgreSQL EXPLAIN JSON one-element array")
        doc = payload[0]
    elif isinstance(payload, dict):
        doc = payload
    else:
        raise TypeError("EXPLAIN payload must be JSON text, list, or object")

    plan = doc.get("Plan")
    if not isinstance(plan, dict):
        raise ValueError("EXPLAIN document has no Plan object")

    settings_raw = doc.get("Settings") or {}
    settings: dict[str, str] = {}
    if isinstance(settings_raw, dict):
        settings = {str(k): str(v) for k, v in settings_raw.items()}
    elif isinstance(settings_raw, list):
        for entry in settings_raw:
            if isinstance(entry, dict) and "Name" in entry and "Setting" in entry:
                settings[str(entry["Name"])] = str(entry["Setting"])

    qid = doc.get("Query Identifier")
    return PlanDocument(
        root=_node(plan),
        planning_time_ms=_f(doc.get("Planning Time")),
        execution_time_ms=_f(doc.get("Execution Time")),
        settings=settings,
        query_identifier=int(qid) if qid is not None else None,
        raw=doc,
    )
