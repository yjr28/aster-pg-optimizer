from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class PlanNode:
    node_type: str
    startup_cost: float = 0.0
    total_cost: float = 0.0
    plan_rows: float = 0.0
    plan_width: float = 0.0
    actual_rows: float | None = None
    actual_total_time_ms: float | None = None
    actual_loops: float | None = None
    relation_name: str | None = None
    alias: str | None = None
    index_name: str | None = None
    join_type: str | None = None
    parent_relationship: str | None = None
    filter_text: str | None = None
    index_cond: str | None = None
    hash_cond: str | None = None
    merge_cond: str | None = None
    sort_key: tuple[str, ...] = ()
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0
    temp_read_blocks: int = 0
    temp_written_blocks: int = 0
    children: tuple["PlanNode", ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def walk(self) -> Iterator[tuple[int, "PlanNode"]]:
        stack: list[tuple[int, PlanNode]] = [(0, self)]
        while stack:
            depth, node = stack.pop()
            yield depth, node
            for child in reversed(node.children):
                stack.append((depth + 1, child))


@dataclass(frozen=True)
class PlanDocument:
    root: PlanNode
    planning_time_ms: float | None = None
    execution_time_ms: float | None = None
    settings: dict[str, str] = field(default_factory=dict)
    query_identifier: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)
