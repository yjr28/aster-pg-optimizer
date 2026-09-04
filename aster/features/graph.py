from __future__ import annotations

import math
from dataclasses import dataclass

from aster.plans.types import PlanDocument


@dataclass(frozen=True)
class PlanGraphNode:
    node_id: int
    depth: int
    operator: str
    relation: str | None
    index: str | None
    join_type: str | None
    numeric: tuple[float, ...]


@dataclass(frozen=True)
class PlanGraph:
    nodes: tuple[PlanGraphNode, ...]
    edges: tuple[tuple[int, int], ...]


def _log1p_nonnegative(value: float | None) -> float:
    return math.log1p(max(0.0, float(value or 0.0)))


def build_plan_graph(plan: PlanDocument) -> PlanGraph:
    nodes: list[PlanGraphNode] = []
    edges: list[tuple[int, int]] = []

    def visit(node, depth: int, parent_id: int | None) -> None:
        node_id = len(nodes)
        nodes.append(
            PlanGraphNode(
                node_id=node_id,
                depth=depth,
                operator=node.node_type,
                relation=node.relation_name,
                index=node.index_name,
                join_type=node.join_type,
                numeric=(
                    _log1p_nonnegative(node.startup_cost),
                    _log1p_nonnegative(node.total_cost),
                    _log1p_nonnegative(node.plan_rows),
                    _log1p_nonnegative(node.plan_width),
                    _log1p_nonnegative(node.shared_hit_blocks),
                    _log1p_nonnegative(node.shared_read_blocks),
                    _log1p_nonnegative(node.temp_read_blocks),
                    _log1p_nonnegative(node.temp_written_blocks),
                ),
            )
        )
        if parent_id is not None:
            edges.append((parent_id, node_id))
        for child in node.children:
            visit(child, depth + 1, node_id)

    visit(plan.root, 0, None)
    return PlanGraph(nodes=tuple(nodes), edges=tuple(edges))
