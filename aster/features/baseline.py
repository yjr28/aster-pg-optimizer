from __future__ import annotations

import math
from collections import Counter
from typing import Any

from aster.plans.types import PlanDocument


def baseline_feature_dict(plan: PlanDocument) -> dict[str, Any]:
    """Leak-free baseline features derived only from planner-visible plan fields.

    Actual rows/times/buffers are deliberately excluded: they are labels/evaluation
    evidence and are unavailable when Aster ranks an unexecuted candidate.
    """
    op_counts: Counter[str] = Counter()
    join_counts: Counter[str] = Counter()
    depths: list[int] = []
    costs: list[float] = []
    rows: list[float] = []
    widths: list[float] = []

    for depth, node in plan.root.walk():
        depths.append(depth)
        costs.append(max(0.0, node.total_cost))
        rows.append(max(0.0, node.plan_rows))
        widths.append(max(0.0, node.plan_width))
        op_counts[node.node_type] += 1
        if node.join_type:
            join_counts[node.join_type] += 1

    features: dict[str, Any] = {
        "root_total_cost_log": math.log1p(max(0.0, plan.root.total_cost)),
        "root_rows_log": math.log1p(max(0.0, plan.root.plan_rows)),
        "root_width_log": math.log1p(max(0.0, plan.root.plan_width)),
        "node_count_log": math.log1p(len(depths)),
        "max_depth": float(max(depths, default=0)),
        "sum_cost_log": math.log1p(sum(costs)),
        "sum_rows_log": math.log1p(sum(rows)),
        "max_rows_log": math.log1p(max(rows, default=0.0)),
        "mean_width_log": math.log1p(sum(widths) / max(1, len(widths))),
    }
    for operator, count in op_counts.items():
        features[f"op={operator}"] = float(count)
    for join_type, count in join_counts.items():
        features[f"join={join_type}"] = float(count)
    return features
