import math

from aster.features import build_plan_graph
from aster.plans import parse_explain_json


def test_build_plan_graph_preserves_tree_edges_and_structural_features():
    plan = parse_explain_json([
        {
            "Plan": {
                "Node Type": "Nested Loop",
                "Plan Rows": 10,
                "Total Cost": 50,
                "Plans": [
                    {"Node Type": "Seq Scan", "Relation Name": "orders", "Plan Rows": 1000},
                    {"Node Type": "Index Scan", "Relation Name": "items", "Index Name": "items_order_idx", "Plan Rows": 4},
                ],
            }
        }
    ])
    graph = build_plan_graph(plan)
    assert [n.operator for n in graph.nodes] == ["Nested Loop", "Seq Scan", "Index Scan"]
    assert graph.edges == ((0, 1), (0, 2))
    assert graph.nodes[1].relation == "orders"
    assert graph.nodes[2].index == "items_order_idx"
    assert graph.nodes[0].numeric[2] == math.log1p(10)
