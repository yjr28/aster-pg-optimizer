import json

import pytest

from aster.plans import canonical_plan, parse_explain_json, plan_fingerprint


SAMPLE = [
    {
        "Plan": {
            "Node Type": "Hash Join",
            "Join Type": "Inner",
            "Startup Cost": 10.0,
            "Total Cost": 20.0,
            "Plan Rows": 3,
            "Plan Width": 16,
            "Actual Rows": 2,
            "Actual Total Time": 1.2,
            "Actual Loops": 1,
            "Hash Cond": "(a.id = b.id)",
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "a",
                    "Alias": "a",
                    "Total Cost": 8,
                    "Plan Rows": 100,
                    "Plan Width": 8,
                    "Shared Hit Blocks": 4,
                },
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "b",
                    "Index Name": "b_pkey",
                    "Total Cost": 4,
                    "Plan Rows": 3,
                    "Plan Width": 8,
                },
            ],
        },
        "Planning Time": 0.22,
        "Execution Time": 1.31,
        "Settings": [{"Name": "enable_hashjoin", "Setting": "on"}],
    }
]


def test_parse_plan_tree_and_runtime_fields():
    plan = parse_explain_json(json.dumps(SAMPLE))
    assert plan.execution_time_ms == pytest.approx(1.31)
    assert plan.settings == {"enable_hashjoin": "on"}
    assert plan.root.node_type == "Hash Join"
    assert [c.node_type for c in plan.root.children] == ["Seq Scan", "Index Scan"]
    assert list(plan.root.walk())[1][0] == 1


def test_canonical_identity_ignores_runtime_and_cost_changes():
    left = parse_explain_json(SAMPLE)
    mutated = json.loads(json.dumps(SAMPLE))
    mutated[0]["Plan"]["Total Cost"] = 9999
    mutated[0]["Plan"]["Actual Total Time"] = 9999
    mutated[0]["Execution Time"] = 9999
    right = parse_explain_json(mutated)
    assert canonical_plan(left) == canonical_plan(right)
    assert plan_fingerprint(left) == plan_fingerprint(right)


def test_canonical_identity_changes_with_physical_path():
    left = parse_explain_json(SAMPLE)
    mutated = json.loads(json.dumps(SAMPLE))
    mutated[0]["Plan"]["Plans"][1]["Node Type"] = "Seq Scan"
    mutated[0]["Plan"]["Plans"][1].pop("Index Name")
    right = parse_explain_json(mutated)
    assert plan_fingerprint(left) != plan_fingerprint(right)


def test_rejects_non_explain_payload():
    with pytest.raises(ValueError):
        parse_explain_json({"hello": "world"})
