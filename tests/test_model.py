from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json


def make_plan(cost, rows, node_type="Seq Scan"):
    return parse_explain_json([
        {
            "Plan": {
                "Node Type": node_type,
                "Total Cost": cost,
                "Plan Rows": rows,
                "Plan Width": 16,
            }
        }
    ])


def test_runtime_ensemble_learns_basic_runtime_order_and_uncertainty():
    examples = [
        TrainingExample(make_plan(10, 100), 3.0, "q1"),
        TrainingExample(make_plan(20, 200), 5.0, "q2"),
        TrainingExample(make_plan(80, 1000), 15.0, "q3"),
        TrainingExample(make_plan(120, 3000), 25.0, "q4"),
        TrainingExample(make_plan(200, 9000), 45.0, "q5"),
        TrainingExample(make_plan(300, 20000), 70.0, "q6"),
    ]
    model = RuntimeEnsemble(trees=32, seed=1, min_samples_leaf=1).fit(examples)
    fast = model.predict(make_plan(15, 120))
    slow = model.predict(make_plan(250, 12000))
    assert fast.runtime_ms < slow.runtime_ms
    assert fast.log_std >= 0
    assert model.in_training_cost_domain(fast)
