from aster.experiments import evaluate_runtime_ranking
from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json


def plan(cost, op):
    return parse_explain_json({"Plan": {"Node Type": op, "Total Cost": cost, "Plan Rows": cost * 10}})


def test_evaluation_uses_actual_selected_runtime_not_prediction_metric():
    train = [
        TrainingExample(plan(10, "Index Scan"), 2, "tr1"),
        TrainingExample(plan(20, "Index Scan"), 4, "tr2"),
        TrainingExample(plan(80, "Seq Scan"), 16, "tr3"),
        TrainingExample(plan(160, "Seq Scan"), 32, "tr4"),
        TrainingExample(plan(240, "Seq Scan"), 50, "tr5"),
    ]
    model = RuntimeEnsemble(trees=48, min_samples_leaf=1, seed=1).fit(train)
    test = [
        TrainingExample(plan(100, "Seq Scan"), 20, "q1", "native"),
        TrainingExample(plan(15, "Index Scan"), 5, "q1", "alt"),
        TrainingExample(plan(120, "Seq Scan"), 24, "q2", "native"),
        TrainingExample(plan(20, "Index Scan"), 6, "q2", "alt"),
    ]
    metrics = evaluate_runtime_ranking(model, test)
    assert metrics.queries == 2
    assert metrics.geometric_mean_speedup_vs_native > 1
    assert metrics.oracle_geometric_mean_speedup >= metrics.geometric_mean_speedup_vs_native
