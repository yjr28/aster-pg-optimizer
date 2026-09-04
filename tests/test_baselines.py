from aster.experiments import evaluate_ranking
from aster.models import PostgresCostRanker, RidgeRuntimeModel, TrainingExample
from aster.plans import parse_explain_json


def plan(cost: float, rows: float, operator: str = "Seq Scan"):
    return parse_explain_json({
        "Plan": {
            "Node Type": operator,
            "Total Cost": cost,
            "Plan Rows": rows,
            "Plan Width": 8,
        }
    })


def test_postgres_cost_ranker_uses_cost_units_without_calling_them_runtime():
    ranker = PostgresCostRanker()
    assert ranker.score(plan(12.5, 100)) == 12.5


def test_ridge_runtime_model_learns_monotonic_fixture_and_ranks_measured_candidates():
    train = [
        TrainingExample(plan(10, 100, "Index Scan"), 2.0, "tr1"),
        TrainingExample(plan(20, 200, "Index Scan"), 4.0, "tr2"),
        TrainingExample(plan(40, 500, "Bitmap Heap Scan"), 8.0, "tr3"),
        TrainingExample(plan(80, 1000), 16.0, "tr4"),
        TrainingExample(plan(160, 4000), 32.0, "tr5"),
        TrainingExample(plan(240, 10000), 48.0, "tr6"),
    ]
    model = RidgeRuntimeModel(alpha=1.0).fit(train)
    assert model.predict_runtime_ms(plan(15, 150, "Index Scan")) < model.predict_runtime_ms(plan(200, 7000))

    held_out = [
        TrainingExample(plan(120, 3000), 24.0, "q1", "native"),
        TrainingExample(plan(25, 250, "Index Scan"), 6.0, "q1", "alt"),
        TrainingExample(plan(180, 5000), 36.0, "q2", "native"),
        TrainingExample(plan(30, 300, "Index Scan"), 7.0, "q2", "alt"),
    ]
    metrics = evaluate_ranking(model, held_out)
    assert metrics.queries == 2
    assert metrics.geometric_mean_speedup_vs_native > 1.0
