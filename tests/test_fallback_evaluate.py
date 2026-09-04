from aster.experiments import evaluate_fallback_policy, fallback_pareto_sweep
from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json


def plan(cost, rows, operator):
    return parse_explain_json({
        "Plan": {
            "Node Type": operator,
            "Total Cost": cost,
            "Plan Rows": rows,
            "Plan Width": 8,
        }
    })


def fitted_model():
    points = [
        (10, 100, "Index Scan", 2),
        (20, 200, "Index Scan", 4),
        (40, 400, "Bitmap Heap Scan", 7),
        (80, 1000, "Seq Scan", 15),
        (120, 3000, "Seq Scan", 25),
        (180, 7000, "Seq Scan", 38),
        (240, 14000, "Seq Scan", 55),
        (320, 25000, "Seq Scan", 80),
    ]
    return RuntimeEnsemble(trees=64, min_samples_leaf=1, seed=9).fit([
        TrainingExample(plan(c, r, op), runtime, f"tr{i}")
        for i, (c, r, op, runtime) in enumerate(points)
    ])


def held_out():
    return [
        TrainingExample(plan(140, 5000, "Seq Scan"), 30, "q1", "native"),
        TrainingExample(plan(25, 250, "Index Scan"), 6, "q1", "alt"),
        TrainingExample(plan(200, 9000, "Seq Scan"), 45, "q2", "native"),
        TrainingExample(plan(35, 350, "Index Scan"), 8, "q2", "alt"),
    ]


def test_offline_fallback_metrics_use_measured_runtime_and_report_fallback_rate():
    metrics = evaluate_fallback_policy(
        fitted_model(),
        held_out(),
        max_log_std=10.0,
        min_predicted_gain=0.0,
        domain_margin=10.0,
    )
    assert metrics.queries == 2
    assert 0.0 <= metrics.fallback_fraction <= 1.0
    assert metrics.oracle_geometric_mean_speedup >= metrics.geometric_mean_speedup_vs_native


def test_pareto_sweep_varies_policy_thresholds_without_changing_query_count():
    points = fallback_pareto_sweep(
        fitted_model(),
        held_out(),
        max_log_stds=(0.0, 10.0),
        min_predicted_gains=(0.0, 0.99),
        domain_margin=10.0,
    )
    assert len(points) == 4
    assert {point.metrics.queries for point in points} == {2}
    assert max(point.metrics.fallback_fraction for point in points) >= min(
        point.metrics.fallback_fraction for point in points
    )
