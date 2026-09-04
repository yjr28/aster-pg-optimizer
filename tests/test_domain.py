from aster.features.baseline import baseline_feature_dict
from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json
from aster.ranking.risk import fallback_reason
from aster.uncertainty import FeatureDomain


def plan(cost, rows, op="Seq Scan"):
    return parse_explain_json({"Plan": {"Node Type": op, "Total Cost": cost, "Plan Rows": rows, "Plan Width": 8}})


def training():
    return [TrainingExample(plan(10,100,"Index Scan"),2,"q1"), TrainingExample(plan(20,200,"Index Scan"),4,"q2"),
            TrainingExample(plan(60,1000),10,"q3"), TrainingExample(plan(100,3000),20,"q4"),
            TrainingExample(plan(160,8000),35,"q5"), TrainingExample(plan(240,15000),55,"q6")]


def test_feature_domain_flags_unseen_operator_and_extreme_continuous_shift():
    domain = FeatureDomain.fit([baseline_feature_dict(e.plan) for e in training()])
    normal = domain.assess(baseline_feature_dict(plan(80,2000)))
    shifted = domain.assess(baseline_feature_dict(plan(100000,1_000_000,"Gather Merge")))
    assert "op=Gather Merge" in shifted.unseen_structural_features
    assert shifted.rms_z_distance > normal.rms_z_distance and shifted.outside_training_range_count > 0


def test_fallback_rejects_unseen_plan_structure_even_with_permissive_variance_threshold():
    model = RuntimeEnsemble(trees=32,min_samples_leaf=1,seed=1).fit(training())
    reason = fallback_reason(model, best_prediction=model.predict(plan(30,300,"Gather Merge")),
                             native_prediction=model.predict(plan(100,3000)), best_is_native=False,
                             max_log_std=100,min_predicted_gain=0,domain_margin=100,
                             max_domain_distance=100,max_outside_features=100)
    assert reason == "unseen_plan_structure"
