from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json


def plan(cost, rows):
    return parse_explain_json({"Plan":{"Node Type":"Seq Scan","Total Cost":cost,"Plan Rows":rows,"Plan Width":8}})


def test_runtime_ensemble_calibration_attaches_intervals_without_changing_point_prediction():
    train=[
        TrainingExample(plan(10,100),2.0,"q1"),
        TrainingExample(plan(20,200),4.0,"q2"),
        TrainingExample(plan(40,400),8.0,"q3"),
        TrainingExample(plan(80,800),16.0,"q4"),
        TrainingExample(plan(160,1600),32.0,"q5"),
        TrainingExample(plan(320,3200),64.0,"q6"),
    ]
    calibration=[
        TrainingExample(plan(30,300),7.0,"c1"),
        TrainingExample(plan(120,1200),28.0,"c2"),
    ]
    model=RuntimeEnsemble(trees=32,seed=2,min_samples_leaf=1).fit(train)
    before=model.predict(plan(60,600))
    assert before.interval_lower_ms is None
    model.calibrate(calibration,alpha=0.2,min_log_scale=0.05)
    after=model.predict(plan(60,600))
    assert after.runtime_ms == before.runtime_ms
    assert after.interval_lower_ms is not None
    assert after.interval_upper_ms is not None
    assert after.interval_lower_ms <= after.runtime_ms <= after.interval_upper_ms
    assert after.calibrated_log_radius is not None
