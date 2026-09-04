import pytest

from aster.models import MLPRuntimeModel, TrainingExample
from aster.plans import parse_explain_json


def _plan(cost, rows, node="Seq Scan"):
    return parse_explain_json({"Plan":{"Node Type":node,"Total Cost":cost,"Plan Rows":rows,"Plan Width":8}})


def test_mlp_runtime_baseline_fits_positive_runtime_and_is_deterministic():
    examples=[
        TrainingExample(_plan(10,100),2.0,"q1"),
        TrainingExample(_plan(20,200),4.0,"q2"),
        TrainingExample(_plan(40,400),8.0,"q3"),
        TrainingExample(_plan(80,800),16.0,"q4"),
        TrainingExample(_plan(160,1600,"Index Scan"),32.0,"q5"),
        TrainingExample(_plan(320,3200,"Index Scan"),64.0,"q6"),
    ]
    first=MLPRuntimeModel(hidden_layer_sizes=(8,),seed=3,max_iter=200).fit(examples)
    second=MLPRuntimeModel(hidden_layer_sizes=(8,),seed=3,max_iter=200).fit(examples)
    prediction=first.predict_runtime_ms(_plan(60,600))
    assert prediction > 0
    assert prediction == pytest.approx(second.predict_runtime_ms(_plan(60,600)),rel=1e-12)


def test_mlp_rejects_invalid_configuration_and_labels():
    with pytest.raises(ValueError,match="positive widths"):
        MLPRuntimeModel(hidden_layer_sizes=(0,))
    with pytest.raises(ValueError,match="at least 4"):
        MLPRuntimeModel(hidden_layer_sizes=(4,)).fit([
            TrainingExample(_plan(1,1),1.0,"q1"),
        ])
