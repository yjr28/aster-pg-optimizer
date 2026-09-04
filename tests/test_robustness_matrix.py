from aster.experiments import TrainingProtocol, run_robustness_matrix
from aster.models import TrainingExample
from aster.plans import parse_explain_json


def _plan(cost, node):
    return parse_explain_json({"Plan":{"Node Type":node,"Total Cost":cost,"Plan Rows":100,"Plan Width":8}})


def _examples():
    rows=[]
    query_index=0
    for workload in ("job","tpch"):
        for template_index in range(3):
            template=f"{workload}-t{template_index}"
            parameter="only-parameter"
            for local in range(2):
                query_id=f"{workload}-q{query_index}"
                query_index += 1
                rows.append(TrainingExample(
                    _plan(100,"Seq Scan"),20.0+local,query_id,"native",template,
                    parameter,workload,f"{workload}-v1",
                ))
                rows.append(TrainingExample(
                    _plan(40,"Index Scan"),10.0+local,query_id,"alt",template,
                    parameter,workload,f"{workload}-v1",
                ))
    return rows


def test_robustness_matrix_runs_template_and_workload_and_marks_parameter_unsupported():
    matrix=run_robustness_matrix(
        _examples(),
        TrainingProtocol(calibration_fraction=0,trees=16,min_samples_leaf=1,seed=3),
    )
    by_regime={result.regime:result for result in matrix.regimes}
    assert matrix.evaluation_kind == "offline_measured_plan_replay"
    assert matrix.workloads == ("job","tpch")
    assert by_regime["template"].status == "ok"
    assert by_regime["workload"].status == "ok"
    assert by_regime["parameter"].status == "unsupported_for_dataset"
    assert "at least two parameter keys" in by_regime["parameter"].reason
    assert by_regime["template"].metrics["fallback_metrics"]["queries"] > 0
