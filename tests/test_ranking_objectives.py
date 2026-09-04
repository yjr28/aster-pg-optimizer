from aster.models import PairwiseLogisticRanker, QueryNormalizedRidgeModel, TrainingExample
from aster.plans import parse_explain_json


def _plan(cost, rows=100):
    return parse_explain_json([{"Plan":{"Node Type":"Seq Scan","Relation Name":"t","Total Cost":cost,"Plan Rows":rows,"Plan Width":8}}])


def _examples():
    examples=[]
    # Two query groups with the same ordering but very different absolute scale.
    for query_id, scale in (("q1",1.0),("q2",100.0)):
        examples.extend([
            TrainingExample(_plan(10),10*scale,query_id,"native",query_template=query_id,workload="unit",dataset_version="v1"),
            TrainingExample(_plan(5),5*scale,query_id,"fast",query_template=query_id,workload="unit",dataset_version="v1"),
            TrainingExample(_plan(20),20*scale,query_id,"slow",query_template=query_id,workload="unit",dataset_version="v1"),
        ])
    return examples


def test_query_normalized_ridge_learns_relative_candidate_order():
    model=QueryNormalizedRidgeModel(alpha=0.01).fit(_examples())
    assert model.score(_plan(5)) < model.score(_plan(10)) < model.score(_plan(20))


def test_pairwise_ranker_uses_within_query_differences_and_balanced_orientations():
    model=PairwiseLogisticRanker(c=10.0,seed=7).fit(_examples())
    assert model.training_pairs == 6
    assert model.score(_plan(5)) < model.score(_plan(10)) < model.score(_plan(20))


def test_query_normalized_model_requires_native_training_reference():
    examples=[TrainingExample(_plan(5),5,"q1","alt",workload="unit",dataset_version="v1"),TrainingExample(_plan(6),6,"q1","alt2",workload="unit",dataset_version="v1")]
    try:
        QueryNormalizedRidgeModel().fit(examples)
    except ValueError as exc:
        assert "no native candidate" in str(exc)
    else:
        raise AssertionError("expected missing-native training group to be rejected")
