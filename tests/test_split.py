import pytest

from aster.experiments import parameter_holdout, query_holdout, template_holdout, workload_holdout
from aster.models import TrainingExample
from aster.plans import parse_explain_json


def example(query_id, template, *, parameter=None, workload=None, candidate="native", dataset="v1"):
    return TrainingExample(
        parse_explain_json({"Plan": {"Node Type": "Seq Scan"}}),
        1.0,
        query_id,
        candidate,
        template,
        parameter,
        workload,
        dataset,
    )


def test_template_holdout_has_zero_template_leakage():
    examples = [
        example("q1a", "t1"), example("q1b", "t1"),
        example("q2a", "t2"), example("q2b", "t2"),
        example("q3a", "t3"), example("q3b", "t3"),
    ]
    split = template_holdout(examples, test_fraction=1/3, seed=4)
    assert split.train_groups.isdisjoint(split.test_groups)
    assert {e.query_template for e in split.train} == split.train_groups
    assert {e.query_template for e in split.test} == split.test_groups


def test_parameter_holdout_keeps_each_template_in_train_and_holds_complete_parameter_groups():
    examples=[]
    for template in ("t1","t2"):
        for parameter in ("a","b","c"):
            for candidate in ("native","alt"):
                examples.append(example(
                    f"{template}-{parameter}",template,parameter=parameter,
                    workload="job",candidate=candidate,
                ))
    split=parameter_holdout(examples,test_fraction=1/3,seed=3)
    assert {e.query_template for e in split.train} == {"t1","t2"}
    assert {e.query_template for e in split.test} == {"t1","t2"}
    train_pairs={(e.query_template,e.parameter_key) for e in split.train}
    test_pairs={(e.query_template,e.parameter_key) for e in split.test}
    assert train_pairs.isdisjoint(test_pairs)
    for pair in test_pairs:
        assert {e.candidate_id for e in split.test if (e.query_template,e.parameter_key)==pair} == {"native","alt"}


def test_parameter_holdout_rejects_single_parameter_templates():
    examples=[example("q1","t1",parameter="a"),example("q2","t2",parameter="a"),example("q3","t2",parameter="b")]
    with pytest.raises(ValueError,match="at least two parameter keys"):
        parameter_holdout(examples)


def test_workload_holdout_has_zero_workload_leakage():
    examples=[
        example("q1","t1",workload="job"),
        example("q2","t2",workload="job"),
        example("q3","t3",workload="tpch"),
        example("q4","t4",workload="tpch"),
    ]
    split=workload_holdout(examples,test_fraction=0.5,seed=1)
    assert split.train_groups.isdisjoint(split.test_groups)
    assert {e.workload for e in split.train} == split.train_groups
    assert {e.workload for e in split.test} == split.test_groups


def test_query_holdout_keeps_all_candidates_for_query_together():
    examples=[]
    for query_id in ("q1","q2","q3","q4"):
        examples.extend([
            example(query_id,"t",parameter=query_id,workload="job",candidate="native"),
            example(query_id,"t",parameter=query_id,workload="job",candidate="alt"),
        ])
    split=query_holdout(examples,test_fraction=0.25,seed=2)
    train_queries={e.query_id for e in split.train}
    test_queries={e.query_id for e in split.test}
    assert train_queries.isdisjoint(test_queries)
    assert len(test_queries)==1
    for query_id in test_queries:
        assert {e.candidate_id for e in split.test if e.query_id==query_id} == {"native","alt"}
