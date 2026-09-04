import pytest

from aster.experiments import (
    dataset_version_holdout,
    parameter_holdout,
    query_holdout,
    relation_holdout,
    template_holdout,
    workload_holdout,
)
from aster.models import TrainingExample
from aster.plans import parse_explain_json


def example(
    query_id,
    template,
    *,
    parameter=None,
    workload=None,
    candidate="native",
    dataset="v1",
    relation=None,
):
    raw_plan={"Node Type":"Seq Scan"}
    if relation:
        raw_plan["Relation Name"]=relation
    return TrainingExample(
        parse_explain_json({"Plan": raw_plan}),
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


def test_dataset_version_holdout_has_zero_snapshot_leakage_and_keeps_candidates_together():
    examples=[]
    for dataset in ("job-imdb-v1","job-imdb-v2"):
        for query_id in ("q1","q2"):
            for candidate in ("native","alt"):
                examples.append(example(
                    query_id,
                    f"template-{query_id}",
                    workload="job",
                    candidate=candidate,
                    dataset=dataset,
                ))
    split=dataset_version_holdout(examples,test_fraction=0.5,seed=5)
    assert split.train_groups.isdisjoint(split.test_groups)
    assert {e.dataset_version for e in split.train} == split.train_groups
    assert {e.dataset_version for e in split.test} == split.test_groups
    for dataset in split.test_groups:
        for query_id in ("q1","q2"):
            assert {
                e.candidate_id for e in split.test
                if e.dataset_version==dataset and e.query_id==query_id
            } == {"native","alt"}


def test_relation_holdout_removes_held_relation_from_every_training_plan():
    examples=[]
    for query_id,relation in (("q1","rare_table"),("q2","common_table"),("q3","common_table"),("q4","common_table"),("q5","common_table")):
        for candidate in ("native","alt"):
            examples.append(example(
                query_id,
                f"template-{query_id}",
                workload="job",
                candidate=candidate,
                relation=relation,
            ))
    split=relation_holdout(examples,test_fraction=0.2,seed=7)
    assert split.notes == ("heldout_relation=rare_table",)
    assert {e.query_id for e in split.test} == {"q1"}
    assert {e.candidate_id for e in split.test} == {"native","alt"}
    train_relations={
        node.relation_name
        for example_ in split.train
        for _depth,node in example_.plan.root.walk()
        if node.relation_name
    }
    assert "rare_table" not in train_relations


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
