from aster.experiments import template_holdout
from aster.models import TrainingExample
from aster.plans import parse_explain_json


def example(query_id, template):
    return TrainingExample(
        parse_explain_json({"Plan": {"Node Type": "Seq Scan"}}),
        1.0,
        query_id,
        "native",
        template,
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
