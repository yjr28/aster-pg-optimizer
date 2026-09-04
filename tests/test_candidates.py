import copy

import pytest

from aster.candidates import CandidateCollector, CandidateSpec


BASE_PLAN = [
    {
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "orders",
            "Total Cost": 10,
            "Plan Rows": 100,
        },
        "Planning Time": 0.1,
    }
]


class FakeRunner:
    def __init__(self):
        self.calls = []

    def postgres_version(self):
        return "17.4"

    def explain(self, query, settings, *, analyze):
        self.calls.append((query, dict(settings), analyze))
        plan = copy.deepcopy(BASE_PLAN)
        if settings.get("enable_seqscan") == "off":
            plan[0]["Plan"].update({"Node Type": "Index Scan", "Index Name": "orders_idx"})
        if analyze:
            plan[0]["Execution Time"] = 4.25 if settings.get("enable_seqscan") == "off" else 9.5
            plan[0]["Plan"]["Actual Rows"] = 100
            plan[0]["Plan"]["Actual Loops"] = 1
        return plan


def test_discover_deduplicates_equivalent_candidate_plans():
    runner = FakeRunner()
    collector = CandidateCollector(runner)
    candidates = [
        CandidateSpec("native", {}),
        CandidateSpec("no_hash", {"enable_hashjoin": "off"}),  # same plan in fake runner
        CandidateSpec("no_seq", {"enable_seqscan": "off"}),
    ]
    discovered = collector.discover("select * from orders", candidates)
    assert [c.spec.candidate_id for c in discovered] == ["native", "no_seq"]
    assert all(call[2] is False for call in runner.calls)


def test_measure_warmups_are_not_persisted_and_plan_identity_is_checked():
    runner = FakeRunner()
    collector = CandidateCollector(runner)
    candidate = collector.discover(
        "select * from orders", [CandidateSpec("no_seq", {"enable_seqscan": "off"})]
    )[0]
    observations = collector.measure(
        "select * from orders",
        candidate,
        workload="unit",
        query_id="q1",
        dataset_version="v1",
        run_seed=7,
        code_revision="abc",
        warmups=2,
        repetitions=3,
    )
    assert len(observations) == 3
    assert [o.repetition for o in observations] == [0, 1, 2]
    assert all(o.execution_time_ms == pytest.approx(4.25) for o in observations)
    analyze_calls = [c for c in runner.calls if c[2]]
    assert len(analyze_calls) == 5


def test_candidate_spec_rejects_arbitrary_guc_injection():
    with pytest.raises(ValueError):
        CandidateSpec("bad", {"search_path": "off"})
