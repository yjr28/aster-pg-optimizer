import copy

from aster.candidates import CandidateCollector, CandidateSpec


BASE = [{"Plan":{"Node Type":"Seq Scan","Relation Name":"t","Total Cost":1,"Plan Rows":1}}]


class Runner:
    def explain(self, query, settings, *, analyze):
        plan=copy.deepcopy(BASE)
        if settings.get("enable_seqscan") == "off":
            plan[0]["Plan"].update({"Node Type":"Index Scan","Index Name":"t_idx"})
        return plan
    def postgres_version(self):
        return "17"


def test_discovery_report_exposes_duplicate_interventions_without_counting_them_as_plans():
    report=CandidateCollector(Runner()).discover_report("select * from t",[
        CandidateSpec("native",{}),
        CandidateSpec("no_hash",{"enable_hashjoin":"off"}),
        CandidateSpec("no_seq",{"enable_seqscan":"off"}),
        CandidateSpec("no_merge",{"enable_mergejoin":"off"}),
    ])
    assert report.attempted_interventions == 4
    assert report.unique_plan_count == 2
    assert report.duplicate_interventions == 2
    assert report.uniqueness_ratio == 0.5
    groups=[set(group.candidate_ids) for group in report.plan_groups]
    assert {"native","no_hash","no_merge"} in groups
    assert {"no_seq"} in groups
    assert [candidate.spec.candidate_id for candidate in report.unique_candidates] == ["native","no_seq"]
