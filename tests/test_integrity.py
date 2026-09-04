import json

from aster.data import audit_dataset
from aster.plans import parse_explain_json, plan_fingerprint


def record(*, experiment="exp-1", query="q1", candidate="native", repetition=0, node="Seq Scan", fingerprint=None):
    plan = {"Plan": {"Node Type": node, "Relation Name": "orders", "Total Cost": 10, "Plan Rows": 100}}
    computed = plan_fingerprint(parse_explain_json(plan))
    return {"provenance": {"experiment_id": experiment, "workload": "unit", "query_id": query,
            "candidate_id": candidate, "postgres_version": "17.4", "dataset_version": "v1",
            "run_seed": 7, "code_revision": "abc", "captured_at_utc": "2026-09-04T00:00:00+00:00",
            "query_template": "t1", "parameter_key": "p1"},
            "plan_fingerprint": fingerprint or computed, "planner_settings": {},
            "planning_time_ms": 0.1, "execution_time_ms": 5.0,
            "repetition": repetition, "plan_json": plan}


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_integrity_report_accepts_complete_contiguous_repetitions(tmp_path):
    path = tmp_path / "ok.jsonl"; write(path, [record(repetition=0), record(repetition=1)])
    report = audit_dataset(path)
    assert report.ok and report.observations == 2 and report.experiments == 1
    assert report.unique_query_plans == 1 and report.min_repetitions_per_plan == 2
    assert len(report.sha256) == 64


def test_integrity_report_rejects_fingerprint_mismatch_and_duplicate_repetition(tmp_path):
    path = tmp_path / "bad.jsonl"; bad = record(fingerprint="deadbeef"); write(path, [bad, bad])
    report = audit_dataset(path)
    assert not report.ok
    assert any("stored fingerprint" in error for error in report.errors)
    assert any("duplicate repetition" in error for error in report.errors)


def test_integrity_report_rejects_candidate_plan_drift_within_experiment(tmp_path):
    path = tmp_path / "drift.jsonl"; write(path, [record(node="Seq Scan"), record(node="Index Scan")])
    assert any("candidate plan drift" in error for error in audit_dataset(path).errors)
