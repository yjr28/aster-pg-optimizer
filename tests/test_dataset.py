import json

from aster.data.load import load_training_examples


def test_loader_uses_median_repetitions_and_keeps_plan_drift_separate(tmp_path):
    path = tmp_path / "data.jsonl"; rows = []
    for runtime in [10, 12, 100]:
        rows.append({"provenance": {"workload": "w", "query_id": "q1", "candidate_id": "native",
                     "query_template": "t1", "parameter_key": "p1", "dataset_version": "v1"},
                     "plan_fingerprint": "fp1", "execution_time_ms": runtime,
                     "plan_json": {"Plan": {"Node Type": "Seq Scan", "Total Cost": 10}}})
    rows.append({"provenance": {"workload": "w", "query_id": "q1", "candidate_id": "native",
                 "query_template": "t1", "parameter_key": "p1", "dataset_version": "v1"},
                 "plan_fingerprint": "fp2", "execution_time_ms": 5,
                 "plan_json": {"Plan": {"Node Type": "Index Scan", "Total Cost": 5}}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    examples = load_training_examples(path)
    assert len(examples) == 2 and sorted(e.runtime_ms for e in examples) == [5.0, 12.0]


def test_loader_keeps_same_query_id_separate_across_dataset_versions(tmp_path):
    from aster.plans import parse_explain_json, plan_fingerprint
    path = tmp_path / "versions.jsonl"
    plan = {"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders"}}
    fingerprint = plan_fingerprint(parse_explain_json(plan)); rows = []
    for version, runtime in [("v1", 5.0), ("v2", 50.0)]:
        rows.append({"provenance": {"experiment_id": f"exp-{version}", "workload": "w",
                     "query_id": "q1", "candidate_id": "native", "query_template": "t1",
                     "parameter_key": "p1", "dataset_version": version},
                     "plan_fingerprint": fingerprint, "execution_time_ms": runtime, "plan_json": plan})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    examples = load_training_examples(path)
    assert len(examples) == 2 and {e.dataset_version for e in examples} == {"v1", "v2"}
