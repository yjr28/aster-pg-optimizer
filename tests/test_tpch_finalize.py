import json

import pytest

from aster.plans import parse_explain_json, plan_fingerprint
from aster.workloads import finalize_tpch_collection


def _row(query_id, *, environment="a"*64):
    plan={"Plan":{"Node Type":"Seq Scan","Relation Name":query_id,"Total Cost":10,"Plan Rows":100,"Plan Width":8}}
    return {
        "provenance":{
            "experiment_id":"tpch-exp",
            "workload":"tpch",
            "query_id":query_id,
            "candidate_id":"native",
            "postgres_version":"17.11",
            "dataset_version":"tpch-sf1-sha256:abc",
            "environment_sha256":environment,
            "run_seed":7,
            "code_revision":"abc",
            "captured_at_utc":"2026-09-04T00:00:00+00:00",
            "query_template":f"tpch-{query_id}",
            "parameter_key":"params",
        },
        "plan_fingerprint":plan_fingerprint(parse_explain_json(plan)),
        "planner_settings":{},
        "planning_time_ms":0.1,
        "execution_time_ms":5.0,
        "repetition":0,
        "plan_json":plan,
    }


def _collection(root, *, failure=False, environment="a"*64):
    (root/"queries").mkdir(parents=True)
    manifest={
        "workload":"tpch",
        "config":{
            "experiment_id":"tpch-exp",
            "dataset_version":"tpch-sf1-sha256:abc",
            "benchmark_input_sha256":"a"*64,
            "environment_sha256":environment,
            "specification_version":"3.0.1",
            "scale_factor":1.0,
        },
        "summary":{"query_count":2},
    }
    (root/"collection_manifest.json").write_text(json.dumps(manifest))
    for query_id in ("q1","q2"):
        (root/"queries"/f"{query_id}.jsonl").write_text(json.dumps(_row(query_id,environment=environment))+"\n")
    if failure:
        (root/"failures").mkdir()
        (root/"failures"/"q2.json").write_text("{}")


def test_finalize_tpch_requires_complete_valid_collection(tmp_path):
    root=tmp_path/"collection"; _collection(root)
    out=tmp_path/"tpch.jsonl"
    result=finalize_tpch_collection(root,out)
    assert result.experiment_id == "tpch-exp"
    assert result.specification_version == "3.0.1"
    assert result.scale_factor == 1.0
    assert result.environment_sha256 == "a"*64
    assert result.query_count == 2
    assert result.observation_count == 2
    assert result.unique_query_plans == 2
    assert out.exists()
    assert (root/"finalized_manifest.json").exists()


def test_finalize_tpch_rejects_unresolved_failure(tmp_path):
    root=tmp_path/"collection"; _collection(root,failure=True)
    with pytest.raises(ValueError,match="unresolved query failures"):
        finalize_tpch_collection(root,tmp_path/"tpch.jsonl")


def test_finalize_tpch_rejects_environment_tampering(tmp_path):
    root=tmp_path/"collection"; _collection(root)
    shard=root/"queries"/"q2.jsonl"
    shard.write_text(shard.read_text().replace("a"*64,"b"*64))
    with pytest.raises(ValueError,match="mixed benchmark environment"):
        finalize_tpch_collection(root,tmp_path/"tpch.jsonl")
