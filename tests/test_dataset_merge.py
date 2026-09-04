import json

import pytest

from aster.data import combine_datasets
from aster.plans import parse_explain_json, plan_fingerprint


def _record(*, experiment, workload, query, candidate, runtime, repetition=0, dataset="v1", environment=None):
    plan={"Plan":{"Node Type":"Seq Scan","Relation Name":f"{workload}_{query}","Total Cost":10,"Plan Rows":100,"Plan Width":8}}
    fingerprint=plan_fingerprint(parse_explain_json(plan))
    provenance={
        "experiment_id":experiment,
        "workload":workload,
        "query_id":query,
        "candidate_id":candidate,
        "postgres_version":"17.11",
        "dataset_version":dataset,
        "run_seed":7,
        "code_revision":"abc",
        "captured_at_utc":"2026-09-04T00:00:00+00:00",
        "query_template":f"{workload}-{query}",
        "parameter_key":"p1",
    }
    if environment is not None:
        provenance["environment_sha256"]=environment
    return {
        "provenance":provenance,
        "plan_fingerprint":fingerprint,
        "planner_settings":{},
        "planning_time_ms":0.1,
        "execution_time_ms":runtime,
        "repetition":repetition,
        "plan_json":plan,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")


def test_combine_datasets_reaudits_and_records_multiple_workloads(tmp_path):
    job=tmp_path/"job.jsonl"; tpch=tmp_path/"tpch.jsonl"
    _write(job,[_record(experiment="job-exp",workload="job",query="1a",candidate="native",runtime=10.0,dataset="job-v1",environment="a"*64)])
    _write(tpch,[_record(experiment="tpch-exp",workload="tpch",query="q1",candidate="native",runtime=20.0,dataset="tpch-v1",environment="b"*64)])
    out=tmp_path/"combined.jsonl"
    result=combine_datasets([job,tpch],out,require_multiple_workloads=True)
    assert result.observations == 2
    assert result.workloads == ("job","tpch")
    assert result.dataset_versions == ("job-v1","tpch-v1")
    assert result.environments == ("a"*64,"b"*64)
    assert len(result.inputs) == 2
    assert result.inputs[0].environments == ("a"*64,)
    assert out.exists()
    assert out.with_suffix(".jsonl.manifest.json").exists()


def test_combine_datasets_rejects_duplicate_observation_identity(tmp_path):
    row=_record(experiment="exp",workload="job",query="1a",candidate="native",runtime=10.0)
    first=tmp_path/"a.jsonl"; second=tmp_path/"b.jsonl"
    _write(first,[row]); _write(second,[row])
    with pytest.raises(ValueError,match="duplicate observation identity"):
        combine_datasets([first,second],tmp_path/"combined.jsonl")


def test_combine_allows_same_experiment_query_repetition_from_distinct_environments(tmp_path):
    first=tmp_path/"a.jsonl"; second=tmp_path/"b.jsonl"
    common={"experiment":"exp","workload":"job","query":"1a","candidate":"native"}
    _write(first,[_record(**common,runtime=10.0,environment="a"*64)])
    _write(second,[_record(**common,runtime=20.0,environment="b"*64)])
    result=combine_datasets([first,second],tmp_path/"combined.jsonl")
    assert result.observations == 2
    assert result.queries == 2
    assert result.unique_query_plans == 2
    assert result.environments == ("a"*64,"b"*64)


def test_combine_requires_multiple_workloads_when_requested(tmp_path):
    first=tmp_path/"a.jsonl"; second=tmp_path/"b.jsonl"
    _write(first,[_record(experiment="a",workload="job",query="1a",candidate="native",runtime=10.0)])
    _write(second,[_record(experiment="b",workload="job",query="2a",candidate="native",runtime=11.0)])
    with pytest.raises(ValueError,match="at least two distinct workloads"):
        combine_datasets([first,second],tmp_path/"combined.jsonl",require_multiple_workloads=True)
