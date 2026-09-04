import json

import pytest

from aster.data import DatasetIntegrityReport
from aster.workloads.finalize import finalize_job_collection


def _write_collection(root, *, failures=False, dataset_version="v1", environment=None):
    (root / "queries").mkdir(parents=True)
    manifest={"config":{"experiment_id":"exp-1","dataset_version":dataset_version,
             "benchmark_input_sha256":"a"*64,"environment_sha256":environment},
             "summary":{"query_count":2}}
    (root / "collection_manifest.json").write_text(json.dumps(manifest))
    for query_id in ("1a","2b"):
        row={"provenance":{"experiment_id":"exp-1","dataset_version":dataset_version,
             "environment_sha256":environment,"workload":"job","query_id":query_id}}
        (root / "queries" / f"{query_id}.jsonl").write_text(json.dumps(row)+"\n")
    if failures:
        (root / "failures").mkdir()
        (root / "failures" / "2b.json").write_text("{}")


def test_finalize_requires_complete_failure_free_single_experiment(tmp_path, monkeypatch):
    root=tmp_path/"collection"; _write_collection(root,environment="a"*64)
    audit=DatasetIntegrityReport(
        sha256="c"*64,
        observations=2,
        experiments=1,
        queries=2,
        query_templates=2,
        unique_query_plans=2,
        min_repetitions_per_plan=1,
        max_repetitions_per_plan=1,
        missing_template_records=0,
        errors=(),
        warnings=(),
    )
    monkeypatch.setattr("aster.workloads.finalize.audit_dataset", lambda path: audit)
    result=finalize_job_collection(root,tmp_path/"dataset.jsonl")
    assert result.query_count==2
    assert result.observation_count==2
    assert result.environment_sha256=="a"*64
    assert (root/"finalized_manifest.json").exists()
    assert (tmp_path/"dataset.jsonl").read_text().count("\n")==2


def test_finalize_rejects_failures_mixed_dataset_versions_and_environment_tampering(tmp_path):
    failed=tmp_path/"failed"; _write_collection(failed,failures=True)
    with pytest.raises(ValueError,match="unresolved query failures"):
        finalize_job_collection(failed,tmp_path/"x.jsonl")

    mixed=tmp_path/"mixed"; _write_collection(mixed)
    shard=mixed/"queries"/"2b.jsonl"
    shard.write_text(shard.read_text().replace('"v1"','"v2"'))
    with pytest.raises(ValueError,match="mixed dataset version"):
        finalize_job_collection(mixed,tmp_path/"y.jsonl")

    environment=tmp_path/"environment"; _write_collection(environment,environment="a"*64)
    env_shard=environment/"queries"/"2b.jsonl"
    env_shard.write_text(env_shard.read_text().replace('"a'*0, '"a'*0) if False else env_shard.read_text().replace("a"*64,"b"*64))
    with pytest.raises(ValueError,match="mixed benchmark environment"):
        finalize_job_collection(environment,tmp_path/"z.jsonl")
