from __future__ import annotations

from dataclasses import dataclass

import pytest

from aster.workloads import TpchCollectionConfig, collect_tpch_workload


@dataclass
class FakeSpec:
    candidate_id: str


@dataclass
class FakeCandidate:
    spec: FakeSpec


class FakeObservation:
    def __init__(self, experiment_id, query_id, candidate_id, repetition, workload, environment_sha256=None):
        self.row = {
            "provenance": {
                "experiment_id": experiment_id,
                "query_id": query_id,
                "candidate_id": candidate_id,
                "workload": workload,
                "environment_sha256": environment_sha256,
            },
            "repetition": repetition,
        }

    def to_jsonable(self):
        return self.row


class FakeCollector:
    def __init__(self, fail_query=None):
        self.fail_query = fail_query
        self.discover_calls = []

    def discover(self, query, candidates):
        self.discover_calls.append(query)
        if self.fail_query and self.fail_query in query:
            raise RuntimeError("seeded failure")
        return (FakeCandidate(FakeSpec("native")), FakeCandidate(FakeSpec("alt")))

    def measure(self, query, candidate, **kwargs):
        return tuple(
            FakeObservation(
                kwargs["experiment_id"], kwargs["query_id"], candidate.spec.candidate_id,
                repetition, kwargs["workload"], kwargs.get("environment_sha256"),
            )
            for repetition in range(kwargs["repetitions"])
        )


def _config(*, environment=None):
    return TpchCollectionConfig(
        experiment_id="exp-tpch",
        dataset_version="tpch-sf1-sha256:abc",
        benchmark_input_sha256="a"*64,
        workload_sha256="b"*64,
        specification_version="3.0.1",
        scale_factor=1.0,
        run_seed=7,
        code_revision="deadbeef",
        environment_sha256=environment,
        warmups=0,
        repetitions=2,
    )


def test_tpch_collection_is_atomic_and_resumable(tmp_path):
    queries=tmp_path/"queries"; queries.mkdir()
    (queries/"q1.sql").write_text("SELECT 'ok';")
    (queries/"q2.sql").write_text("SELECT 'fail';")
    output=tmp_path/"out"

    first=FakeCollector(fail_query="fail")
    summary=collect_tpch_workload(first,queries,output,config=_config(),candidates=(),strict_workload=False)
    assert summary.completed_queries == 1
    assert summary.failed_queries == 1
    assert (output/"queries"/"q1.jsonl").exists()
    assert not (output/"queries"/"q2.jsonl").exists()

    second=FakeCollector()
    resumed=collect_tpch_workload(second,queries,output,config=_config(),candidates=(),strict_workload=False)
    assert resumed.skipped_queries == 1
    assert resumed.completed_queries == 1
    assert second.discover_calls == ["SELECT 'fail';"]
    assert not (output/"failures"/"q2.json").exists()


def test_tpch_resume_refuses_other_experiment(tmp_path):
    queries=tmp_path/"queries"; queries.mkdir()
    (queries/"q1.sql").write_text("SELECT 1;")
    output=tmp_path/"out"
    collect_tpch_workload(FakeCollector(),queries,output,config=_config(),candidates=(),strict_workload=False)
    shard=output/"queries"/"q1.jsonl"
    shard.write_text(shard.read_text().replace('"exp-tpch"','"other"'))
    with pytest.raises(ValueError,match="another experiment"):
        collect_tpch_workload(FakeCollector(),queries,output,config=_config(),candidates=(),strict_workload=False)


def test_tpch_resume_refuses_other_benchmark_environment(tmp_path):
    queries=tmp_path/"queries"; queries.mkdir()
    (queries/"q1.sql").write_text("SELECT 1;")
    output=tmp_path/"out"
    collect_tpch_workload(
        FakeCollector(),queries,output,config=_config(environment="a"*64),
        candidates=(),strict_workload=False,
    )
    with pytest.raises(ValueError,match="another benchmark environment"):
        collect_tpch_workload(
            FakeCollector(),queries,output,config=_config(environment="b"*64),
            candidates=(),strict_workload=False,
        )
