from __future__ import annotations

from dataclasses import dataclass

import pytest

from aster.workloads.collect import JobCollectionConfig, collect_job_workload


@dataclass
class FakeSpec:
    candidate_id: str


@dataclass
class FakeCandidate:
    spec: FakeSpec


class FakeObservation:
    def __init__(self, experiment_id, query_id, candidate_id, repetition, environment_sha256=None):
        self.row = {
            "provenance": {
                "experiment_id": experiment_id,
                "query_id": query_id,
                "candidate_id": candidate_id,
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
        return tuple(FakeObservation(
            kwargs["experiment_id"],
            kwargs["query_id"],
            candidate.spec.candidate_id,
            repetition,
            kwargs.get("environment_sha256"),
        ) for repetition in range(kwargs["repetitions"]))


def _config(*, environment=None):
    return JobCollectionConfig(
        experiment_id="exp-1",
        dataset_version="job-imdb-sha256:abc",
        benchmark_input_sha256="a" * 64,
        workload_sha256="b" * 64,
        run_seed=7,
        code_revision="deadbeef",
        environment_sha256=environment,
        warmups=0,
        repetitions=2,
    )


def test_collection_is_atomic_per_query_and_resumable(tmp_path):
    queries = tmp_path / "job"; queries.mkdir()
    (queries / "1a.sql").write_text("SELECT 'ok-1';")
    (queries / "2a.sql").write_text("SELECT 'fail-me';")
    output = tmp_path / "out"

    first = FakeCollector(fail_query="fail-me")
    summary = collect_job_workload(first, queries, output, config=_config(), candidates=(), strict_workload=False)
    assert summary.completed_queries == 1
    assert summary.failed_queries == 1
    assert (output / "queries" / "1a.jsonl").exists()
    assert not (output / "queries" / "2a.jsonl").exists()
    assert (output / "failures" / "2a.json").exists()

    second = FakeCollector()
    resumed = collect_job_workload(second, queries, output, config=_config(), candidates=(), strict_workload=False)
    assert resumed.skipped_queries == 1
    assert resumed.completed_queries == 1
    assert resumed.failed_queries == 0
    assert second.discover_calls == ["SELECT 'fail-me';"]
    assert (output / "queries" / "2a.jsonl").exists()
    assert not (output / "failures" / "2a.json").exists()


def test_resume_refuses_shard_from_another_experiment(tmp_path):
    queries = tmp_path / "job"; queries.mkdir()
    (queries / "1a.sql").write_text("SELECT 1;")
    output = tmp_path / "out"
    collector = FakeCollector()
    collect_job_workload(collector, queries, output, config=_config(), candidates=(), strict_workload=False)
    shard = output / "queries" / "1a.jsonl"
    text = shard.read_text().replace('"exp-1"', '"other-exp"')
    shard.write_text(text)
    with pytest.raises(ValueError, match="another experiment"):
        collect_job_workload(FakeCollector(), queries, output, config=_config(), candidates=(), strict_workload=False)


def test_resume_refuses_shard_from_another_benchmark_environment(tmp_path):
    queries=tmp_path/"job"; queries.mkdir()
    (queries/"1a.sql").write_text("SELECT 1;")
    output=tmp_path/"out"
    collect_job_workload(
        FakeCollector(),queries,output,config=_config(environment="a"*64),
        candidates=(),strict_workload=False,
    )
    with pytest.raises(ValueError,match="another benchmark environment"):
        collect_job_workload(
            FakeCollector(),queries,output,config=_config(environment="b"*64),
            candidates=(),strict_workload=False,
        )
