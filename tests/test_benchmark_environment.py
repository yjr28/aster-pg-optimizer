import pytest

from aster.benchmarks.environment import (
    HostEnvironment,
    _canonical_sha256,
    capture_benchmark_environment,
)


class FakeRunner:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def benchmark_catalog_snapshot(self):
        return self.snapshot


def _host():
    return HostEnvironment(
        system="Linux",
        release="6.8",
        machine="x86_64",
        platform="Linux-test",
        cpu_count=8,
        cpu_model="Test CPU",
        memory_total_bytes=16 * 1024**3,
        python_version="3.12.0",
    )


def test_environment_hash_is_stable_for_same_host_and_catalog(monkeypatch):
    monkeypatch.setattr("aster.benchmarks.environment.capture_host_environment", _host)
    snapshot={"server_version":"17.11","database_size_bytes":123,"indexes":[]}
    first=capture_benchmark_environment(FakeRunner(snapshot))
    second=capture_benchmark_environment(FakeRunner(dict(snapshot)))
    assert first.host_sha256 == second.host_sha256
    assert first.postgres_sha256 == second.postgres_sha256
    assert first.environment_sha256 == second.environment_sha256
    assert first.captured_at_utc != ""


def test_environment_hash_changes_when_database_state_changes(monkeypatch):
    monkeypatch.setattr("aster.benchmarks.environment.capture_host_environment", _host)
    first=capture_benchmark_environment(FakeRunner({"database_size_bytes":123,"indexes":[]}))
    second=capture_benchmark_environment(FakeRunner({"database_size_bytes":124,"indexes":[]}))
    assert first.postgres_sha256 != second.postgres_sha256
    assert first.environment_sha256 != second.environment_sha256


def test_environment_hash_rejects_non_json_catalog_evidence(monkeypatch):
    monkeypatch.setattr("aster.benchmarks.environment.capture_host_environment", _host)
    snapshot={"database_size_bytes":123,"captured_object":object()}

    with pytest.raises(TypeError, match="JSON serializable"):
        capture_benchmark_environment(FakeRunner(snapshot))


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf"), float("-inf")))
def test_environment_hash_rejects_non_finite_catalog_evidence(monkeypatch, non_finite):
    monkeypatch.setattr("aster.benchmarks.environment.capture_host_environment", _host)
    snapshot={"database_size_bytes":non_finite,"indexes":[]}

    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        capture_benchmark_environment(FakeRunner(snapshot))


def test_captured_postgres_evidence_is_isolated_from_runner_mutation(monkeypatch):
    monkeypatch.setattr("aster.benchmarks.environment.capture_host_environment", _host)
    snapshot={
        "database_size_bytes":123,
        "settings":{"work_mem":{"setting":"4096","unit":"kB"}},
        "indexes":[],
    }
    environment=capture_benchmark_environment(FakeRunner(snapshot))

    snapshot["database_size_bytes"]=999
    snapshot["settings"]["work_mem"]["setting"]="65536"

    assert environment.postgres["database_size_bytes"] == 123
    assert environment.postgres["settings"]["work_mem"]["setting"] == "4096"
    assert environment.postgres_sha256 == _canonical_sha256(environment.postgres)
