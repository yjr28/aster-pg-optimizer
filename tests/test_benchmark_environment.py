from aster.benchmarks.environment import (
    HostEnvironment,
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
