from copy import deepcopy

import pytest

from aster.benchmarks import compare_benchmark_environments
from aster.benchmarks.environment import _canonical_sha256


def _environment():
    host = {
        "system": "Linux",
        "release": "6.8",
        "machine": "x86_64",
        "platform": "Linux-6.8",
        "cpu_count": 8,
        "cpu_model": "cpu",
        "memory_total_bytes": 16_000_000_000,
        "python_version": "3.12.4",
    }
    postgres = {
        "server_version": "17.4",
        "server_version_num": "170004",
        "database": "bench",
        "database_size_bytes": 1000,
        "settings": {},
        "relations": [],
        "indexes": [],
        "statistics_state": [],
        "statistics_targets": [],
    }
    host_sha = _canonical_sha256(host)
    postgres_sha = _canonical_sha256(postgres)
    return {
        "captured_at_utc": "2026-09-06T00:00:00+00:00",
        "host": host,
        "postgres": postgres,
        "host_sha256": host_sha,
        "postgres_sha256": postgres_sha,
        "environment_sha256": _canonical_sha256({
            "schema_version": 1,
            "host_sha256": host_sha,
            "postgres_sha256": postgres_sha,
        }),
    }


def _rehash_environment(environment):
    environment["host_sha256"] = _canonical_sha256(environment["host"])
    environment["postgres_sha256"] = _canonical_sha256(environment["postgres"])
    environment["environment_sha256"] = _canonical_sha256({
        "schema_version": 1,
        "host_sha256": environment["host_sha256"],
        "postgres_sha256": environment["postgres_sha256"],
    })


def test_environment_diff_requires_capture_timestamp_evidence():
    before = _environment()
    after = deepcopy(before)
    del after["captured_at_utc"]

    with pytest.raises(ValueError, match=r"environment fields must be exactly"):
        compare_benchmark_environments(before, after)


def test_environment_diff_rejects_unmodeled_top_level_evidence():
    before = _environment()
    after = deepcopy(before)
    after["collector_note"] = "unhashed evidence"

    with pytest.raises(ValueError, match=r"environment fields must be exactly"):
        compare_benchmark_environments(before, after)


@pytest.mark.parametrize("invalid_value", (None, 123, ""))
def test_environment_diff_rejects_invalid_capture_timestamp_type(invalid_value):
    before = _environment()
    after = deepcopy(before)
    after["captured_at_utc"] = invalid_value

    with pytest.raises(ValueError, match=r"captured_at_utc must be a non-empty string"):
        compare_benchmark_environments(before, after)


@pytest.mark.parametrize("field", ("cpu_count", "memory_total_bytes"))
def test_environment_diff_rejects_negative_host_capacity_evidence(field):
    before = _environment()
    after = deepcopy(before)
    after["host"][field] = -1
    _rehash_environment(after)

    with pytest.raises(ValueError, match=rf"host field {field} has invalid type or value"):
        compare_benchmark_environments(before, after)
