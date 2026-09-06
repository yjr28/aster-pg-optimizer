from copy import deepcopy

import pytest

from aster.benchmarks import compare_benchmark_environments
from aster.benchmarks.environment import _canonical_sha256


def _refresh_hashes(environment):
    environment["host_sha256"] = _canonical_sha256(environment["host"])
    environment["postgres_sha256"] = _canonical_sha256(environment["postgres"])
    environment["environment_sha256"] = _canonical_sha256({
        "schema_version": 1,
        "host_sha256": environment["host_sha256"],
        "postgres_sha256": environment["postgres_sha256"],
    })
    return environment


def _environment():
    return _refresh_hashes({
        "captured_at_utc": "2026-09-06T00:00:00+00:00",
        "host": {
            "system": "Linux",
            "release": "6.8",
            "machine": "x86_64",
            "platform": "Linux-6.8",
            "cpu_count": 8,
            "cpu_model": "cpu",
            "memory_total_bytes": 16_000_000_000,
            "python_version": "3.12.4",
        },
        "postgres": {
            "server_version": "17.4",
            "server_version_num": "170004",
            "database": "bench",
            "database_size_bytes": 1000,
            "settings": {},
            "relations": [],
            "indexes": [],
            "statistics_state": [],
            "statistics_targets": [],
        },
    })


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("server_version", 17.4),
        ("server_version", ""),
        ("server_version_num", 170004),
        ("server_version_num", ""),
        ("database", None),
        ("database", ""),
        ("database_size_bytes", "1000"),
        ("database_size_bytes", True),
        ("database_size_bytes", -1),
    ),
)
def test_environment_diff_rejects_invalid_postgres_metadata_evidence(field, invalid_value):
    before = _environment()
    after = deepcopy(before)
    after["postgres"][field] = invalid_value
    _refresh_hashes(after)

    with pytest.raises(ValueError, match=rf"PostgreSQL field {field} has invalid type or value"):
        compare_benchmark_environments(before, after)
