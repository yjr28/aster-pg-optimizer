from copy import deepcopy

import pytest

from aster.benchmarks import compare_benchmark_environments
from aster.benchmarks.environment import _canonical_sha256


def _environment():
    environment = {
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
    }
    return _refresh_hashes(environment)


def _refresh_hashes(environment):
    environment["host_sha256"] = _canonical_sha256(environment["host"])
    environment["postgres_sha256"] = _canonical_sha256(environment["postgres"])
    environment["environment_sha256"] = _canonical_sha256(
        {
            "schema_version": 1,
            "host_sha256": environment["host_sha256"],
            "postgres_sha256": environment["postgres_sha256"],
        }
    )
    return environment


@pytest.mark.parametrize(
    "missing_field",
    (
        "system",
        "release",
        "machine",
        "platform",
        "cpu_count",
        "cpu_model",
        "memory_total_bytes",
        "python_version",
    ),
)
def test_environment_diff_rejects_missing_modeled_host_evidence(missing_field):
    before = _environment()
    after = deepcopy(before)
    del after["host"][missing_field]
    _refresh_hashes(after)

    with pytest.raises(ValueError, match=r"after host snapshot missing modeled fields"):
        compare_benchmark_environments(before, after)
