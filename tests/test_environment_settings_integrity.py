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
            "settings": {
                "work_mem": {"setting": "4096", "unit": "kB", "source": "default"}
            },
            "relations": [],
            "indexes": [],
            "statistics_state": [],
            "statistics_targets": [],
        },
    })


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        ({7: {"setting": "4096", "unit": "kB", "source": "default"}}, "setting names must be non-empty strings"),
        ({"": {"setting": "4096", "unit": "kB", "source": "default"}}, "setting names must be non-empty strings"),
        ({"work_mem": "4096"}, "setting work_mem evidence must be an object"),
        ({"work_mem": {"setting": "4096", "unit": "kB"}}, "setting work_mem evidence fields"),
        ({"work_mem": {"setting": "4096", "unit": "kB", "source": "default", "pending_restart": False}}, "setting work_mem evidence fields"),
        ({"work_mem": {"setting": 4096, "unit": "kB", "source": "default"}}, "setting work_mem field setting has invalid type"),
        ({"work_mem": {"setting": "4096", "unit": 1024, "source": "default"}}, "setting work_mem field unit has invalid type"),
        ({"work_mem": {"setting": "4096", "unit": "kB", "source": None}}, "setting work_mem field source has invalid type"),
    ),
)
def test_environment_diff_rejects_malformed_setting_evidence(settings, message):
    before = _environment()
    after = deepcopy(before)
    after["postgres"]["settings"] = settings
    _refresh_hashes(after)

    with pytest.raises(ValueError, match=message):
        compare_benchmark_environments(before, after)


def test_environment_diff_accepts_nullable_setting_unit():
    before = _environment()
    after = deepcopy(before)
    after["postgres"]["settings"]["work_mem"]["unit"] = None
    _refresh_hashes(after)

    diff = compare_benchmark_environments(before, after)

    assert diff.changed_sections == ("settings",)
