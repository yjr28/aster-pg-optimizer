from copy import deepcopy

from aster.benchmarks import compare_benchmark_environments, validate_perturbation
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


def test_unknown_host_change_is_not_classified_as_modeled_host_change():
    before = _environment()
    after = deepcopy(before)
    after["host"]["container_runtime"] = "docker"
    _refresh_hashes(after)

    diff = compare_benchmark_environments(before, after)

    assert diff.host_changes == {}
    assert diff.changed_sections == ()
    assert diff.unclassified_host_changes == {
        "container_runtime": {"before": None, "after": "docker"}
    }


def test_allowed_host_perturbation_does_not_allow_unknown_host_evidence_drift():
    before = _environment()
    after = deepcopy(before)
    after["host"]["cpu_count"] = 16
    after["host"]["container_runtime"] = "docker"
    _refresh_hashes(after)

    diff = compare_benchmark_environments(before, after)
    validation = validate_perturbation(
        diff,
        allowed_sections=("host",),
        required_sections=("host",),
    )

    assert diff.changed_sections == ("host",)
    assert diff.host_changes == {"cpu_count": {"before": 8, "after": 16}}
    assert diff.unclassified_host_changes == {
        "container_runtime": {"before": None, "after": "docker"}
    }
    assert not validation.valid
    assert validation.observed_sections == ("host",)
    assert validation.unexpected_sections == ()
    assert validation.missing_required_sections == ()
    assert validation.unexplained_fingerprint_change
