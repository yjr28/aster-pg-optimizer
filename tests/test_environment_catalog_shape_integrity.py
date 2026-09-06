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
            "relations": [{
                "schema_name": "public",
                "relation_name": "orders",
                "relkind": "r",
                "relpersistence": "p",
                "estimated_rows": 100.0,
                "pages": 10,
            }],
            "indexes": [{
                "schema_name": "public",
                "table_name": "orders",
                "index_name": "orders_pkey",
                "index_definition": "CREATE UNIQUE INDEX orders_pkey ON public.orders USING btree (id)",
            }],
            "statistics_state": [{
                "schema_name": "public",
                "relation_name": "orders",
                "n_live_tup": 100,
                "n_dead_tup": 0,
                "last_analyze": "2026-09-06T00:00:00+00:00",
                "last_autoanalyze": None,
                "analyze_count": 1,
                "autoanalyze_count": 0,
            }],
            "statistics_targets": [{
                "schema_name": "public",
                "relation_name": "orders",
                "column_name": "id",
                "statistics_target": -1,
            }],
        },
    })


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("relations", "pages"),
        ("indexes", "index_definition"),
        ("statistics_state", "analyze_count"),
        ("statistics_targets", "statistics_target"),
    ),
)
def test_environment_diff_rejects_catalog_rows_missing_modeled_fields(section, field):
    before = _environment()
    after = deepcopy(before)
    del after["postgres"][section][0][field]
    _refresh_hashes(after)

    with pytest.raises(ValueError, match=rf"{section} row fields must be exactly"):
        compare_benchmark_environments(before, after)


@pytest.mark.parametrize(
    "section",
    ("relations", "indexes", "statistics_state", "statistics_targets"),
)
def test_environment_diff_rejects_catalog_rows_with_unmodeled_fields(section):
    before = _environment()
    after = deepcopy(before)
    after["postgres"][section][0]["unmodeled_evidence"] = "value"
    _refresh_hashes(after)

    with pytest.raises(ValueError, match=rf"{section} row fields must be exactly"):
        compare_benchmark_environments(before, after)
