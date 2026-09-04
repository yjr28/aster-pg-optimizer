from copy import deepcopy

import pytest

from aster.benchmarks import compare_benchmark_environments, validate_perturbation


def _environment(sha="a"*64):
    return {
        "captured_at_utc":"2026-09-04T00:00:00+00:00",
        "host":{
            "system":"Linux","release":"6.8","machine":"x86_64","platform":"Linux-6.8",
            "cpu_count":8,"cpu_model":"cpu","memory_total_bytes":16_000_000_000,
            "python_version":"3.12.4",
        },
        "postgres":{
            "server_version":"17.4","server_version_num":"170004","database":"bench",
            "database_size_bytes":1000,
            "settings":{"work_mem":{"setting":"4096","unit":"kB","source":"default"}},
            "relations":[{"schema_name":"public","relation_name":"orders","relkind":"r","relpersistence":"p","estimated_rows":100.0,"pages":10}],
            "indexes":[{"schema_name":"public","table_name":"orders","index_name":"orders_pkey","index_definition":"CREATE UNIQUE INDEX orders_pkey ON public.orders USING btree (id)"}],
            "statistics_state":[{"schema_name":"public","relation_name":"orders","n_live_tup":100,"n_dead_tup":0,"last_analyze":"2026-09-04T00:00:00+00:00","last_autoanalyze":None,"analyze_count":1,"autoanalyze_count":0}],
            "statistics_targets":[{"schema_name":"public","relation_name":"orders","column_name":"id","statistics_target":-1}],
        },
        "host_sha256":"c"*64,
        "postgres_sha256":"d"*64,
        "environment_sha256":sha,
    }


def _multi_change_diff():
    before=_environment("a"*64)
    after=deepcopy(before)
    after["environment_sha256"]="b"*64
    after["postgres"]["settings"]["work_mem"]["setting"]="65536"
    after["postgres"]["database_size_bytes"]=1200
    after["postgres"]["indexes"]=[]
    after["postgres"]["statistics_state"][0]["n_live_tup"]=250
    after["postgres"]["statistics_state"][0]["last_analyze"]="2026-09-04T01:00:00+00:00"
    return compare_benchmark_environments(before,after)


def test_environment_diff_classifies_semantic_database_changes():
    diff=_multi_change_diff()
    assert not diff.identical_fingerprint
    assert diff.settings_changes["work_mem"]["before"]["setting"] == "4096"
    assert diff.settings_changes["work_mem"]["after"]["setting"] == "65536"
    assert diff.postgres_metadata_changes["database_size_bytes"] == {"before":1000,"after":1200}
    assert len(diff.index_changes.removed) == 1
    assert len(diff.statistics_state_changes.changed) == 1
    assert set(diff.changed_sections) == {"postgres_metadata","settings","indexes","statistics_state"}


def test_environment_diff_ignores_capture_timestamp_when_fingerprint_is_same():
    before=_environment("a"*64)
    after=deepcopy(before)
    after["captured_at_utc"]="2026-09-04T03:00:00+00:00"
    diff=compare_benchmark_environments(before,after)
    assert diff.identical_fingerprint
    assert diff.changed_sections == ()
    assert diff.to_jsonable()["changed_sections"] == []


def test_perturbation_validation_rejects_unexpected_confounders_and_requires_declared_change():
    diff=_multi_change_diff()
    valid=validate_perturbation(
        diff,
        allowed_sections=("statistics_state","settings","indexes","postgres_metadata"),
        required_sections=("statistics_state",),
    )
    assert valid.valid
    assert valid.missing_required_sections == ()

    statistics_only=validate_perturbation(
        diff,
        allowed_sections=("statistics_state",),
        required_sections=("statistics_state",),
    )
    assert not statistics_only.valid
    assert statistics_only.unexpected_sections == ("indexes","postgres_metadata","settings")

    no_statistics=compare_benchmark_environments(_environment("a"*64),_environment("a"*64))
    missing=validate_perturbation(
        no_statistics,
        allowed_sections=("statistics_state",),
        required_sections=("statistics_state",),
    )
    assert not missing.valid
    assert missing.missing_required_sections == ("statistics_state",)

    with pytest.raises(ValueError,match="subset"):
        validate_perturbation(
            diff,
            allowed_sections=("settings",),
            required_sections=("indexes",),
        )
