import pytest

from aster.workloads import (
    EXPECTED_TPCH_TABLES,
    build_tpch_dataset_manifest,
    build_tpch_workload_manifest,
    load_tpch_queries,
)


def test_tpch_query_loader_accepts_q_prefix_and_natural_order(tmp_path):
    for number in range(1,23):
        name=f"q{number}.sql" if number % 2 else f"{number}.sql"
        (tmp_path/name).write_text(f"select {number};\n",encoding="utf-8")
    queries=load_tpch_queries(tmp_path)
    assert [query.number for query in queries] == list(range(1,23))
    manifest=build_tpch_workload_manifest(tmp_path)
    assert manifest.query_count == 22
    assert len(manifest.workload_sha256) == 64
    old=manifest.workload_sha256
    (tmp_path/"q1.sql").write_text("select 100;\n",encoding="utf-8")
    assert build_tpch_workload_manifest(tmp_path).workload_sha256 != old


def test_tpch_dataset_manifest_requires_eight_tables_and_hashes_bytes(tmp_path):
    for table in EXPECTED_TPCH_TABLES:
        (tmp_path/f"{table}.tbl").write_bytes((table+"|1|\n").encode())
    manifest=build_tpch_dataset_manifest(tmp_path,scale_factor=1.0)
    assert manifest.file_count == 8
    assert manifest.dataset_version.startswith("tpch-sf1-sha256:")
    old=manifest.dataset_sha256
    (tmp_path/"lineitem.tbl").write_bytes(b"changed|\n")
    assert build_tpch_dataset_manifest(tmp_path,scale_factor=1.0).dataset_sha256 != old


def test_tpch_dataset_manifest_rejects_ambiguous_formats(tmp_path):
    for table in EXPECTED_TPCH_TABLES:
        (tmp_path/f"{table}.tbl").write_text("x|\n")
    (tmp_path/"region.csv").write_text("x\n")
    with pytest.raises(ValueError,match="ambiguous"):
        build_tpch_dataset_manifest(tmp_path)
