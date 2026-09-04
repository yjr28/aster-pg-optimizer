from aster.workloads.imdb import EXPECTED_IMDB_TABLES
from aster.workloads.preflight import build_job_preflight


def test_preflight_combines_query_and_dataset_identity(tmp_path):
    query_dir = tmp_path / "queries"; query_dir.mkdir()
    csv_dir = tmp_path / "csv"; csv_dir.mkdir()
    (query_dir / "1a.sql").write_text("SELECT 1;")
    (query_dir / "2b.sql").write_text("SELECT 2;")
    for table in EXPECTED_IMDB_TABLES:
        (csv_dir / f"{table}.csv").write_text(f"1,{table}\n")

    first = build_job_preflight(query_dir, csv_dir, strict=False)
    assert first.query_count == 2
    assert first.family_count == 2
    assert first.dataset_file_count == 21
    assert len(first.benchmark_input_sha256) == 64

    (query_dir / "1a.sql").write_text("SELECT 99;")
    second = build_job_preflight(query_dir, csv_dir, strict=False)
    assert second.workload_sha256 != first.workload_sha256
    assert second.benchmark_input_sha256 != first.benchmark_input_sha256


def test_preflight_dataset_change_changes_combined_identity(tmp_path):
    query_dir = tmp_path / "queries"; query_dir.mkdir()
    csv_dir = tmp_path / "csv"; csv_dir.mkdir()
    (query_dir / "1a.sql").write_text("SELECT 1;")
    for table in EXPECTED_IMDB_TABLES:
        (csv_dir / f"{table}.csv").write_text(f"1,{table}\n")
    first = build_job_preflight(query_dir, csv_dir, strict=False)
    with (csv_dir / "title.csv").open("a") as handle:
        handle.write("2,changed\n")
    second = build_job_preflight(query_dir, csv_dir, strict=False)
    assert second.dataset_sha256 != first.dataset_sha256
    assert second.benchmark_input_sha256 != first.benchmark_input_sha256
