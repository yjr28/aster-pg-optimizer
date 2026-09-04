import pytest

from aster.workloads.imdb import EXPECTED_IMDB_TABLES, build_imdb_manifest


def _write_complete_dataset(root):
    for index, table in enumerate(EXPECTED_IMDB_TABLES):
        (root / f"{table}.csv").write_text(f"{index},row-{table}\n")


def test_imdb_manifest_fingerprints_all_expected_tables(tmp_path):
    _write_complete_dataset(tmp_path)
    manifest = build_imdb_manifest(tmp_path)
    assert manifest.file_count == 21
    assert [entry.table for entry in manifest.files] == list(EXPECTED_IMDB_TABLES)
    assert manifest.total_bytes > 0
    assert len(manifest.dataset_sha256) == 64
    assert manifest.dataset_version.startswith("job-imdb-sha256:")


def test_imdb_manifest_changes_when_dataset_bytes_change(tmp_path):
    _write_complete_dataset(tmp_path)
    first = build_imdb_manifest(tmp_path)
    with (tmp_path / "title.csv").open("a") as handle:
        handle.write("999,changed\n")
    second = build_imdb_manifest(tmp_path)
    assert first.dataset_sha256 != second.dataset_sha256
    assert first.dataset_version != second.dataset_version


def test_imdb_manifest_strict_mode_rejects_missing_tables(tmp_path):
    (tmp_path / "title.csv").write_text("1,title\n")
    with pytest.raises(ValueError, match="missing JOB IMDB CSV tables"):
        build_imdb_manifest(tmp_path, strict=True)
    partial = build_imdb_manifest(tmp_path, strict=False)
    assert partial.file_count == 1
