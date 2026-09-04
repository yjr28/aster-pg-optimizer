import hashlib

import pytest

from aster.workloads import build_job_manifest, load_job_queries


def test_job_loader_natural_sorts_groups_and_fingerprints(tmp_path):
    (tmp_path / "10b.sql").write_text("SELECT 10;\n")
    (tmp_path / "2a.sql").write_text("SELECT 2;\n")
    (tmp_path / "10a.sql").write_text("SELECT 9;\n")
    (tmp_path / "README.md").write_text("ignored")

    queries = load_job_queries(tmp_path, strict=False)
    assert [query.query_id for query in queries] == ["2a", "10a", "10b"]
    assert [query.family for query in queries] == ["2", "10", "10"]
    assert queries[0].sha256 == hashlib.sha256(b"SELECT 2;").hexdigest()

    manifest = build_job_manifest(tmp_path, strict=False)
    assert manifest.query_count == 3
    assert manifest.family_count == 2
    assert len(manifest.workload_sha256) == 64
    assert manifest.query_ids == ("2a", "10a", "10b")


def test_job_loader_strict_mode_rejects_incomplete_checkout(tmp_path):
    (tmp_path / "1a.sql").write_text("SELECT 1;")
    with pytest.raises(ValueError, match="expected 113 JOB queries"):
        load_job_queries(tmp_path, strict=True)


def test_job_loader_rejects_empty_or_missing_workload(tmp_path):
    with pytest.raises(ValueError, match="no JOB query files"):
        load_job_queries(tmp_path, strict=False)
    with pytest.raises(ValueError, match="does not exist"):
        load_job_queries(tmp_path / "missing", strict=False)
