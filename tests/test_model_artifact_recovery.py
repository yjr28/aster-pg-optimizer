from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_retains_prior_metadata_with_model_when_pair_sync_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_directory_sync = model_io._fsync_directory
    sync_count = 0

    def fail_pair_directory_sync(directory):
        nonlocal sync_count
        sync_count += 1
        if sync_count == 2:
            raise OSError("simulated published-pair directory sync failure")
        return original_directory_sync(directory)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io, "_fsync_directory", fail_pair_directory_sync)

    with pytest.raises(OSError, match="simulated published-pair directory sync failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    model_backups = list(tmp_path.glob(".model.joblib.backup.*.tmp"))
    metadata_backups = list(tmp_path.glob(".model.joblib.metadata.json.backup.*.tmp"))
    assert len(model_backups) == 1
    assert model_backups[0].read_bytes() == b"published-model"
    assert len(metadata_backups) == 1
    assert metadata_backups[0].read_text(encoding="utf-8") == '{"version": "old"}\n'
