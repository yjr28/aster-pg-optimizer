from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_syncs_backup_cleanup_after_successful_rollback(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")
    backup_presence_at_sync: list[bool] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_replace = model_io.os.replace
    original_directory_sync = model_io._fsync_directory

    def fail_metadata_publish(src, dst):
        if Path(dst) == metadata_path:
            raise OSError("simulated metadata publish failure")
        return original_replace(src, dst)

    def record_directory_sync(directory):
        backup_presence_at_sync.append(
            any(Path(directory).glob(".model.joblib.backup.*.tmp"))
        )
        return original_directory_sync(directory)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "replace", fail_metadata_publish)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)

    with pytest.raises(OSError, match="simulated metadata publish failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert path.read_bytes() == b"published-model"
    assert metadata_path.read_text(encoding="utf-8") == '{"version": "old"}\n'
    assert backup_presence_at_sync == [True, True, False]
