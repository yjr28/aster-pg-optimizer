from pathlib import Path

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_syncs_new_artifact_directory_entry_before_first_publish(
    tmp_path, monkeypatch
):
    artifact_dir = tmp_path / "artifacts"
    path = artifact_dir / "model.joblib"
    synced_directories: list[Path] = []
    synced_before_first_publish: list[tuple[Path, ...]] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_directory_sync = model_io._fsync_directory
    original_replace = model_io.os.replace

    def record_directory_sync(directory):
        synced_directories.append(Path(directory))
        return original_directory_sync(directory)

    def record_first_publish(src, dst):
        if not synced_before_first_publish:
            synced_before_first_publish.append(tuple(synced_directories))
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)
    monkeypatch.setattr(model_io.os, "replace", record_first_publish)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert synced_before_first_publish
    assert tmp_path in synced_before_first_publish[0]
