from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_syncs_model_backup_removal_before_reporting_metadata_cleanup_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    events: list[str] = []
    original_unlink = Path.unlink

    def record_directory_sync(directory):
        events.append("directory-sync")

    def fail_metadata_backup_cleanup(self, *args, **kwargs):
        if ".model.joblib.backup." in self.name:
            events.append("model-backup-unlink")
            return original_unlink(self, *args, **kwargs)
        if ".metadata.json.backup." in self.name:
            events.append("metadata-backup-unlink-failed")
            raise OSError("simulated metadata backup cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)
    monkeypatch.setattr(Path, "unlink", fail_metadata_backup_cleanup)

    with pytest.raises(OSError, match="simulated metadata backup cleanup failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    model_unlink_index = events.index("model-backup-unlink")
    assert "directory-sync" in events[model_unlink_index + 1 :], (
        "a successful model-backup removal must be made durable before the "
        "metadata-cleanup failure is returned; otherwise the filesystem may "
        "still resurrect evidence the code claims was removed"
    )
