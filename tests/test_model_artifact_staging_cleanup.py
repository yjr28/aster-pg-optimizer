from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_syncs_staging_cleanup_after_prepublication_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    events: list[str] = []

    def fail_after_partial_write(model, target):
        Path(target).write_bytes(b"partial-new-model")
        raise OSError("simulated model write failure")

    original_unlink = Path.unlink
    original_directory_sync = model_io._fsync_directory

    def record_unlink(self, *args, **kwargs):
        existed = self.exists()
        result = original_unlink(self, *args, **kwargs)
        if existed and self.parent == tmp_path and self.name.endswith(".tmp"):
            events.append("unlink")
        return result

    def record_directory_sync(directory):
        events.append("directory-fsync")
        return original_directory_sync(directory)

    monkeypatch.setattr(model_io.joblib, "dump", fail_after_partial_write)
    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)

    with pytest.raises(OSError, match="simulated model write failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert events[-3:] == ["unlink", "unlink", "directory-fsync"]
    assert list(tmp_path.glob(".*.tmp")) == []
