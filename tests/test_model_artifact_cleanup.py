from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_retains_prior_pair_when_backup_cleanup_fails(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_unlink = Path.unlink
    failed_cleanup = False

    def fail_first_model_backup_cleanup(self, *args, **kwargs):
        nonlocal failed_cleanup
        if ".model.joblib.backup." in self.name and not failed_cleanup:
            failed_cleanup = True
            raise OSError("simulated backup cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(Path, "unlink", fail_first_model_backup_cleanup)

    with pytest.raises(OSError, match="simulated backup cleanup failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    model_backups = list(tmp_path.glob(".model.joblib.backup.*.tmp"))
    metadata_backups = list(tmp_path.glob(".model.joblib.metadata.json.backup.*.tmp"))
    assert len(model_backups) == 1
    assert model_backups[0].read_bytes() == b"published-model"
    assert len(metadata_backups) == 1
    assert metadata_backups[0].read_text(encoding="utf-8") == '{"version": "old"}\n'
    assert path.read_bytes() == b"complete-new-model"
    assert metadata_path.read_text(encoding="utf-8") == '{\n  "version": "new"\n}\n'
