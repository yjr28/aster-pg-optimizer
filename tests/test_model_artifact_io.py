from pathlib import Path

import pytest

import aster.artifacts.model_io as model_io
from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_rejects_unserializable_metadata_before_writing_artifact(tmp_path):
    path = tmp_path / "model.joblib"

    with pytest.raises(TypeError):
        save_model(path, RuntimeEnsemble(), {"bad": object()})

    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".metadata.json").exists()


def test_save_model_preserves_published_artifact_when_model_write_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def fail_after_partial_write(model, target):
        Path(target).write_bytes(b"partial-new-model")
        raise OSError("simulated model write failure")

    monkeypatch.setattr(model_io.joblib, "dump", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated model write failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert path.read_bytes() == b"published-model"
    assert metadata_path.read_text(encoding="utf-8") == '{"version": "old"}\n'


def test_save_model_preserves_published_pair_when_metadata_write_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_write_text = Path.write_text

    def fail_after_partial_metadata_write(self, data, *args, **kwargs):
        if "metadata.json" in self.name:
            self.write_bytes(b'{"version":')
            raise OSError("simulated metadata write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(Path, "write_text", fail_after_partial_metadata_write)

    with pytest.raises(OSError, match="simulated metadata write failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert path.read_bytes() == b"published-model"
    assert metadata_path.read_text(encoding="utf-8") == '{"version": "old"}\n'
