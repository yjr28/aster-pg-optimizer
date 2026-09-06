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


def test_save_model_restores_published_pair_when_metadata_publish_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_replace = model_io.os.replace

    def fail_metadata_publish(src, dst):
        if Path(dst) == metadata_path:
            raise OSError("simulated metadata publish failure")
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "replace", fail_metadata_publish)

    with pytest.raises(OSError, match="simulated metadata publish failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert path.read_bytes() == b"published-model"
    assert metadata_path.read_text(encoding="utf-8") == '{"version": "old"}\n'


def test_save_model_retains_synced_backup_when_rollback_publish_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    path.write_bytes(b"published-model")
    metadata_path.write_text('{"version": "old"}\n', encoding="utf-8")

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_replace = model_io.os.replace
    publish_count = 0

    def fail_metadata_and_rollback_publish(src, dst):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 2 and Path(dst) == metadata_path:
            raise OSError("simulated metadata publish failure")
        if publish_count == 3 and Path(dst) == path:
            raise OSError("simulated rollback publish failure")
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "replace", fail_metadata_and_rollback_publish)

    with pytest.raises(OSError, match="simulated rollback publish failure"):
        save_model(path, RuntimeEnsemble(), {"version": "new"})

    backups = list(tmp_path.glob(".model.joblib.backup.*.tmp"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"published-model"
    assert path.read_bytes() == b"complete-new-model"
    assert metadata_path.read_text(encoding="utf-8") == '{"version": "old"}\n'


def test_save_model_syncs_staged_bytes_before_first_publish(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    synced_fds: list[int] = []
    first_publish_sync_count: list[int] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    def record_fsync(fd):
        synced_fds.append(fd)

    original_replace = model_io.os.replace

    def record_first_publish(src, dst):
        if not first_publish_sync_count:
            first_publish_sync_count.append(len(synced_fds))
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "fsync", record_fsync)
    monkeypatch.setattr(model_io.os, "replace", record_first_publish)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert first_publish_sync_count == [2]


def test_save_model_syncs_rollback_backup_before_first_publish(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"published-model")
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        '{"version": "old"}\n', encoding="utf-8"
    )
    synced_fds: list[int] = []
    first_publish_sync_count: list[int] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    def record_fsync(fd):
        synced_fds.append(fd)

    original_replace = model_io.os.replace

    def record_first_publish(src, dst):
        if not first_publish_sync_count:
            first_publish_sync_count.append(len(synced_fds))
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "fsync", record_fsync)
    monkeypatch.setattr(model_io.os, "replace", record_first_publish)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    # Staged metadata, staged model, rollback backup bytes, and the backup's
    # parent-directory entry are all synced before publication starts.
    assert first_publish_sync_count == [4]


def test_save_model_syncs_directory_after_publishing_pair(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    events: list[str] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_fsync = model_io.os.fsync
    original_replace = model_io.os.replace

    def record_fsync(fd):
        events.append("fsync")
        return original_fsync(fd)

    def record_replace(src, dst):
        events.append("replace")
        return original_replace(src, dst)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "fsync", record_fsync)
    monkeypatch.setattr(model_io.os, "replace", record_replace)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert events.count("replace") == 2
    assert events[-1] == "fsync"


def test_save_model_syncs_backup_directory_entry_before_first_publish(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"published-model")
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        '{"version": "old"}\n', encoding="utf-8"
    )
    events: list[str] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_replace = model_io.os.replace
    original_directory_sync = model_io._fsync_directory

    def record_replace(src, dst):
        events.append("replace")
        return original_replace(src, dst)

    def record_directory_sync(directory):
        events.append("directory-fsync")
        return original_directory_sync(directory)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io.os, "replace", record_replace)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert events[0:2] == ["directory-fsync", "replace"]


def test_save_model_removes_backup_before_final_directory_sync(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"published-model")
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        '{"version": "old"}\n', encoding="utf-8"
    )
    backup_presence_at_sync: list[bool] = []

    def write_new_model(model, target):
        Path(target).write_bytes(b"complete-new-model")

    original_directory_sync = model_io._fsync_directory

    def record_directory_sync(directory):
        backup_presence_at_sync.append(
            any(Path(directory).glob(".model.joblib.backup.*.tmp"))
        )
        return original_directory_sync(directory)

    monkeypatch.setattr(model_io.joblib, "dump", write_new_model)
    monkeypatch.setattr(model_io, "_fsync_directory", record_directory_sync)

    save_model(path, RuntimeEnsemble(), {"version": "new"})

    assert backup_presence_at_sync == [True, False]
