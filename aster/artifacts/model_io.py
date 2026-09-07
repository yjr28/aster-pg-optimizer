from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib

from aster.models import RuntimeEnsemble


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_model(path: str | Path, model: RuntimeEnsemble, metadata: dict[str, Any]) -> None:
    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    model_fd, staged_model_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    metadata_fd, staged_metadata_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(model_fd)
    os.close(metadata_fd)
    staged_model_path = Path(staged_model_name)
    staged_metadata_path = Path(staged_metadata_name)
    backup_model_path: Path | None = None
    backup_metadata_path: Path | None = None

    try:
        staged_metadata_path.write_text(metadata_text, encoding="utf-8")
        joblib.dump(model, staged_model_path)
        _fsync_file(staged_metadata_path)
        _fsync_file(staged_model_path)

        had_published_model = path.exists()
        had_published_metadata = metadata_path.exists()
        if had_published_model:
            backup_fd, backup_model_name = tempfile.mkstemp(
                prefix=f".{path.name}.backup.", suffix=".tmp", dir=path.parent
            )
            os.close(backup_fd)
            backup_model_path = Path(backup_model_name)
            shutil.copyfile(path, backup_model_path)
            _fsync_file(backup_model_path)

            if had_published_metadata:
                metadata_backup_fd, backup_metadata_name = tempfile.mkstemp(
                    prefix=f".{metadata_path.name}.backup.",
                    suffix=".tmp",
                    dir=path.parent,
                )
                os.close(metadata_backup_fd)
                backup_metadata_path = Path(backup_metadata_name)
                shutil.copyfile(metadata_path, backup_metadata_path)
                _fsync_file(backup_metadata_path)

            _fsync_directory(path.parent)

        os.replace(staged_model_path, path)
        try:
            os.replace(staged_metadata_path, metadata_path)
        except Exception:
            if had_published_model:
                assert backup_model_path is not None
                try:
                    # Restore from a separate synced copy so the durable backup
                    # remains independent until the rollback rename itself has
                    # been made durable in the parent directory.
                    shutil.copyfile(backup_model_path, staged_model_path)
                    _fsync_file(staged_model_path)
                    os.replace(staged_model_path, path)
                    _fsync_directory(path.parent)
                except Exception:
                    # Preserve any synced prior-pair evidence when rollback
                    # preparation, publication, or directory sync cannot be
                    # completed. The caller can recover explicitly from copies
                    # that were durable before publication began.
                    backup_model_path = None
                    backup_metadata_path = None
                    raise
            else:
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            raise

        if backup_model_path is not None:
            # First make the newly published pair's directory entries durable
            # while any synced prior-pair backups are still available. If that
            # sync fails, preserve those backups for explicit recovery rather
            # than allowing finally-cleanup to discard the strongest evidence.
            try:
                _fsync_directory(path.parent)
            except Exception:
                backup_model_path = None
                backup_metadata_path = None
                raise
            try:
                backup_model_path.unlink()
            except Exception:
                # A failed cleanup must not be retried by finally: the synced
                # prior-model backup, and its matching metadata backup when
                # present, are stronger evidence than an unsynced retry-removal.
                backup_model_path = None
                backup_metadata_path = None
                raise
            backup_model_path = None
            if backup_metadata_path is not None:
                backup_metadata_path.unlink()
                backup_metadata_path = None
        _fsync_directory(path.parent)
    finally:
        staged_model_path.unlink(missing_ok=True)
        staged_metadata_path.unlink(missing_ok=True)
        if backup_model_path is not None:
            backup_model_path.unlink(missing_ok=True)
        if backup_metadata_path is not None:
            backup_metadata_path.unlink(missing_ok=True)


def load_model(path: str | Path) -> RuntimeEnsemble:
    model = joblib.load(Path(path))
    if not isinstance(model, RuntimeEnsemble):
        raise TypeError("artifact is not an Aster RuntimeEnsemble")
    return model
