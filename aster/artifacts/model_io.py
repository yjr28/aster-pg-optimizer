from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib

from aster.models import RuntimeEnsemble


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

    try:
        staged_metadata_path.write_text(metadata_text, encoding="utf-8")
        joblib.dump(model, staged_model_path)

        had_published_model = path.exists()
        if had_published_model:
            backup_fd, backup_model_name = tempfile.mkstemp(
                prefix=f".{path.name}.backup.", suffix=".tmp", dir=path.parent
            )
            os.close(backup_fd)
            backup_model_path = Path(backup_model_name)
            shutil.copyfile(path, backup_model_path)

        os.replace(staged_model_path, path)
        try:
            os.replace(staged_metadata_path, metadata_path)
        except Exception:
            if had_published_model:
                assert backup_model_path is not None
                os.replace(backup_model_path, path)
            else:
                path.unlink(missing_ok=True)
            raise
    finally:
        staged_model_path.unlink(missing_ok=True)
        staged_metadata_path.unlink(missing_ok=True)
        if backup_model_path is not None:
            backup_model_path.unlink(missing_ok=True)


def load_model(path: str | Path) -> RuntimeEnsemble:
    model = joblib.load(Path(path))
    if not isinstance(model, RuntimeEnsemble):
        raise TypeError("artifact is not an Aster RuntimeEnsemble")
    return model
