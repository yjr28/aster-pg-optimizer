from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

from aster.models import RuntimeEnsemble


def save_model(path: str | Path, model: RuntimeEnsemble, metadata: dict[str, Any]) -> None:
    path = Path(path)
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    staged_path = Path(staged_name)
    try:
        joblib.dump(model, staged_path)
        os.replace(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)

    path.with_suffix(path.suffix + ".metadata.json").write_text(
        metadata_text,
        encoding="utf-8",
    )


def load_model(path: str | Path) -> RuntimeEnsemble:
    model = joblib.load(Path(path))
    if not isinstance(model, RuntimeEnsemble):
        raise TypeError("artifact is not an Aster RuntimeEnsemble")
    return model
