from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from .records import PlanObservation


def append_observations(path: str | Path, observations: Iterable[PlanObservation]) -> int:
    path = Path(path)
    count = 0
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as staged:
        for observation in observations:
            staged.write(json.dumps(observation.to_jsonable(), sort_keys=True) + "\n")
            count += 1
        staged.seek(0)

        path.parent.mkdir(parents=True, exist_ok=True)
        replacement_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+t",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as replacement:
                replacement_path = Path(replacement.name)
                if path.exists():
                    with path.open("r", encoding="utf-8") as existing:
                        while chunk := existing.read(1024 * 1024):
                            replacement.write(chunk)
                shutil.copyfileobj(staged, replacement)
                replacement.flush()
                os.fsync(replacement.fileno())

            os.replace(replacement_path, path)
            replacement_path = None
        finally:
            if replacement_path is not None:
                replacement_path.unlink(missing_ok=True)
    return count
