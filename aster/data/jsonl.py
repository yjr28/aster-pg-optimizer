from __future__ import annotations

import json
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
        with path.open("a", encoding="utf-8") as handle:
            shutil.copyfileobj(staged, handle)
    return count
