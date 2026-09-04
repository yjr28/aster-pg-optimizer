from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .records import PlanObservation


def append_observations(path: str | Path, observations: Iterable[PlanObservation]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_jsonable(), sort_keys=True) + "\n")
            count += 1
    return count
