from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from aster.models import TrainingExample
from aster.plans import parse_explain_json


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            records.append(record)
    return records


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    """Aggregate repeated observations to one median-runtime plan example.

    Grouping includes structural fingerprint so plan drift is never silently averaged
    into one label.
    """
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        provenance = record["provenance"]
        key = (
            provenance["workload"],
            provenance["query_id"],
            provenance["candidate_id"],
            record["plan_fingerprint"],
        )
        groups[key].append(record)

    examples: list[TrainingExample] = []
    for (_, query_id, candidate_id, _), rows in groups.items():
        runtimes = [float(row["execution_time_ms"]) for row in rows]
        first = rows[0]
        provenance = first["provenance"]
        plan_json = first["plan_json"]
        examples.append(
            TrainingExample(
                plan=parse_explain_json(plan_json),
                runtime_ms=float(statistics.median(runtimes)),
                query_id=query_id,
                candidate_id=candidate_id,
                query_template=provenance.get("query_template"),
                parameter_key=provenance.get("parameter_key"),
            )
        )
    return examples
