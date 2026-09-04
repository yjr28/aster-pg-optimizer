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
    """Aggregate repeated observations to one median-runtime physical-plan example.

    Environment identity is part of the aggregation key. The same physical plan measured
    under two PostgreSQL/index/statistics/hardware snapshots must never be averaged into
    one label.
    """
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(path):
        provenance = record["provenance"]
        environment_sha256 = str(provenance.get("environment_sha256") or "")
        key = (
            provenance["dataset_version"],
            provenance["workload"],
            provenance["query_id"],
            provenance["candidate_id"],
            record["plan_fingerprint"],
            environment_sha256,
        )
        groups[key].append(record)
    examples: list[TrainingExample] = []
    for (dataset_version, workload, query_id, candidate_id, _, environment_sha256), rows in groups.items():
        runtimes = [float(row["execution_time_ms"]) for row in rows]
        provenance = rows[0]["provenance"]
        examples.append(TrainingExample(
            plan=parse_explain_json(rows[0]["plan_json"]), runtime_ms=float(statistics.median(runtimes)),
            query_id=query_id, candidate_id=candidate_id,
            query_template=provenance.get("query_template"), parameter_key=provenance.get("parameter_key"),
            workload=workload, dataset_version=dataset_version,
            environment_sha256=environment_sha256 or None,
        ))
    return examples
