from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from aster.data import audit_dataset

_QUERY_SHARD_RE = re.compile(r"^(?P<family>[1-9][0-9]*)(?P<variant>[a-z])\.jsonl$")


@dataclass(frozen=True)
class FinalizedJobDataset:
    experiment_id: str
    dataset_version: str
    benchmark_input_sha256: str
    query_count: int
    observation_count: int
    unique_query_plans: int
    dataset_sha256: str
    output_path: str


def _shard_sort_key(path: Path) -> tuple[int, str]:
    match = _QUERY_SHARD_RE.fullmatch(path.name)
    if not match:
        return (10**9, path.name)
    return int(match.group("family")), match.group("variant")


def _load_collection_manifest(root: Path) -> dict:
    path = root / "collection_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid collection manifest: {path}") from exc


def finalize_job_collection(
    collection_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> FinalizedJobDataset:
    root = Path(collection_dir)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite finalized dataset: {output}")

    manifest = _load_collection_manifest(root)
    try:
        config = manifest["config"]
        summary = manifest["summary"]
        experiment_id = str(config["experiment_id"])
        dataset_version = str(config["dataset_version"])
        benchmark_input_sha256 = str(config["benchmark_input_sha256"])
        expected_queries = int(summary["query_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("collection manifest is missing required config/summary fields") from exc

    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    if failures:
        raise ValueError(f"cannot finalize with unresolved query failures: {[p.stem for p in failures]}")

    shards = sorted((root / "queries").glob("*.jsonl"), key=_shard_sort_key)
    if len(shards) != expected_queries:
        raise ValueError(f"expected {expected_queries} completed query shards, found {len(shards)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as target:
            for shard in shards:
                shard_query_id = shard.stem
                with shard.open("r", encoding="utf-8") as source:
                    rows = 0
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            provenance = row["provenance"]
                        except (json.JSONDecodeError, KeyError, TypeError) as exc:
                            raise ValueError(f"invalid observation in {shard}:{line_number}") from exc
                        if provenance.get("experiment_id") != experiment_id:
                            raise ValueError(f"mixed experiment in {shard}")
                        if provenance.get("dataset_version") != dataset_version:
                            raise ValueError(f"mixed dataset version in {shard}")
                        if provenance.get("workload") != "job":
                            raise ValueError(f"non-JOB workload record in {shard}")
                        if provenance.get("query_id") != shard_query_id:
                            raise ValueError(f"query id mismatch in {shard}")
                        target.write(line if line.endswith("\n") else line + "\n")
                        rows += 1
                    if rows == 0:
                        raise ValueError(f"empty completed shard: {shard}")
            target.flush()
            os.fsync(target.fileno())

        audit = audit_dataset(tmp)
        if not audit.ok:
            raise ValueError("finalized dataset failed integrity audit: " + "; ".join(audit.errors))
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()

    result = FinalizedJobDataset(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        benchmark_input_sha256=benchmark_input_sha256,
        query_count=expected_queries,
        observation_count=audit.observations,
        unique_query_plans=audit.unique_query_plans,
        dataset_sha256=audit.sha256,
        output_path=str(output),
    )
    finalized_manifest = {
        "schema_version": 1,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": asdict(result),
        "integrity": asdict(audit),
    }
    (root / "finalized_manifest.json").write_text(
        json.dumps(finalized_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
