from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from aster.data import audit_dataset

_JOB_QUERY_SHARD_RE = re.compile(r"^(?P<family>[1-9][0-9]*)(?P<variant>[a-z])\.jsonl$")
_TPCH_QUERY_SHARD_RE = re.compile(r"^q(?P<number>[1-9]|1[0-9]|2[0-2])\.jsonl$")


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


@dataclass(frozen=True)
class FinalizedTpchDataset:
    experiment_id: str
    dataset_version: str
    benchmark_input_sha256: str
    specification_version: str
    scale_factor: float | None
    query_count: int
    observation_count: int
    unique_query_plans: int
    dataset_sha256: str
    output_path: str


def _job_shard_sort_key(path: Path) -> tuple[int, str]:
    match = _JOB_QUERY_SHARD_RE.fullmatch(path.name)
    if not match:
        return (10**9, path.name)
    return int(match.group("family")), match.group("variant")


def _tpch_shard_sort_key(path: Path) -> tuple[int, str]:
    match = _TPCH_QUERY_SHARD_RE.fullmatch(path.name)
    if not match:
        return (10**9, path.name)
    return int(match.group("number")), ""


def _load_collection_manifest(root: Path) -> dict:
    path = root / "collection_manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid collection manifest: {path}") from exc


def _merge_and_audit_collection(
    collection_dir: str | Path,
    output_path: str | Path,
    *,
    expected_workload: str,
    shard_sort_key: Callable[[Path], tuple[int, str]],
    overwrite: bool,
):
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

    shards = sorted((root / "queries").glob("*.jsonl"), key=shard_sort_key)
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
                        if provenance.get("workload") != expected_workload:
                            raise ValueError(f"non-{expected_workload} workload record in {shard}")
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

    common = {
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "benchmark_input_sha256": benchmark_input_sha256,
        "query_count": expected_queries,
        "observation_count": audit.observations,
        "unique_query_plans": audit.unique_query_plans,
        "dataset_sha256": audit.sha256,
        "output_path": str(output),
    }
    return root, manifest, common, audit


def _write_finalized_manifest(root: Path, result, audit) -> None:
    finalized_manifest = {
        "schema_version": 1,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": asdict(result),
        "integrity": asdict(audit),
    }
    (root / "finalized_manifest.json").write_text(
        json.dumps(finalized_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_job_collection(
    collection_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> FinalizedJobDataset:
    root, _manifest, common, audit = _merge_and_audit_collection(
        collection_dir,
        output_path,
        expected_workload="job",
        shard_sort_key=_job_shard_sort_key,
        overwrite=overwrite,
    )
    result = FinalizedJobDataset(**common)
    _write_finalized_manifest(root, result, audit)
    return result


def finalize_tpch_collection(
    collection_dir: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> FinalizedTpchDataset:
    root, manifest, common, audit = _merge_and_audit_collection(
        collection_dir,
        output_path,
        expected_workload="tpch",
        shard_sort_key=_tpch_shard_sort_key,
        overwrite=overwrite,
    )
    try:
        config = manifest["config"]
        specification_version = str(config["specification_version"])
        scale_factor = config.get("scale_factor")
        if scale_factor is not None:
            scale_factor = float(scale_factor)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TPC-H collection manifest is missing specification metadata") from exc
    result = FinalizedTpchDataset(
        specification_version=specification_version,
        scale_factor=scale_factor,
        **common,
    )
    _write_finalized_manifest(root, result, audit)
    return result
