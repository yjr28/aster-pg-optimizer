from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from .job import load_job_queries


class WorkloadCollector(Protocol):
    def discover(self, query: str, candidates): ...
    def measure(self, query: str, candidate, **kwargs): ...


@dataclass(frozen=True)
class JobCollectionConfig:
    experiment_id: str
    dataset_version: str
    benchmark_input_sha256: str
    workload_sha256: str
    run_seed: int
    code_revision: str
    warmups: int = 1
    repetitions: int = 3

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.dataset_version:
            raise ValueError("dataset_version is required")
        if len(self.benchmark_input_sha256) != 64 or len(self.workload_sha256) != 64:
            raise ValueError("benchmark/workload identities must be SHA-256 hex strings")
        if self.warmups < 0 or self.repetitions < 1:
            raise ValueError("warmups must be >=0 and repetitions >=1")


@dataclass(frozen=True)
class JobCollectionSummary:
    query_count: int
    completed_queries: int
    skipped_queries: int
    failed_queries: int
    observations_written: int
    unique_plans_measured: int
    failure_query_ids: tuple[str, ...]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_observation_shard(path: Path, observations: Iterable) -> int:
    rows = [json.dumps(observation.to_jsonable(), sort_keys=True) for observation in observations]
    if not rows:
        raise RuntimeError("refusing to write an empty completed query shard")
    _atomic_write(path, "\n".join(rows) + "\n")
    return len(rows)


def _validate_completed_shard(path: Path, *, query_id: str, experiment_id: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                provenance = row["provenance"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid completed shard {path}:{line_number}") from exc
            if provenance.get("query_id") != query_id:
                raise ValueError(f"completed shard {path} contains another query id")
            if provenance.get("experiment_id") != experiment_id:
                raise ValueError(f"completed shard {path} belongs to another experiment")
            count += 1
    if count == 0:
        raise ValueError(f"completed shard is empty: {path}")
    return count


def collect_job_workload(
    collector: WorkloadCollector,
    query_dir: str | Path,
    output_dir: str | Path,
    *,
    config: JobCollectionConfig,
    candidates,
    strict_workload: bool = True,
    resume: bool = True,
    fail_fast: bool = False,
) -> JobCollectionSummary:
    queries = load_job_queries(query_dir, strict=strict_workload)
    root = Path(output_dir)
    shard_dir = root / "queries"
    failure_dir = root / "failures"

    completed = skipped = failed = observations_written = unique_plans = 0
    failure_ids: list[str] = []

    for query in queries:
        shard = shard_dir / f"{query.query_id}.jsonl"
        if resume and shard.exists():
            _validate_completed_shard(
                shard, query_id=query.query_id, experiment_id=config.experiment_id
            )
            skipped += 1
            continue

        try:
            discovered = collector.discover(query.sql, candidates)
            if not discovered:
                raise RuntimeError("candidate discovery produced no physical plans")
            observations = []
            for candidate in discovered:
                observations.extend(collector.measure(
                    query.sql,
                    candidate,
                    workload="job",
                    query_id=query.query_id,
                    query_template=f"job-family-{query.family}",
                    parameter_key=query.variant,
                    dataset_version=config.dataset_version,
                    run_seed=config.run_seed,
                    code_revision=config.code_revision,
                    experiment_id=config.experiment_id,
                    warmups=config.warmups,
                    repetitions=config.repetitions,
                ))
            count = _write_observation_shard(shard, observations)
            observations_written += count
            unique_plans += len(discovered)
            completed += 1
            failure_path = failure_dir / f"{query.query_id}.json"
            if failure_path.exists():
                failure_path.unlink()
        except Exception as exc:
            failed += 1
            failure_ids.append(query.query_id)
            failure = {
                "query_id": query.query_id,
                "family": query.family,
                "variant": query.variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(
                failure_dir / f"{query.query_id}.json",
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
            )
            if fail_fast:
                raise

    summary = JobCollectionSummary(
        query_count=len(queries),
        completed_queries=completed,
        skipped_queries=skipped,
        failed_queries=failed,
        observations_written=observations_written,
        unique_plans_measured=unique_plans,
        failure_query_ids=tuple(failure_ids),
    )
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "summary": asdict(summary),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(root / "collection_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary
