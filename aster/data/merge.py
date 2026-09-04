from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .integrity import audit_dataset
from .load import read_jsonl


@dataclass(frozen=True)
class CorpusInput:
    path: str
    sha256: str
    observations: int
    workloads: tuple[str, ...]
    dataset_versions: tuple[str, ...]
    environments: tuple[str, ...]


@dataclass(frozen=True)
class CombinedDataset:
    output_path: str
    sha256: str
    observations: int
    experiments: int
    queries: int
    query_templates: int
    unique_query_plans: int
    workloads: tuple[str, ...]
    dataset_versions: tuple[str, ...]
    environments: tuple[str, ...]
    inputs: tuple[CorpusInput, ...]


def _record_identity(record: dict) -> tuple:
    provenance=record.get("provenance") or {}
    return (
        provenance.get("experiment_id"),
        provenance.get("environment_sha256") or "",
        provenance.get("dataset_version"),
        provenance.get("workload"),
        provenance.get("query_id"),
        provenance.get("candidate_id"),
        record.get("plan_fingerprint"),
        record.get("repetition"),
    )


def combine_datasets(
    paths: list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
    *,
    overwrite: bool = False,
    require_multiple_workloads: bool = False,
) -> CombinedDataset:
    if len(paths) < 1:
        raise ValueError("at least one input dataset is required")
    output=Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite combined dataset: {output}")

    input_metadata: list[CorpusInput]=[]
    input_records: list[tuple[Path,list[dict]]]=[]
    all_workloads: set[str]=set()
    all_versions: set[str]=set()
    all_environments: set[str]=set()
    for raw_path in paths:
        path=Path(raw_path)
        audit=audit_dataset(path)
        if not audit.ok:
            raise ValueError(f"input dataset failed integrity audit {path}: {'; '.join(audit.errors)}")
        records=read_jsonl(path)
        workloads=sorted({str(row["provenance"]["workload"]) for row in records})
        versions=sorted({str(row["provenance"]["dataset_version"]) for row in records})
        environments=sorted({
            str(row["provenance"].get("environment_sha256"))
            for row in records
            if row["provenance"].get("environment_sha256")
        })
        all_workloads.update(workloads)
        all_versions.update(versions)
        all_environments.update(environments)
        input_metadata.append(CorpusInput(
            path=str(path),
            sha256=audit.sha256,
            observations=audit.observations,
            workloads=tuple(workloads),
            dataset_versions=tuple(versions),
            environments=tuple(environments),
        ))
        input_records.append((path,records))

    if require_multiple_workloads and len(all_workloads) < 2:
        raise ValueError("cross-workload corpus requires at least two distinct workloads")

    seen: dict[tuple,Path]={}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+".tmp")
    try:
        with tmp.open("w",encoding="utf-8") as target:
            for path,records in input_records:
                for row in records:
                    identity=_record_identity(row)
                    if identity in seen:
                        raise ValueError(
                            "duplicate observation identity across corpora: "
                            f"current={path} first={seen[identity]} identity={identity}"
                        )
                    seen[identity]=path
                    target.write(json.dumps(row,sort_keys=True)+"\n")
            target.flush(); os.fsync(target.fileno())
        combined_audit=audit_dataset(tmp)
        if not combined_audit.ok:
            raise ValueError("combined dataset failed integrity audit: "+"; ".join(combined_audit.errors))
        os.replace(tmp,output)
    finally:
        if tmp.exists(): tmp.unlink()

    result=CombinedDataset(
        output_path=str(output),
        sha256=combined_audit.sha256,
        observations=combined_audit.observations,
        experiments=combined_audit.experiments,
        queries=combined_audit.queries,
        query_templates=combined_audit.query_templates,
        unique_query_plans=combined_audit.unique_query_plans,
        workloads=tuple(sorted(all_workloads)),
        dataset_versions=tuple(sorted(all_versions)),
        environments=tuple(sorted(all_environments)),
        inputs=tuple(input_metadata),
    )
    output.with_suffix(output.suffix+".manifest.json").write_text(
        json.dumps(asdict(result),indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    return result
