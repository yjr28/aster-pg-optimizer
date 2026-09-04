from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from aster.plans import parse_explain_json, plan_fingerprint

from .load import read_jsonl


@dataclass(frozen=True)
class DatasetIntegrityReport:
    sha256: str
    observations: int
    experiments: int
    queries: int
    query_templates: int
    unique_query_plans: int
    min_repetitions_per_plan: int
    max_repetitions_per_plan: int
    missing_template_records: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


_REQUIRED_PROVENANCE = (
    "experiment_id", "workload", "query_id", "candidate_id", "postgres_version",
    "dataset_version", "run_seed", "code_revision", "captured_at_utc",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(path: str | Path) -> DatasetIntegrityReport:
    """Audit raw JSONL observations before model training."""
    path = Path(path)
    records = read_jsonl(path)
    errors: list[str] = []
    warnings: list[str] = []
    missing_templates = 0
    experiments: set[str] = set()
    queries: set[tuple[str, str, str]] = set()
    templates: set[str] = set()
    query_plans: set[tuple[str, str, str, str]] = set()
    repetition_keys: set[tuple[str, str, str, str, str, int]] = set()
    repetition_groups: dict[tuple[str, str, str, str, str], set[int]] = defaultdict(set)
    drift_groups: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    for index, record in enumerate(records, start=1):
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"record {index}: provenance must be an object")
            continue
        missing = [field for field in _REQUIRED_PROVENANCE if not provenance.get(field)]
        if missing:
            errors.append(f"record {index}: missing provenance fields {missing}")
            continue
        experiment_id = str(provenance["experiment_id"])
        dataset_version = str(provenance["dataset_version"])
        workload = str(provenance["workload"])
        query_id = str(provenance["query_id"])
        candidate_id = str(provenance["candidate_id"])
        template = provenance.get("query_template")
        experiments.add(experiment_id)
        queries.add((dataset_version, workload, query_id))
        if template:
            templates.add(str(template))
        else:
            missing_templates += 1

        runtime = record.get("execution_time_ms")
        if not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)) or runtime <= 0:
            errors.append(f"record {index}: execution_time_ms must be finite and > 0")
        repetition = record.get("repetition")
        if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 0:
            errors.append(f"record {index}: repetition must be a non-negative integer")
            continue
        stored_fp = record.get("plan_fingerprint")
        if not isinstance(stored_fp, str) or not stored_fp:
            errors.append(f"record {index}: plan_fingerprint must be a non-empty string")
            continue
        try:
            actual_fp = plan_fingerprint(parse_explain_json(record.get("plan_json")))
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"record {index}: invalid plan_json ({exc})")
            continue
        if actual_fp != stored_fp:
            errors.append(f"record {index}: stored fingerprint {stored_fp[:12]} does not match plan {actual_fp[:12]}")

        query_plans.add((dataset_version, workload, query_id, stored_fp))
        rep_key = (experiment_id, workload, query_id, candidate_id, stored_fp, repetition)
        if rep_key in repetition_keys:
            errors.append(f"record {index}: duplicate repetition for experiment={experiment_id} query={query_id} candidate={candidate_id} repetition={repetition}")
        repetition_keys.add(rep_key)
        group_key = (experiment_id, workload, query_id, candidate_id, stored_fp)
        repetition_groups[group_key].add(repetition)
        drift_key = (experiment_id, dataset_version, workload, query_id, candidate_id)
        drift_groups[drift_key].add(stored_fp)

    for group, repetitions in repetition_groups.items():
        expected = set(range(max(repetitions, default=-1) + 1))
        if repetitions != expected:
            errors.append(f"non-contiguous repetitions for experiment={group[0]} query={group[2]} candidate={group[3]}: observed={sorted(repetitions)} expected={sorted(expected)}")
    for group, fingerprints in drift_groups.items():
        if len(fingerprints) > 1:
            errors.append(f"candidate plan drift within one experiment: experiment={group[0]} query={group[3]} candidate={group[4]} fingerprints={sorted(fp[:12] for fp in fingerprints)}")
    if missing_templates:
        warnings.append(f"{missing_templates} observation(s) have no query_template; template-holdout training cannot use those records")
    if records and not templates:
        warnings.append("dataset has no query templates")
    repetition_counts = [len(values) for values in repetition_groups.values()]
    return DatasetIntegrityReport(
        sha256=_file_sha256(path), observations=len(records), experiments=len(experiments),
        queries=len(queries), query_templates=len(templates), unique_query_plans=len(query_plans),
        min_repetitions_per_plan=min(repetition_counts, default=0),
        max_repetitions_per_plan=max(repetition_counts, default=0),
        missing_template_records=missing_templates, errors=tuple(errors), warnings=tuple(warnings),
    )
