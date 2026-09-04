from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .imdb import build_imdb_manifest
from .job import build_job_manifest
from .tpch import build_tpch_dataset_manifest, build_tpch_workload_manifest


@dataclass(frozen=True)
class JobPreflightManifest:
    query_count: int
    family_count: int
    workload_sha256: str
    dataset_file_count: int
    dataset_total_bytes: int
    dataset_sha256: str
    dataset_version: str
    benchmark_input_sha256: str


@dataclass(frozen=True)
class TpchPreflightManifest:
    specification_version: str
    scale_factor: float | None
    query_count: int
    workload_sha256: str
    dataset_file_count: int
    dataset_total_bytes: int
    dataset_sha256: str
    dataset_version: str
    benchmark_input_sha256: str


def build_job_preflight(
    query_dir: str | Path,
    csv_dir: str | Path,
    *,
    strict: bool = True,
) -> JobPreflightManifest:
    workload = build_job_manifest(query_dir, strict=strict)
    dataset = build_imdb_manifest(csv_dir, strict=strict)
    digest = hashlib.sha256()
    digest.update(b"aster-job-preflight-v1\n")
    digest.update(workload.workload_sha256.encode("ascii"))
    digest.update(b"\n")
    digest.update(dataset.dataset_sha256.encode("ascii"))
    digest.update(b"\n")
    return JobPreflightManifest(
        query_count=workload.query_count,
        family_count=workload.family_count,
        workload_sha256=workload.workload_sha256,
        dataset_file_count=dataset.file_count,
        dataset_total_bytes=dataset.total_bytes,
        dataset_sha256=dataset.dataset_sha256,
        dataset_version=dataset.dataset_version,
        benchmark_input_sha256=digest.hexdigest(),
    )


def build_tpch_preflight(
    query_dir: str | Path,
    data_dir: str | Path,
    *,
    scale_factor: float | None,
    specification_version: str = "3.0.1",
    strict: bool = True,
) -> TpchPreflightManifest:
    if not specification_version.strip():
        raise ValueError("specification_version is required")
    workload = build_tpch_workload_manifest(query_dir, strict=strict)
    dataset = build_tpch_dataset_manifest(
        data_dir,
        scale_factor=scale_factor,
        strict=strict,
    )
    digest = hashlib.sha256()
    digest.update(b"aster-tpch-preflight-v1\n")
    digest.update(specification_version.strip().encode("utf-8"))
    digest.update(b"\n")
    digest.update(workload.workload_sha256.encode("ascii"))
    digest.update(b"\n")
    digest.update(dataset.dataset_sha256.encode("ascii"))
    digest.update(b"\n")
    digest.update(dataset.dataset_version.encode("utf-8"))
    digest.update(b"\n")
    return TpchPreflightManifest(
        specification_version=specification_version.strip(),
        scale_factor=scale_factor,
        query_count=workload.query_count,
        workload_sha256=workload.workload_sha256,
        dataset_file_count=dataset.file_count,
        dataset_total_bytes=dataset.total_bytes,
        dataset_sha256=dataset.dataset_sha256,
        dataset_version=dataset.dataset_version,
        benchmark_input_sha256=digest.hexdigest(),
    )
