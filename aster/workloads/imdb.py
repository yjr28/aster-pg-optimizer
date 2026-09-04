from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

EXPECTED_IMDB_TABLES = (
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
)


@dataclass(frozen=True)
class DatasetFile:
    table: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ImdbDatasetManifest:
    file_count: int
    total_bytes: int
    dataset_sha256: str
    files: tuple[DatasetFile, ...]

    @property
    def dataset_version(self) -> str:
        return f"job-imdb-sha256:{self.dataset_sha256[:16]}"


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_imdb_manifest(csv_dir: str | Path, *, strict: bool = True) -> ImdbDatasetManifest:
    """Fingerprint the local 21-table IMDB snapshot used by JOB.

    The data itself is external to Aster. Hashing the actual CSV bytes makes dataset
    identity explicit so runs from different IMDB snapshots cannot be combined under
    the same vague workload name.
    """
    root = Path(csv_dir)
    if not root.is_dir():
        raise ValueError(f"IMDB CSV directory does not exist: {root}")

    files: list[DatasetFile] = []
    missing: list[str] = []
    for table in EXPECTED_IMDB_TABLES:
        path = root / f"{table}.csv"
        if not path.is_file():
            missing.append(table)
            continue
        files.append(DatasetFile(
            table=table,
            filename=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        ))

    if strict and missing:
        raise ValueError(f"missing JOB IMDB CSV tables: {', '.join(missing)}")
    if not files:
        raise ValueError(f"no recognized JOB IMDB CSV files found in {root}")

    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry.table.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")

    return ImdbDatasetManifest(
        file_count=len(files),
        total_bytes=sum(entry.size_bytes for entry in files),
        dataset_sha256=digest.hexdigest(),
        files=tuple(files),
    )
