from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

EXPECTED_TPCH_QUERY_COUNT = 22
EXPECTED_TPCH_TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)
_TPCH_QUERY_RE = re.compile(r"^(?:q)?(?P<number>[1-9]|1[0-9]|2[0-2])\.sql$", re.IGNORECASE)


@dataclass(frozen=True)
class TpchQuery:
    query_id: str
    number: int
    path: Path
    sql: str
    sha256: str


@dataclass(frozen=True)
class TpchWorkloadManifest:
    query_count: int
    workload_sha256: str
    query_ids: tuple[str, ...]
    query_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TpchDatasetFile:
    table: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class TpchDatasetManifest:
    file_count: int
    total_bytes: int
    dataset_sha256: str
    scale_factor: float | None
    files: tuple[TpchDatasetFile, ...]

    @property
    def dataset_version(self) -> str:
        scale = "unknown" if self.scale_factor is None else f"{self.scale_factor:g}"
        return f"tpch-sf{scale}-sha256:{self.dataset_sha256[:16]}"


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_tpch_queries(query_dir: str | Path, *, strict: bool = True) -> tuple[TpchQuery, ...]:
    """Load executable TPC-H SQL supplied by the user/tool checkout.

    Aster does not redistribute TPC-H query text. The loader accepts q1.sql..q22.sql
    or 1.sql..22.sql, fingerprints the exact executable SQL, and keeps that identity in
    benchmark provenance.
    """
    root = Path(query_dir)
    if not root.is_dir():
        raise ValueError(f"TPC-H query directory does not exist: {root}")
    queries: list[TpchQuery] = []
    seen: set[int] = set()
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _TPCH_QUERY_RE.fullmatch(path.name)
        if not match:
            continue
        number = int(match.group("number"))
        if number in seen:
            raise ValueError(f"duplicate TPC-H query number: {number}")
        sql = path.read_text(encoding="utf-8").strip()
        if not sql:
            raise ValueError(f"TPC-H query is empty: {path}")
        seen.add(number)
        queries.append(TpchQuery(
            query_id=f"q{number}",
            number=number,
            path=path,
            sql=sql,
            sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        ))
    queries.sort(key=lambda query: query.number)
    if strict and len(queries) != EXPECTED_TPCH_QUERY_COUNT:
        raise ValueError(f"expected {EXPECTED_TPCH_QUERY_COUNT} TPC-H queries, found {len(queries)}")
    if not queries:
        raise ValueError(f"no TPC-H query files matching q1.sql..q22.sql found in {root}")
    return tuple(queries)


def build_tpch_workload_manifest(query_dir: str | Path, *, strict: bool = True) -> TpchWorkloadManifest:
    queries = load_tpch_queries(query_dir, strict=strict)
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(query.sha256.encode("ascii"))
        digest.update(b"\n")
    return TpchWorkloadManifest(
        query_count=len(queries),
        workload_sha256=digest.hexdigest(),
        query_ids=tuple(query.query_id for query in queries),
        query_sha256=tuple((query.query_id, query.sha256) for query in queries),
    )


def build_tpch_dataset_manifest(
    data_dir: str | Path,
    *,
    scale_factor: float | None = None,
    strict: bool = True,
) -> TpchDatasetManifest:
    """Fingerprint external dbgen output for the eight TPC-H base tables.

    dbgen conventionally emits `<table>.tbl`; CSV files are also accepted for users who
    convert the data before PostgreSQL loading. If both exist for one table, the loader
    refuses the ambiguous snapshot rather than choosing silently.
    """
    if scale_factor is not None and scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"TPC-H data directory does not exist: {root}")

    files: list[TpchDatasetFile] = []
    missing: list[str] = []
    for table in EXPECTED_TPCH_TABLES:
        candidates = [root / f"{table}.tbl", root / f"{table}.csv"]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) > 1:
            raise ValueError(f"ambiguous TPC-H data files for {table}: {[path.name for path in existing]}")
        if not existing:
            missing.append(table)
            continue
        path = existing[0]
        files.append(TpchDatasetFile(
            table=table,
            filename=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        ))
    if strict and missing:
        raise ValueError(f"missing TPC-H tables: {', '.join(missing)}")
    if not files:
        raise ValueError(f"no recognized TPC-H table files found in {root}")

    digest = hashlib.sha256()
    for entry in files:
        digest.update(entry.table.encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.filename.rsplit(".", 1)[-1].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")
    return TpchDatasetManifest(
        file_count=len(files),
        total_bytes=sum(entry.size_bytes for entry in files),
        dataset_sha256=digest.hexdigest(),
        scale_factor=scale_factor,
        files=tuple(files),
    )
