from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_JOB_QUERY_RE = re.compile(r"^(?P<family>[1-9][0-9]*)(?P<variant>[a-z])\.sql$")
EXPECTED_JOB_QUERY_COUNT = 113
EXPECTED_JOB_FAMILY_COUNT = 33


@dataclass(frozen=True)
class JobQuery:
    query_id: str
    family: str
    variant: str
    path: Path
    sql: str
    sha256: str


@dataclass(frozen=True)
class JobWorkloadManifest:
    query_count: int
    family_count: int
    workload_sha256: str
    query_ids: tuple[str, ...]
    query_sha256: tuple[tuple[str, str], ...]


def _sort_key(query: JobQuery) -> tuple[int, str]:
    return int(query.family), query.variant


def load_job_queries(query_dir: str | Path, *, strict: bool = True) -> tuple[JobQuery, ...]:
    """Load a local Join Order Benchmark query directory without redistributing JOB.

    Aster deliberately does not vendor JOB SQL. Users point this loader at a local
    benchmark checkout so the benchmark source/license remains external and explicit.
    """
    root = Path(query_dir)
    if not root.is_dir():
        raise ValueError(f"JOB query directory does not exist: {root}")

    queries: list[JobQuery] = []
    seen_ids: set[str] = set()
    for path in root.iterdir():
        if not path.is_file():
            continue
        match = _JOB_QUERY_RE.fullmatch(path.name)
        if not match:
            continue
        query_id = f"{match.group('family')}{match.group('variant')}"
        if query_id in seen_ids:
            raise ValueError(f"duplicate JOB query id: {query_id}")
        sql = path.read_text(encoding="utf-8").strip()
        if not sql:
            raise ValueError(f"JOB query is empty: {path}")
        seen_ids.add(query_id)
        queries.append(JobQuery(
            query_id=query_id,
            family=match.group("family"),
            variant=match.group("variant"),
            path=path,
            sql=sql,
            sha256=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        ))

    queries.sort(key=_sort_key)
    families = {query.family for query in queries}
    if strict:
        if len(queries) != EXPECTED_JOB_QUERY_COUNT:
            raise ValueError(
                f"expected {EXPECTED_JOB_QUERY_COUNT} JOB queries, found {len(queries)}"
            )
        if len(families) != EXPECTED_JOB_FAMILY_COUNT:
            raise ValueError(
                f"expected {EXPECTED_JOB_FAMILY_COUNT} JOB families, found {len(families)}"
            )
    if not queries:
        raise ValueError(f"no JOB query files matching '<family><variant>.sql' found in {root}")
    return tuple(queries)


def build_job_manifest(query_dir: str | Path, *, strict: bool = True) -> JobWorkloadManifest:
    queries = load_job_queries(query_dir, strict=strict)
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query.sha256.encode("ascii"))
        digest.update(b"\n")
    return JobWorkloadManifest(
        query_count=len(queries),
        family_count=len({query.family for query in queries}),
        workload_sha256=digest.hexdigest(),
        query_ids=tuple(query.query_id for query in queries),
        query_sha256=tuple((query.query_id, query.sha256) for query in queries),
    )
