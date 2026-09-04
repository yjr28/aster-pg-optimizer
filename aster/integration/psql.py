from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Protocol

from aster.planner import render_set_local


class ExplainRunner(Protocol):
    def explain(self, query: str, settings: dict[str, str], *, analyze: bool) -> list[dict]: ...
    def postgres_version(self) -> str: ...


class PsqlExplainRunner:
    """Run EXPLAIN through PostgreSQL's psql client.

    Measured queries execute inside READ ONLY transactions. This prevents ordinary
    writes but is not a sandbox: expensive reads and volatile external functions can
    still be unsafe. Use benchmark databases, never production.
    """

    def __init__(self, dsn: str, *, psql_bin: str = "psql", timeout_s: int = 120):
        self.dsn = dsn
        self.psql_bin = psql_bin
        self.timeout_s = timeout_s

    def _run(self, sql: str) -> str:
        env = os.environ.copy()
        proc = subprocess.run(
            [self.psql_bin, self.dsn, "-X", "-qAt", "-v", "ON_ERROR_STOP=1"],
            input=sql,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"psql failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def postgres_version(self) -> str:
        return self._run("SHOW server_version;").splitlines()[-1]

    def benchmark_catalog_snapshot(self) -> dict:
        """Return fixed read-only PostgreSQL state needed to interpret benchmarks.

        The SQL is intentionally static: callers cannot inject arbitrary catalog queries.
        Schema/index/statistics rows are sorted before JSON aggregation so the resulting
        snapshot is stable enough to hash as part of benchmark provenance.
        """
        sql = r"""
BEGIN TRANSACTION READ ONLY;
SELECT json_build_object(
  'server_version', current_setting('server_version'),
  'server_version_num', current_setting('server_version_num'),
  'database', current_database(),
  'database_size_bytes', pg_database_size(current_database()),
  'settings', (
    SELECT COALESCE(json_object_agg(name, value ORDER BY name), '{}'::json)
    FROM (
      SELECT name,
             json_build_object('setting', setting, 'unit', unit, 'source', source) AS value
      FROM pg_settings
      WHERE name = ANY (ARRAY[
        'shared_buffers','work_mem','effective_cache_size','random_page_cost',
        'seq_page_cost','cpu_tuple_cost','cpu_index_tuple_cost','cpu_operator_cost',
        'max_parallel_workers_per_gather','jit','geqo','geqo_threshold','geqo_effort',
        'default_statistics_target','effective_io_concurrency'
      ])
      ORDER BY name
    ) s
  ),
  'relations', (
    SELECT COALESCE(json_agg(row_to_json(r) ORDER BY r.schema_name, r.relation_name), '[]'::json)
    FROM (
      SELECT n.nspname AS schema_name,
             c.relname AS relation_name,
             c.relkind,
             c.relpersistence,
             c.reltuples::double precision AS estimated_rows,
             c.relpages::bigint AS pages
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE c.relkind IN ('r','p')
        AND n.nspname NOT IN ('pg_catalog','information_schema')
        AND n.nspname NOT LIKE 'pg_toast%'
      ORDER BY n.nspname, c.relname
    ) r
  ),
  'indexes', (
    SELECT COALESCE(json_agg(row_to_json(i) ORDER BY i.schema_name, i.table_name, i.index_name), '[]'::json)
    FROM (
      SELECT schemaname AS schema_name,
             tablename AS table_name,
             indexname AS index_name,
             indexdef AS index_definition
      FROM pg_indexes
      WHERE schemaname NOT IN ('pg_catalog','information_schema')
      ORDER BY schemaname, tablename, indexname
    ) i
  ),
  'statistics_state', (
    SELECT COALESCE(json_agg(row_to_json(st) ORDER BY st.schema_name, st.relation_name), '[]'::json)
    FROM (
      SELECT schemaname AS schema_name,
             relname AS relation_name,
             n_live_tup::bigint,
             n_dead_tup::bigint,
             last_analyze,
             last_autoanalyze,
             analyze_count::bigint,
             autoanalyze_count::bigint
      FROM pg_stat_all_tables
      WHERE schemaname NOT IN ('pg_catalog','information_schema')
        AND schemaname NOT LIKE 'pg_toast%'
      ORDER BY schemaname, relname
    ) st
  ),
  'statistics_targets', (
    SELECT COALESCE(json_agg(row_to_json(t) ORDER BY t.schema_name, t.relation_name, t.column_name), '[]'::json)
    FROM (
      SELECT n.nspname AS schema_name,
             c.relname AS relation_name,
             a.attname AS column_name,
             a.attstattarget AS statistics_target
      FROM pg_attribute a
      JOIN pg_class c ON c.oid = a.attrelid
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE c.relkind IN ('r','p')
        AND a.attnum > 0
        AND NOT a.attisdropped
        AND n.nspname NOT IN ('pg_catalog','information_schema')
        AND n.nspname NOT LIKE 'pg_toast%'
      ORDER BY n.nspname, c.relname, a.attname
    ) t
  )
);
ROLLBACK;
"""
        output = self._run(sql)
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"PostgreSQL benchmark catalog snapshot was not JSON: {output[:200]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("PostgreSQL benchmark catalog snapshot was not an object")
        return parsed

    def explain(self, query: str, settings: dict[str, str], *, analyze: bool) -> list[dict]:
        query = query.strip().rstrip(";")
        if not query:
            raise ValueError("query must not be empty")

        set_sql = "\n".join(render_set_local(k, v) for k, v in sorted(settings.items()))
        options = ["FORMAT JSON", "SETTINGS", "SUMMARY"]
        if analyze:
            options.extend(["ANALYZE", "BUFFERS", "TIMING OFF"])
        explain = f"EXPLAIN ({', '.join(options)}) {query};"
        script = "\n".join(
            part for part in [
                "BEGIN TRANSACTION READ ONLY;",
                set_sql,
                explain,
                "ROLLBACK;",
            ] if part
        )
        output = self._run(script)
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"psql returned non-JSON EXPLAIN output: {output[:200]!r}") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("PostgreSQL EXPLAIN JSON was not an array")
        return parsed


def read_query(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
