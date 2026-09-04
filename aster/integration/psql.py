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
