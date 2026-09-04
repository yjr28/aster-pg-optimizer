from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class BenchmarkCatalogRunner(Protocol):
    def benchmark_catalog_snapshot(self) -> dict: ...


@dataclass(frozen=True)
class HostEnvironment:
    system: str
    release: str
    machine: str
    platform: str
    cpu_count: int | None
    cpu_model: str
    memory_total_bytes: int | None
    python_version: str


@dataclass(frozen=True)
class BenchmarkEnvironment:
    captured_at_utc: str
    host: HostEnvironment
    postgres: dict
    host_sha256: str
    postgres_sha256: str
    environment_sha256: str

    def to_jsonable(self) -> dict:
        return asdict(self)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _read_memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
        except (OSError, ValueError):
            pass
    return None


def capture_host_environment() -> HostEnvironment:
    return HostEnvironment(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        cpu_model=_read_cpu_model(),
        memory_total_bytes=_read_memory_total_bytes(),
        python_version=sys.version.split()[0],
    )


def capture_benchmark_environment(runner: BenchmarkCatalogRunner) -> BenchmarkEnvironment:
    host = capture_host_environment()
    postgres = runner.benchmark_catalog_snapshot()
    if not isinstance(postgres, dict):
        raise TypeError("benchmark_catalog_snapshot() must return a dictionary")
    host_payload = asdict(host)
    host_sha = _canonical_sha256(host_payload)
    postgres_sha = _canonical_sha256(postgres)
    combined_sha = _canonical_sha256(
        {
            "schema_version": 1,
            "host_sha256": host_sha,
            "postgres_sha256": postgres_sha,
        }
    )
    return BenchmarkEnvironment(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        host=host,
        postgres=postgres,
        host_sha256=host_sha,
        postgres_sha256=postgres_sha,
        environment_sha256=combined_sha,
    )
