from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class KeyedRowsDiff:
    added: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]
    changed: tuple[dict[str, Any], ...]

    @property
    def changed_count(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)


@dataclass(frozen=True)
class BenchmarkEnvironmentDiff:
    before_environment_sha256: str
    after_environment_sha256: str
    identical_fingerprint: bool
    host_changes: dict[str, dict[str, Any]]
    postgres_metadata_changes: dict[str, dict[str, Any]]
    settings_changes: dict[str, dict[str, Any]]
    relation_changes: KeyedRowsDiff
    index_changes: KeyedRowsDiff
    statistics_state_changes: KeyedRowsDiff
    statistics_target_changes: KeyedRowsDiff

    @property
    def changed_sections(self) -> tuple[str, ...]:
        sections: list[str] = []
        if self.host_changes:
            sections.append("host")
        if self.postgres_metadata_changes:
            sections.append("postgres_metadata")
        if self.settings_changes:
            sections.append("settings")
        if self.relation_changes.changed_count:
            sections.append("relations")
        if self.index_changes.changed_count:
            sections.append("indexes")
        if self.statistics_state_changes.changed_count:
            sections.append("statistics_state")
        if self.statistics_target_changes.changed_count:
            sections.append("statistics_targets")
        return tuple(sections)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_sections"] = list(self.changed_sections)
        return payload


def _require_environment(payload: dict[str, Any], label: str) -> None:
    required = {"host", "postgres", "environment_sha256"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{label} environment missing fields: {missing}")
    sha = payload.get("environment_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError(f"{label} environment_sha256 must be a SHA-256 hex string")
    if not isinstance(payload.get("host"), dict) or not isinstance(payload.get("postgres"), dict):
        raise ValueError(f"{label} host/postgres environment sections must be objects")


def _scalar_changes(before: dict[str, Any], after: dict[str, Any], *, keys: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    selected = sorted(set(keys) if keys is not None else set(before) | set(after))
    changes: dict[str, dict[str, Any]] = {}
    for key in selected:
        left = before.get(key)
        right = after.get(key)
        if left != right:
            changes[key] = {"before": left, "after": right}
    return changes


def _keyed_rows(
    rows: Any,
    *,
    key_fields: tuple[str, ...],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if rows is None:
        return {}
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} rows must be objects")
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None for value in key):
            raise ValueError(f"{label} row missing identity fields {key_fields}: {row}")
        if key in keyed:
            raise ValueError(f"duplicate {label} identity {key}")
        keyed[key] = row
    return keyed


def _diff_keyed_rows(
    before_rows: Any,
    after_rows: Any,
    *,
    key_fields: tuple[str, ...],
    label: str,
) -> KeyedRowsDiff:
    before = _keyed_rows(before_rows, key_fields=key_fields, label=f"before {label}")
    after = _keyed_rows(after_rows, key_fields=key_fields, label=f"after {label}")
    added = tuple(after[key] for key in sorted(after.keys() - before.keys()))
    removed = tuple(before[key] for key in sorted(before.keys() - after.keys()))
    changed: list[dict[str, Any]] = []
    for key in sorted(before.keys() & after.keys()):
        if before[key] != after[key]:
            changed.append({
                "identity": dict(zip(key_fields, key)),
                "before": before[key],
                "after": after[key],
            })
    return KeyedRowsDiff(added=added, removed=removed, changed=tuple(changed))


def compare_benchmark_environments(
    before: dict[str, Any],
    after: dict[str, Any],
) -> BenchmarkEnvironmentDiff:
    """Compare two captured benchmark environments by research-relevant semantics.

    `captured_at_utc` is intentionally ignored. A different timestamp does not make a
    perturbation. All fields that feed the environment fingerprint are represented by
    host/postgres sections and are diffed into interpretable categories.
    """
    _require_environment(before, "before")
    _require_environment(after, "after")
    before_pg = before["postgres"]
    after_pg = after["postgres"]

    settings_before = before_pg.get("settings") or {}
    settings_after = after_pg.get("settings") or {}
    if not isinstance(settings_before, dict) or not isinstance(settings_after, dict):
        raise ValueError("PostgreSQL settings sections must be objects")

    metadata_keys = (
        "server_version",
        "server_version_num",
        "database",
        "database_size_bytes",
    )
    return BenchmarkEnvironmentDiff(
        before_environment_sha256=before["environment_sha256"],
        after_environment_sha256=after["environment_sha256"],
        identical_fingerprint=before["environment_sha256"] == after["environment_sha256"],
        host_changes=_scalar_changes(before["host"], after["host"]),
        postgres_metadata_changes=_scalar_changes(before_pg, after_pg, keys=metadata_keys),
        settings_changes=_scalar_changes(settings_before, settings_after),
        relation_changes=_diff_keyed_rows(
            before_pg.get("relations"), after_pg.get("relations"),
            key_fields=("schema_name", "relation_name"), label="relations",
        ),
        index_changes=_diff_keyed_rows(
            before_pg.get("indexes"), after_pg.get("indexes"),
            key_fields=("schema_name", "table_name", "index_name"), label="indexes",
        ),
        statistics_state_changes=_diff_keyed_rows(
            before_pg.get("statistics_state"), after_pg.get("statistics_state"),
            key_fields=("schema_name", "relation_name"), label="statistics_state",
        ),
        statistics_target_changes=_diff_keyed_rows(
            before_pg.get("statistics_targets"), after_pg.get("statistics_targets"),
            key_fields=("schema_name", "relation_name", "column_name"), label="statistics_targets",
        ),
    )
