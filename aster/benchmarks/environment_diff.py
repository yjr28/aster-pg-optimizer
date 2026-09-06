from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .environment import _canonical_sha256


ENVIRONMENT_DIFF_SECTIONS = frozenset({
    "host",
    "postgres_metadata",
    "settings",
    "relations",
    "indexes",
    "statistics_state",
    "statistics_targets",
})

_MODELED_HOST_KEYS = frozenset({
    "system",
    "release",
    "machine",
    "platform",
    "cpu_count",
    "cpu_model",
    "memory_total_bytes",
    "python_version",
})

_HOST_TEXT_FIELDS = (
    "system",
    "release",
    "machine",
    "platform",
    "cpu_model",
    "python_version",
)

_HOST_OPTIONAL_INT_FIELDS = (
    "cpu_count",
    "memory_total_bytes",
)

_MODELED_POSTGRES_KEYS = frozenset({
    "server_version",
    "server_version_num",
    "database",
    "database_size_bytes",
    "settings",
    "relations",
    "indexes",
    "statistics_state",
    "statistics_targets",
})

_POSTGRES_TEXT_FIELDS = (
    "server_version",
    "server_version_num",
    "database",
)

_POSTGRES_INT_FIELDS = (
    "database_size_bytes",
)


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
    unclassified_host_changes: dict[str, dict[str, Any]]
    unclassified_postgres_changes: dict[str, dict[str, Any]]

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


@dataclass(frozen=True)
class PerturbationValidation:
    valid: bool
    allowed_sections: tuple[str, ...]
    required_sections: tuple[str, ...]
    observed_sections: tuple[str, ...]
    unexpected_sections: tuple[str, ...]
    missing_required_sections: tuple[str, ...]
    unexplained_fingerprint_change: bool
    fingerprint_evidence_mismatch: bool

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def validate_perturbation(
    diff: BenchmarkEnvironmentDiff,
    *,
    allowed_sections: Iterable[str],
    required_sections: Iterable[str] = (),
) -> PerturbationValidation:
    """Validate that an environment change matches a declared experiment policy.

    This is intentionally strict. A statistics-only experiment should not silently
    include an index/config/host change. Callers can explicitly widen the allowed set
    when a perturbation legitimately changes more than one section. A changed
    environment fingerprint with no modeled semantic change, or a change in an
    unmodeled host/PostgreSQL snapshot field, is rejected as unexplained evidence
    drift. Modeled semantic changes paired with an identical environment fingerprint
    are also rejected because the semantic evidence contradicts the recorded identity.
    """
    allowed = frozenset(allowed_sections)
    required = frozenset(required_sections)
    unknown = (allowed | required) - ENVIRONMENT_DIFF_SECTIONS
    if unknown:
        raise ValueError(f"unknown environment diff sections: {sorted(unknown)}")
    if not required <= allowed:
        raise ValueError("required_sections must be a subset of allowed_sections")
    observed = frozenset(diff.changed_sections)
    unexpected = tuple(sorted(observed - allowed))
    missing = tuple(sorted(required - observed))
    unexplained_fingerprint_change = (
        bool(diff.unclassified_host_changes)
        or bool(diff.unclassified_postgres_changes)
        or (not diff.identical_fingerprint and not observed)
    )
    fingerprint_evidence_mismatch = diff.identical_fingerprint and bool(observed)
    return PerturbationValidation(
        valid=(
            not unexpected
            and not missing
            and not unexplained_fingerprint_change
            and not fingerprint_evidence_mismatch
        ),
        allowed_sections=tuple(sorted(allowed)),
        required_sections=tuple(sorted(required)),
        observed_sections=tuple(sorted(observed)),
        unexpected_sections=unexpected,
        missing_required_sections=missing,
        unexplained_fingerprint_change=unexplained_fingerprint_change,
        fingerprint_evidence_mismatch=fingerprint_evidence_mismatch,
    )


def _require_sha256(value: Any, field: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value.lower()
    ):
        raise ValueError(f"{label} {field} must be a SHA-256 hex string")
    return value.lower()


def _require_environment(payload: dict[str, Any], label: str) -> None:
    required = {
        "host",
        "postgres",
        "host_sha256",
        "postgres_sha256",
        "environment_sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{label} environment missing fields: {missing}")
    host = payload.get("host")
    postgres = payload.get("postgres")
    if not isinstance(host, dict) or not isinstance(postgres, dict):
        raise ValueError(f"{label} host/postgres environment sections must be objects")

    missing_host = sorted(_MODELED_HOST_KEYS - host.keys())
    if missing_host:
        raise ValueError(f"{label} host snapshot missing modeled fields: {missing_host}")
    for field in _HOST_TEXT_FIELDS:
        if not isinstance(host[field], str):
            raise ValueError(f"{label} host field {field} has invalid type")
    for field in _HOST_OPTIONAL_INT_FIELDS:
        value = host[field]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{label} host field {field} has invalid type")

    missing_postgres = sorted(_MODELED_POSTGRES_KEYS - postgres.keys())
    if missing_postgres:
        raise ValueError(
            f"{label} PostgreSQL snapshot missing modeled fields: {missing_postgres}"
        )
    for field in _POSTGRES_TEXT_FIELDS:
        if not isinstance(postgres[field], str) or postgres[field] == "":
            raise ValueError(f"{label} PostgreSQL field {field} has invalid type or value")
    for field in _POSTGRES_INT_FIELDS:
        value = postgres[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} PostgreSQL field {field} has invalid type or value")
    if not isinstance(postgres["settings"], dict):
        raise ValueError(f"{label} PostgreSQL settings section must be an object")
    for section in ("relations", "indexes", "statistics_state", "statistics_targets"):
        if not isinstance(postgres[section], list):
            raise ValueError(f"{label} PostgreSQL {section} section must be a list")

    host_sha = _require_sha256(payload.get("host_sha256"), "host_sha256", label)
    postgres_sha = _require_sha256(payload.get("postgres_sha256"), "postgres_sha256", label)
    environment_sha = _require_sha256(
        payload.get("environment_sha256"), "environment_sha256", label
    )

    expected_host_sha = _canonical_sha256(host)
    expected_postgres_sha = _canonical_sha256(postgres)
    expected_environment_sha = _canonical_sha256({
        "schema_version": 1,
        "host_sha256": expected_host_sha,
        "postgres_sha256": expected_postgres_sha,
    })
    if host_sha != expected_host_sha:
        raise ValueError(f"{label} host_sha256 does not match host payload")
    if postgres_sha != expected_postgres_sha:
        raise ValueError(f"{label} postgres_sha256 does not match postgres payload")
    if environment_sha != expected_environment_sha:
        raise ValueError(
            f"{label} environment_sha256 does not match component fingerprints"
        )


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
        if any(not isinstance(value, str) or value == "" for value in key):
            raise ValueError(f"{label} row has invalid identity fields {key_fields}: {row}")
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
    perturbation. Known fields that feed the environment fingerprint are diffed into
    interpretable categories; changed unknown host/PostgreSQL snapshot fields are
    retained separately so perturbation validation can fail closed. Incomplete modeled
    snapshot sections are rejected rather than treated as empty evidence.
    """
    _require_environment(before, "before")
    _require_environment(after, "after")
    before_host = before["host"]
    after_host = after["host"]
    before_pg = before["postgres"]
    after_pg = after["postgres"]

    settings_before = before_pg["settings"]
    settings_after = after_pg["settings"]

    metadata_keys = (
        "server_version",
        "server_version_num",
        "database",
        "database_size_bytes",
    )
    unclassified_host_keys = (set(before_host) | set(after_host)) - _MODELED_HOST_KEYS
    unclassified_postgres_keys = (set(before_pg) | set(after_pg)) - _MODELED_POSTGRES_KEYS
    return BenchmarkEnvironmentDiff(
        before_environment_sha256=before["environment_sha256"],
        after_environment_sha256=after["environment_sha256"],
        identical_fingerprint=before["environment_sha256"] == after["environment_sha256"],
        host_changes=_scalar_changes(
            before_host,
            after_host,
            keys=_MODELED_HOST_KEYS,
        ),
        postgres_metadata_changes=_scalar_changes(before_pg, after_pg, keys=metadata_keys),
        settings_changes=_scalar_changes(settings_before, settings_after),
        relation_changes=_diff_keyed_rows(
            before_pg["relations"], after_pg["relations"],
            key_fields=("schema_name", "relation_name"), label="relations",
        ),
        index_changes=_diff_keyed_rows(
            before_pg["indexes"], after_pg["indexes"],
            key_fields=("schema_name", "table_name", "index_name"), label="indexes",
        ),
        statistics_state_changes=_diff_keyed_rows(
            before_pg["statistics_state"], after_pg["statistics_state"],
            key_fields=("schema_name", "relation_name"), label="statistics_state",
        ),
        statistics_target_changes=_diff_keyed_rows(
            before_pg["statistics_targets"], after_pg["statistics_targets"],
            key_fields=("schema_name", "relation_name", "column_name"), label="statistics_targets",
        ),
        unclassified_host_changes=_scalar_changes(
            before_host,
            after_host,
            keys=unclassified_host_keys,
        ),
        unclassified_postgres_changes=_scalar_changes(
            before_pg,
            after_pg,
            keys=unclassified_postgres_keys,
        ),
    )
