from __future__ import annotations

import math

_BOOLEAN_SETTINGS = {
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_seqscan",
    "enable_indexscan",
    "enable_indexonlyscan",
    "enable_bitmapscan",
    "enable_material",
    "enable_memoize",
    "geqo",
}

_INTEGER_RANGES = {
    "geqo_threshold": (2, 100),
    "geqo_effort": (1, 10),
    "geqo_pool_size": (0, 10_000),
    "geqo_generations": (0, 10_000),
}

_FLOAT_RANGES = {
    "geqo_seed": (0.0, 1.0),
    "geqo_selection_bias": (1.5, 2.0),
}


def validate_planner_setting(name: str, value: str) -> None:
    if name in _BOOLEAN_SETTINGS:
        if value not in {"on", "off"}:
            raise ValueError(f"{name} must be on/off, got {value!r}")
        return
    if name in _INTEGER_RANGES:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        low, high = _INTEGER_RANGES[name]
        if not low <= parsed <= high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {parsed}")
        if name in {"geqo_pool_size", "geqo_generations"} and parsed == 1:
            raise ValueError(f"{name}=1 is not a supported research setting")
        return
    if name in _FLOAT_RANGES:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, got {value!r}") from exc
        low, high = _FLOAT_RANGES[name]
        if not math.isfinite(parsed) or not low <= parsed <= high:
            raise ValueError(f"{name} must be finite and in [{low}, {high}], got {value!r}")
        return
    raise ValueError(f"unsupported planner GUC: {name}")


def render_set_local(name: str, value: str) -> str:
    """Render a SET LOCAL statement only after strict allowlist/range validation."""
    validate_planner_setting(name, value)
    return f"SET LOCAL {name} = {value};"
