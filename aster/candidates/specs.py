from __future__ import annotations

from dataclasses import dataclass, field


_ALLOWED_PLANNER_GUCS = {
    "enable_hashjoin",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_seqscan",
    "enable_indexscan",
    "enable_indexonlyscan",
    "enable_bitmapscan",
    "enable_material",
    "enable_memoize",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    settings: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.settings) - _ALLOWED_PLANNER_GUCS
        if unknown:
            raise ValueError(f"unsupported planner GUC(s): {sorted(unknown)}")
        invalid = {k: v for k, v in self.settings.items() if v not in {"on", "off"}}
        if invalid:
            raise ValueError(f"planner settings must be on/off: {invalid}")


def default_candidates() -> tuple[CandidateSpec, ...]:
    """Small, bounded first-stage search over PostgreSQL physical-planner switches.

    These settings do not guarantee a unique physical plan. CandidateCollector
    canonicalizes plans before any measured execution and drops duplicates.
    """
    return (
        CandidateSpec("native", {}),
        CandidateSpec("no_hashjoin", {"enable_hashjoin": "off"}),
        CandidateSpec("no_mergejoin", {"enable_mergejoin": "off"}),
        CandidateSpec("no_nestloop", {"enable_nestloop": "off"}),
        CandidateSpec("no_seqscan", {"enable_seqscan": "off"}),
        CandidateSpec(
            "hash_join_preferred",
            {"enable_mergejoin": "off", "enable_nestloop": "off"},
        ),
        CandidateSpec(
            "merge_join_preferred",
            {"enable_hashjoin": "off", "enable_nestloop": "off"},
        ),
        CandidateSpec(
            "nested_loop_preferred",
            {"enable_hashjoin": "off", "enable_mergejoin": "off"},
        ),
    )
