from __future__ import annotations

from dataclasses import dataclass, field

from aster.planner import validate_planner_setting


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    settings: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        for name, value in self.settings.items():
            validate_planner_setting(name, value)


def default_candidates() -> tuple[CandidateSpec, ...]:
    """Fast bounded search over PostgreSQL physical-planner switches."""
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


def research_candidates() -> tuple[CandidateSpec, ...]:
    """Candidate set for join-order research workloads such as JOB.

    In addition to physical join-method interventions, force GEQO at two or more FROM
    items and vary its documented random seed. PostgreSQL's GEQO seed changes which
    join paths are explored. CandidateCollector still fingerprints/deduplicates the
    resulting physical plans before any EXPLAIN ANALYZE executions are measured.
    """
    candidates = list(default_candidates())
    for seed in ("0.00", "0.20", "0.40", "0.60", "0.80", "1.00"):
        candidates.append(CandidateSpec(
            f"geqo_seed_{seed.replace('.', '')}",
            {"geqo": "on", "geqo_threshold": "2", "geqo_seed": seed},
        ))
    candidates.extend((
        CandidateSpec(
            "geqo_low_effort",
            {"geqo": "on", "geqo_threshold": "2", "geqo_seed": "0.50", "geqo_effort": "1"},
        ),
        CandidateSpec(
            "geqo_high_effort",
            {"geqo": "on", "geqo_threshold": "2", "geqo_seed": "0.50", "geqo_effort": "10"},
        ),
    ))
    return tuple(candidates)
