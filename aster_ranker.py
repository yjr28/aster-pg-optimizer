"""Transparent query-plan ranking prototype for Aster.

This module deliberately starts with an auditable deterministic scoring model.
It is a real executable baseline for the project, not a claim that a trained
ML ranker or production benchmark already exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Iterable


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    postgres_cost: float
    estimated_rows: int
    seq_scans: int = 0
    nested_loops: int = 0
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("plan name is required")
        if self.postgres_cost < 0 or self.estimated_rows < 0:
            raise ValueError("cost and estimated_rows must be non-negative")
        if self.seq_scans < 0 or self.nested_loops < 0:
            raise ValueError("operator counts must be non-negative")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class RankedPlan:
    plan: CandidatePlan
    score: float


@dataclass(frozen=True)
class Decision:
    selected_plan: str
    model_choice: str
    postgres_choice: str
    fallback: bool
    reason: str
    margin: float
    ranking: tuple[RankedPlan, ...]


def score_plan(plan: CandidatePlan) -> float:
    """Lower is better.

    The weights are intentionally explicit so this baseline can be challenged,
    benchmarked, and eventually replaced by a learned ranker.
    """
    return (
        0.46 * log1p(plan.postgres_cost)
        + 0.14 * log1p(plan.estimated_rows)
        + 0.13 * plan.seq_scans
        + 0.11 * plan.nested_loops
        + 0.16 * plan.uncertainty
    )


def rank_plans(
    candidates: Iterable[CandidatePlan],
    *,
    postgres_choice: str,
    min_margin: float = 0.08,
    max_uncertainty: float = 0.35,
) -> Decision:
    plans = tuple(candidates)
    if len(plans) < 2:
        raise ValueError("at least two candidate plans are required")
    if postgres_choice not in {p.name for p in plans}:
        raise ValueError("postgres_choice must name one candidate")
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")
    if not 0 <= max_uncertainty <= 1:
        raise ValueError("max_uncertainty must be in [0, 1]")

    ranking = tuple(sorted((RankedPlan(p, score_plan(p)) for p in plans), key=lambda x: x.score))
    best, second = ranking[0], ranking[1]
    margin = second.score - best.score

    if best.plan.uncertainty > max_uncertainty:
        return Decision(
            selected_plan=postgres_choice,
            model_choice=best.plan.name,
            postgres_choice=postgres_choice,
            fallback=True,
            reason="best candidate exceeds uncertainty threshold",
            margin=margin,
            ranking=ranking,
        )

    if margin < min_margin:
        return Decision(
            selected_plan=postgres_choice,
            model_choice=best.plan.name,
            postgres_choice=postgres_choice,
            fallback=True,
            reason="ranking margin is too small",
            margin=margin,
            ranking=ranking,
        )

    return Decision(
        selected_plan=best.plan.name,
        model_choice=best.plan.name,
        postgres_choice=postgres_choice,
        fallback=False,
        reason="ranker confidence gate passed",
        margin=margin,
        ranking=ranking,
    )
