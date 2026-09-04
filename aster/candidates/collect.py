from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from aster.data.records import PlanObservation, Provenance
from aster.integration.psql import ExplainRunner
from aster.plans import parse_explain_json, plan_fingerprint

from .specs import CandidateSpec


@dataclass(frozen=True)
class DiscoveredCandidate:
    spec: CandidateSpec
    fingerprint: str
    plan_json: list[dict]


@dataclass(frozen=True)
class CandidatePlanGroup:
    fingerprint: str
    representative_candidate_id: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDiscoveryReport:
    attempted_interventions: int
    unique_plan_count: int
    duplicate_interventions: int
    unique_candidates: tuple[DiscoveredCandidate, ...]
    plan_groups: tuple[CandidatePlanGroup, ...]

    @property
    def uniqueness_ratio(self) -> float:
        return self.unique_plan_count / self.attempted_interventions if self.attempted_interventions else 0.0


class CandidateCollector:
    def __init__(self, runner: ExplainRunner):
        self.runner = runner

    def discover_report(
        self,
        query: str,
        candidates: tuple[CandidateSpec, ...] | list[CandidateSpec],
    ) -> CandidateDiscoveryReport:
        specs = tuple(candidates)
        unique: dict[str, DiscoveredCandidate] = {}
        grouped_ids: dict[str, list[str]] = {}
        for spec in specs:
            raw = self.runner.explain(query, spec.settings, analyze=False)
            plan = parse_explain_json(raw)
            fingerprint = plan_fingerprint(plan)
            grouped_ids.setdefault(fingerprint, []).append(spec.candidate_id)
            unique.setdefault(
                fingerprint,
                DiscoveredCandidate(spec=spec, fingerprint=fingerprint, plan_json=raw),
            )
        unique_candidates = tuple(unique.values())
        groups = tuple(
            CandidatePlanGroup(
                fingerprint=fingerprint,
                representative_candidate_id=unique[fingerprint].spec.candidate_id,
                candidate_ids=tuple(grouped_ids[fingerprint]),
            )
            for fingerprint in unique
        )
        return CandidateDiscoveryReport(
            attempted_interventions=len(specs),
            unique_plan_count=len(unique_candidates),
            duplicate_interventions=len(specs) - len(unique_candidates),
            unique_candidates=unique_candidates,
            plan_groups=groups,
        )

    def discover(self, query: str, candidates: tuple[CandidateSpec, ...] | list[CandidateSpec]) -> tuple[DiscoveredCandidate, ...]:
        return self.discover_report(query, candidates).unique_candidates

    def measure(self, query: str, candidate: DiscoveredCandidate, *, workload: str, query_id: str,
                dataset_version: str, run_seed: int, code_revision: str,
                experiment_id: str | None = None, query_template: str | None = None,
                parameter_key: str | None = None, environment_sha256: str | None = None,
                warmups: int = 1, repetitions: int = 3) -> tuple[PlanObservation, ...]:
        if warmups < 0 or repetitions < 1:
            raise ValueError("warmups must be >=0 and repetitions >=1")
        if environment_sha256 is not None and (
            len(environment_sha256) != 64
            or any(char not in "0123456789abcdef" for char in environment_sha256.lower())
        ):
            raise ValueError("environment_sha256 must be a SHA-256 hex string when provided")
        for _ in range(warmups):
            self.runner.explain(query, candidate.spec.settings, analyze=True)
        version = self.runner.postgres_version()
        resolved_experiment_id = experiment_id or str(uuid4())
        observations: list[PlanObservation] = []
        for repetition in range(repetitions):
            raw = self.runner.explain(query, candidate.spec.settings, analyze=True)
            plan = parse_explain_json(raw)
            if plan.execution_time_ms is None:
                raise RuntimeError("EXPLAIN ANALYZE did not report Execution Time")
            measured_fp = plan_fingerprint(plan)
            if measured_fp != candidate.fingerprint:
                raise RuntimeError("candidate plan changed between discovery and measurement; "
                                   f"expected {candidate.fingerprint[:12]}, got {measured_fp[:12]}")
            provenance = Provenance.now(
                experiment_id=resolved_experiment_id, workload=workload, query_id=query_id,
                candidate_id=candidate.spec.candidate_id, postgres_version=version,
                dataset_version=dataset_version, run_seed=run_seed, code_revision=code_revision,
                query_template=query_template, parameter_key=parameter_key,
                environment_sha256=environment_sha256,
            )
            observations.append(PlanObservation(
                provenance=provenance, plan_fingerprint=measured_fp,
                planner_settings=dict(candidate.spec.settings), planning_time_ms=plan.planning_time_ms,
                execution_time_ms=plan.execution_time_ms, repetition=repetition, plan_json=raw[0],
            ))
        return tuple(observations)
