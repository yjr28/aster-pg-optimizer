from __future__ import annotations

from dataclasses import dataclass

from aster.data.records import PlanObservation, Provenance
from aster.integration.psql import ExplainRunner
from aster.plans import parse_explain_json, plan_fingerprint

from .specs import CandidateSpec


@dataclass(frozen=True)
class DiscoveredCandidate:
    spec: CandidateSpec
    fingerprint: str
    plan_json: list[dict]


class CandidateCollector:
    def __init__(self, runner: ExplainRunner):
        self.runner = runner

    def discover(
        self,
        query: str,
        candidates: tuple[CandidateSpec, ...] | list[CandidateSpec],
    ) -> tuple[DiscoveredCandidate, ...]:
        unique: dict[str, DiscoveredCandidate] = {}
        for spec in candidates:
            raw = self.runner.explain(query, spec.settings, analyze=False)
            plan = parse_explain_json(raw)
            fingerprint = plan_fingerprint(plan)
            # Preserve first occurrence so native wins ties/duplicates when ordered first.
            unique.setdefault(
                fingerprint,
                DiscoveredCandidate(spec=spec, fingerprint=fingerprint, plan_json=raw),
            )
        return tuple(unique.values())

    def measure(
        self,
        query: str,
        candidate: DiscoveredCandidate,
        *,
        workload: str,
        query_id: str,
        dataset_version: str,
        run_seed: int,
        code_revision: str,
        warmups: int = 1,
        repetitions: int = 3,
    ) -> tuple[PlanObservation, ...]:
        if warmups < 0 or repetitions < 1:
            raise ValueError("warmups must be >=0 and repetitions >=1")
        for _ in range(warmups):
            self.runner.explain(query, candidate.spec.settings, analyze=True)

        version = self.runner.postgres_version()
        observations: list[PlanObservation] = []
        for repetition in range(repetitions):
            raw = self.runner.explain(query, candidate.spec.settings, analyze=True)
            plan = parse_explain_json(raw)
            if plan.execution_time_ms is None:
                raise RuntimeError("EXPLAIN ANALYZE did not report Execution Time")
            measured_fp = plan_fingerprint(plan)
            if measured_fp != candidate.fingerprint:
                raise RuntimeError(
                    "candidate plan changed between discovery and measurement; "
                    f"expected {candidate.fingerprint[:12]}, got {measured_fp[:12]}"
                )
            provenance = Provenance.now(
                workload=workload,
                query_id=query_id,
                candidate_id=candidate.spec.candidate_id,
                postgres_version=version,
                dataset_version=dataset_version,
                run_seed=run_seed,
                code_revision=code_revision,
            )
            observations.append(
                PlanObservation(
                    provenance=provenance,
                    plan_fingerprint=measured_fp,
                    planner_settings=dict(candidate.spec.settings),
                    planning_time_ms=plan.planning_time_ms,
                    execution_time_ms=plan.execution_time_ms,
                    repetition=repetition,
                    plan_json=raw[0],
                )
            )
        return tuple(observations)
