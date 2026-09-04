from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Provenance:
    workload: str
    query_id: str
    candidate_id: str
    postgres_version: str
    dataset_version: str
    run_seed: int
    code_revision: str
    captured_at_utc: str

    @classmethod
    def now(
        cls,
        *,
        workload: str,
        query_id: str,
        candidate_id: str,
        postgres_version: str,
        dataset_version: str,
        run_seed: int,
        code_revision: str,
    ) -> "Provenance":
        return cls(
            workload=workload,
            query_id=query_id,
            candidate_id=candidate_id,
            postgres_version=postgres_version,
            dataset_version=dataset_version,
            run_seed=run_seed,
            code_revision=code_revision,
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class PlanObservation:
    provenance: Provenance
    plan_fingerprint: str
    planner_settings: dict[str, str]
    planning_time_ms: float | None
    execution_time_ms: float
    repetition: int
    plan_json: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)
