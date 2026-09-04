from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from aster.models import TrainingExample

from .training import TrainingProtocol, run_training_experiment


@dataclass(frozen=True)
class RobustnessRegimeResult:
    regime: str
    status: str
    reason: str | None
    metrics: dict | None


@dataclass(frozen=True)
class RobustnessMatrix:
    evaluation_kind: str
    observations: int
    query_groups: int
    templates: int
    workloads: tuple[str, ...]
    dataset_versions: tuple[str, ...]
    protocol: dict
    regimes: tuple[RobustnessRegimeResult, ...]

    def to_jsonable(self) -> dict:
        return asdict(self)


def _profile(examples: list[TrainingExample]) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    queries={
        (example.dataset_version or "", example.workload or "", example.query_id)
        for example in examples
    }
    templates={example.query_template for example in examples if example.query_template}
    workloads=tuple(sorted({str(example.workload) for example in examples if example.workload}))
    versions=tuple(sorted({str(example.dataset_version) for example in examples if example.dataset_version}))
    return len(queries), len(templates), workloads, versions


def run_robustness_matrix(
    examples: list[TrainingExample],
    protocol: TrainingProtocol,
    *,
    regimes: tuple[str, ...] = ("template", "parameter", "workload"),
) -> RobustnessMatrix:
    """Evaluate leakage-resistant split regimes on already measured plan runtimes.

    This is an offline measured-corpus replay, not a live database benchmark. Each test
    query selects among candidate plans whose labels came from real EXPLAIN ANALYZE
    collection. Live paired execution remains a separate publication gate.
    """
    if not examples:
        raise ValueError("dataset is empty")
    unknown=set(regimes)-{"template","parameter","workload"}
    if unknown:
        raise ValueError(f"unsupported robustness regimes: {sorted(unknown)}")

    results: list[RobustnessRegimeResult]=[]
    for regime in regimes:
        regime_protocol=replace(protocol,split_regime=regime)
        try:
            _model, experiment=run_training_experiment(examples,regime_protocol)
        except ValueError as exc:
            results.append(RobustnessRegimeResult(
                regime=regime,
                status="unsupported_for_dataset",
                reason=str(exc),
                metrics=None,
            ))
            continue
        results.append(RobustnessRegimeResult(
            regime=regime,
            status="ok",
            reason=None,
            metrics=experiment.to_jsonable(),
        ))

    query_groups, templates, workloads, versions=_profile(examples)
    return RobustnessMatrix(
        evaluation_kind="offline_measured_plan_replay",
        observations=len(examples),
        query_groups=query_groups,
        templates=templates,
        workloads=workloads,
        dataset_versions=versions,
        protocol=asdict(protocol),
        regimes=tuple(results),
    )
