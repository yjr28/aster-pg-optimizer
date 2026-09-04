import pytest

from aster.benchmarks import (
    DistributionSummary,
    PairedBenchmarkResult,
    PairedExecutionSample,
    native_control_from_paired,
)


def _dist(value):
    return DistributionSummary(value,value,value,value,value)


def test_native_fallback_reuses_existing_control_samples_and_charges_overhead():
    samples=(
        PairedExecutionSample(0,("native","aster"),10.0,5.0,1.0,1.0),
        PairedExecutionSample(1,("aster","native"),12.0,6.0,1.0,1.0),
    )
    paired=PairedBenchmarkResult(
        postgres_version="17",native_candidate_id="native",selected_candidate_id="alt",
        same_physical_plan=False,selection_overhead_ms=2.0,
        native_execution=_dist(11.0),aster_execution=_dist(5.5),
        native_database_latency=_dist(12.0),aster_database_latency=_dist(6.5),
        execution_speedup_geomean=2.0,end_to_end_speedup_geomean=1.5,
        improved_fraction=1.0,regressed_fraction=0.0,worst_regression_ratio=0.5,samples=samples,
    )
    fallback=native_control_from_paired(paired,selection_overhead_ms=2.0)
    assert fallback.selected_candidate_id == "native"
    assert fallback.same_physical_plan is True
    assert fallback.execution_speedup_geomean == 1.0
    assert fallback.end_to_end_speedup_geomean < 1.0
    assert [s.aster_execution_ms for s in fallback.samples] == [10.0,12.0]
    assert [s.native_execution_ms for s in fallback.samples] == [10.0,12.0]
    assert all(s.execution_order == ("native_shared_control",) for s in fallback.samples)
    assert fallback.worst_regression_ratio == pytest.approx(1.0)
