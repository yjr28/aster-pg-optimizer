import pytest

from aster.benchmarks import (
    DistributionSummary,
    PairedBenchmarkResult,
    WorkloadQueryBenchmark,
    summarize_workload_benchmark,
)


def _dist(value):
    return DistributionSummary(value, value, value, value, value)


def _paired(native, selected, overhead=1.0):
    return PairedBenchmarkResult(
        postgres_version="17",
        native_candidate_id="native",
        selected_candidate_id="selected",
        same_physical_plan=native == selected,
        selection_overhead_ms=overhead,
        native_execution=_dist(native),
        aster_execution=_dist(selected),
        native_database_latency=_dist(native),
        aster_database_latency=_dist(selected),
        execution_speedup_geomean=native/selected,
        end_to_end_speedup_geomean=native/(selected+overhead),
        improved_fraction=float(selected<native),
        regressed_fraction=float(selected>native),
        worst_regression_ratio=selected/native,
        samples=(),
    )


def test_workload_summary_compares_no_fallback_and_fallback_distributions():
    queries=[
        WorkloadQueryBenchmark("q1",_paired(10,5),_paired(10,5),False,"",1.0),
        WorkloadQueryBenchmark("q2",_paired(20,40),_paired(20,20),True,"uncertain",2.0),
    ]
    summary=summarize_workload_benchmark(queries)
    assert summary.query_count == 2
    assert summary.no_fallback.geometric_mean_speedup_vs_native == pytest.approx(1.0)
    assert summary.no_fallback.improved_fraction == 0.5
    assert summary.no_fallback.regressed_fraction == 0.5
    assert summary.no_fallback.worst_regression_ratio == pytest.approx(2.0)
    assert summary.fallback.geometric_mean_speedup_vs_native == pytest.approx(2**0.5)
    assert summary.fallback.regressed_fraction == 0.0
    assert summary.fallback_fraction == 0.5
    assert summary.selection_overhead_ms.median_ms == pytest.approx(1.5)
    assert summary.fallback.end_to_end_latency_ms.maximum_ms == pytest.approx(22.0)


def test_workload_summary_rejects_duplicate_query_ids():
    query=WorkloadQueryBenchmark("q1",_paired(10,5),_paired(10,5),False,"",1.0)
    with pytest.raises(ValueError,match="duplicate query IDs"):
        summarize_workload_benchmark([query,query])
