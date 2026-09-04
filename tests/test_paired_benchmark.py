from __future__ import annotations

from collections import defaultdict

import pytest

from aster.benchmarks import run_paired_benchmark
from aster.candidates import CandidateSpec, DiscoveredCandidate
from aster.plans import parse_explain_json, plan_fingerprint


def _plan(node_type: str, execution: float, planning: float = 1.0):
    return [{
        "Plan": {
            "Node Type": node_type,
            "Startup Cost": 0.0,
            "Total Cost": 10.0,
            "Plan Rows": 10,
            "Plan Width": 8,
            "Actual Rows": 10,
            "Actual Loops": 1,
        },
        "Planning Time": planning,
        "Execution Time": execution,
    }]


def _candidate(candidate_id: str, settings: dict[str, str], node_type: str):
    raw = _plan(node_type, 1.0)
    return DiscoveredCandidate(
        CandidateSpec(candidate_id, settings),
        plan_fingerprint(parse_explain_json(raw)),
        raw,
    )


class FakeRunner:
    def __init__(self, native_times, aster_times, *, drift_after=None):
        self.times = {
            "native": iter(native_times),
            "aster": iter(aster_times),
        }
        self.calls = []
        self.counts = defaultdict(int)
        self.drift_after = drift_after

    def explain(self, query, settings, *, analyze):
        label = "aster" if settings else "native"
        self.calls.append(label)
        self.counts[label] += 1
        execution = next(self.times[label])
        node = "Index Scan" if label == "aster" else "Seq Scan"
        if self.drift_after is not None and self.counts[label] > self.drift_after:
            node = "Bitmap Heap Scan"
        return _plan(node, execution, planning=0.5)

    def postgres_version(self):
        return "17.11"


def test_paired_benchmark_interleaves_and_reports_execution_vs_end_to_end():
    native = _candidate("native", {}, "Seq Scan")
    selected = _candidate("no_seqscan", {"enable_seqscan": "off"}, "Index Scan")
    runner = FakeRunner([11, 10, 10, 10, 10], [6, 5, 5, 5, 5])
    result = run_paired_benchmark(
        runner, "select 1", native, selected,
        selection_overhead_ms=1.0, warmups=1, repetitions=4, seed=3,
    )

    assert result.postgres_version == "17.11"
    assert result.same_physical_plan is False
    assert result.native_execution.median_ms == 10
    assert result.aster_execution.median_ms == 5
    assert result.execution_speedup_geomean == pytest.approx(2.0)
    assert result.end_to_end_speedup_geomean == pytest.approx(10.5 / 6.5)
    assert result.improved_fraction == 1.0
    assert result.regressed_fraction == 0.0
    assert {sample.execution_order for sample in result.samples} == {
        ("native", "aster"), ("aster", "native")
    }
    assert len(result.to_jsonable()["samples"]) == 4


def test_native_fallback_executes_one_shared_sample_per_repetition():
    native = _candidate("native", {}, "Seq Scan")
    runner = FakeRunner([9, 8, 8, 8], [])
    result = run_paired_benchmark(
        runner, "select 1", native, native,
        selection_overhead_ms=0.2, warmups=1, repetitions=3,
    )

    assert runner.calls == ["native"] * 4
    assert result.same_physical_plan is True
    assert result.execution_speedup_geomean == pytest.approx(1.0)
    assert result.worst_regression_ratio == pytest.approx(1.0)
    assert all(s.execution_order == ("native_shared",) for s in result.samples)
    assert result.end_to_end_speedup_geomean < 1.0


def test_paired_benchmark_rejects_plan_drift():
    native = _candidate("native", {}, "Seq Scan")
    selected = _candidate("no_seqscan", {"enable_seqscan": "off"}, "Index Scan")
    runner = FakeRunner([10, 10], [5, 5], drift_after=1)
    with pytest.raises(RuntimeError, match="plan changed during paired benchmark"):
        run_paired_benchmark(
            runner, "select 1", native, selected,
            selection_overhead_ms=0.1, warmups=0, repetitions=2, seed=1,
        )
