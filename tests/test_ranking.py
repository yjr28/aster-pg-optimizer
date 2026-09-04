from aster.candidates import CandidateSpec, DiscoveredCandidate
from aster.models import RuntimeEnsemble, TrainingExample
from aster.plans import parse_explain_json, plan_fingerprint
from aster.ranking import rank_with_fallback


def raw_plan(cost, rows, node_type):
    return [{"Plan": {"Node Type": node_type, "Total Cost": cost, "Plan Rows": rows, "Plan Width": 8}}]


def discovered(candidate_id, cost, rows, node_type, settings=None):
    raw = raw_plan(cost, rows, node_type)
    return DiscoveredCandidate(
        CandidateSpec(candidate_id, settings or {}),
        plan_fingerprint(parse_explain_json(raw)),
        raw,
    )


def fitted_model():
    rows = [
        (10, 100, "Index Scan", 2),
        (15, 200, "Index Scan", 3),
        (30, 500, "Bitmap Heap Scan", 6),
        (60, 1000, "Seq Scan", 12),
        (100, 5000, "Seq Scan", 25),
        (180, 10000, "Seq Scan", 45),
        (240, 20000, "Seq Scan", 65),
        (300, 30000, "Seq Scan", 85),
    ]
    return RuntimeEnsemble(trees=64, seed=3, min_samples_leaf=1).fit([
        TrainingExample(parse_explain_json(raw_plan(c, r, op)), runtime, f"q{i}")
        for i, (c, r, op, runtime) in enumerate(rows)
    ])


def test_ranker_can_select_non_native_candidate_when_confident():
    model = fitted_model()
    native = discovered("native", 150, 9000, "Seq Scan")
    alternative = discovered("no_seqscan", 20, 250, "Index Scan", {"enable_seqscan": "off"})
    decision = rank_with_fallback(
        model,
        [native, alternative],
        max_log_std=10,
        min_predicted_gain=0.01,
    )
    assert decision.selected.spec.candidate_id == "no_seqscan"
    assert not decision.fallback
    assert decision.decision_overhead_ms >= 0


def test_ranker_falls_back_when_required_gain_is_unreachable():
    model = fitted_model()
    native = discovered("native", 150, 9000, "Seq Scan")
    alternative = discovered("no_seqscan", 20, 250, "Index Scan", {"enable_seqscan": "off"})
    decision = rank_with_fallback(
        model,
        [native, alternative],
        max_log_std=10,
        min_predicted_gain=0.99,
    )
    assert decision.selected is native
    assert decision.fallback
    assert decision.reason == "predicted_gain_too_small"
