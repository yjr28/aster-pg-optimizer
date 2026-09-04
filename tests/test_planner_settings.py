import pytest

from aster.candidates import CandidateSpec, research_candidates
from aster.planner import render_set_local


def test_research_candidates_are_bounded_and_include_deterministic_geqo_seeds():
    candidates = research_candidates()
    ids = [candidate.candidate_id for candidate in candidates]
    assert ids[0] == "native"
    assert len(candidates) == 16
    assert len(ids) == len(set(ids))
    geqo = [candidate for candidate in candidates if candidate.candidate_id.startswith("geqo_seed_")]
    assert len(geqo) == 6
    assert {candidate.settings["geqo_seed"] for candidate in geqo} == {
        "0.00", "0.20", "0.40", "0.60", "0.80", "1.00"
    }
    assert all(candidate.settings["geqo_threshold"] == "2" for candidate in geqo)


def test_planner_setting_validation_rejects_injection_and_out_of_range_values():
    with pytest.raises(ValueError, match="unsupported planner GUC"):
        CandidateSpec("bad-name", {"search_path": "public"})
    with pytest.raises(ValueError, match="geqo_seed"):
        CandidateSpec("bad-seed", {"geqo_seed": "0.5; DROP TABLE x"})
    with pytest.raises(ValueError, match="geqo_effort"):
        CandidateSpec("bad-effort", {"geqo_effort": "11"})
    with pytest.raises(ValueError, match="geqo_selection_bias"):
        CandidateSpec("bad-bias", {"geqo_selection_bias": "nan"})


def test_render_set_local_accepts_documented_geqo_values_after_validation():
    assert render_set_local("geqo", "on") == "SET LOCAL geqo = on;"
    assert render_set_local("geqo_threshold", "2") == "SET LOCAL geqo_threshold = 2;"
    assert render_set_local("geqo_effort", "10") == "SET LOCAL geqo_effort = 10;"
    assert render_set_local("geqo_seed", "0.60") == "SET LOCAL geqo_seed = 0.60;"
