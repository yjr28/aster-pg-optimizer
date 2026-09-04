import pytest

from aster.planner import render_set_local


def test_setting_boundary_allows_safe_boolean_and_bounded_geqo_gucs():
    assert render_set_local("enable_hashjoin", "off") == "SET LOCAL enable_hashjoin = off;"
    assert render_set_local("geqo_seed", "0.60") == "SET LOCAL geqo_seed = 0.60;"
    assert render_set_local("geqo_effort", "10") == "SET LOCAL geqo_effort = 10;"


def test_setting_boundary_rejects_arbitrary_gucs_and_unsafe_values():
    with pytest.raises(ValueError, match="unsupported planner GUC"):
        render_set_local("search_path", "public")
    with pytest.raises(ValueError, match="geqo_seed"):
        render_set_local("geqo_seed", "0.5; DROP TABLE users")
