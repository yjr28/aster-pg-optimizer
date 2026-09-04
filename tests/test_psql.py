from aster.integration.psql import _quote_setting


def test_setting_boundary_only_allows_safe_boolean_gucs():
    assert _quote_setting("enable_hashjoin", "off") == "SET LOCAL enable_hashjoin = off;"
