"""Tests for the health_check tool."""

import json

from sci_mcp.tools.health import health_check

DOMAIN_MODULES = [
    "data_io",
    "health",
    "montecarlo",
    "numeric",
    "ode_analytic",
    "pde",
    "physics",
    "planning",
    "sandbox",
    "signal_processing",
    "stats_inference",
    "stats_plot",
    "symbolic",
    "vector_calculus",
]


def _payload() -> dict:
    """Call health_check and return its result payload."""
    result = health_check()
    assert result["ok"] is True
    assert isinstance(result["result"], dict)
    return result["result"]


def test_health_check_returns_ok() -> None:
    result = health_check()
    assert result["ok"] is True
    assert "result" in result


def test_module_count_is_at_least_eight() -> None:
    payload = _payload()
    assert payload["module_count"] >= 8
    assert payload["module_count"] == len(payload["modules"])


def test_result_includes_health_module_itself() -> None:
    payload = _payload()
    names = [record["module"] for record in payload["modules"]]
    assert "health" in names
    assert names == sorted(names)


def test_no_failed_modules() -> None:
    payload = _payload()
    assert payload["failed_modules"] == []
    assert payload["healthy"] is True
    assert all(record["imported"] is True for record in payload["modules"])
    assert all(record["error"] is None for record in payload["modules"])


def test_all_domain_modules_have_register() -> None:
    payload = _payload()
    by_name = {record["module"]: record for record in payload["modules"]}
    for name in DOMAIN_MODULES:
        assert name in by_name, f"missing domain module: {name}"
        assert by_name[name]["has_register"] is True, f"{name} lacks register()"


def test_every_module_exposes_tool_functions() -> None:
    payload = _payload()
    for record in payload["modules"]:
        assert record["tool_functions"], f"{record['module']} exposes no tool functions"
        assert record["tool_functions"] == sorted(record["tool_functions"])
        assert all(not name.startswith("_") for name in record["tool_functions"])
        assert "register" not in record["tool_functions"]

    by_name = {record["module"]: record for record in payload["modules"]}
    assert "health_check" in by_name["health"]["tool_functions"]
    # Imported helpers must not be reported as tool functions of this module.
    assert "ok" not in by_name["health"]["tool_functions"]


def test_tool_function_count_matches_module_records() -> None:
    payload = _payload()
    expected = sum(len(record["tool_functions"]) for record in payload["modules"])
    assert payload["tool_function_count"] == expected
    assert payload["tool_function_count"] == 65
    assert payload["tool_function_count"] >= payload["module_count"]


def test_result_is_strict_json_serializable() -> None:
    payload = _payload()
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == payload
