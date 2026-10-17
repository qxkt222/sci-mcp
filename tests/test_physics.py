"""Tests for the physics constants and unit tools (public function interface)."""

from __future__ import annotations

import pint
import pytest

from sci_mcp.tools.physics import physical_constant, unit_calculate, unit_convert

_UREG = pint.UnitRegistry()


def test_convert_inch_to_cm() -> None:
    result = unit_convert(1, "inch", "cm")
    assert result["ok"] is True
    assert result["result"]["value"] == pytest.approx(2.54, rel=1e-12)
    assert _UREG.Unit(result["result"]["unit"]) == _UREG.centimeter


def test_convert_degc_to_kelvin() -> None:
    # pint offset units must be built as Quantity(value, unit); value * ureg.degC
    # raises OffsetUnitCalculusError, which is why unit_convert uses Quantity.
    result = unit_convert(100, "degC", "K")
    assert result["ok"] is True
    assert result["result"]["value"] == pytest.approx(373.15, rel=1e-12)
    assert _UREG.Unit(result["result"]["unit"]) == _UREG.kelvin
    assert "degC" not in result["result"]["expression"]


def test_constant_speed_of_light() -> None:
    result = physical_constant("c")
    assert result["ok"] is True
    payload = result["result"]
    assert payload["value"] == 299792458
    assert "m/s" in payload["unit"]
    assert payload["uncertainty"] == 0.0
    assert payload["name"] == "speed of light in vacuum"


def test_constant_gravitational_constant() -> None:
    for query in ("gravitational constant", "G"):
        result = physical_constant(query)
        assert result["ok"] is True, result
        assert result["result"]["value"] == pytest.approx(6.674e-11, rel=1e-3)
        assert result["result"]["name"] == "Newtonian constant of gravitation"


def test_required_aliases_resolve() -> None:
    aliases = [
        "c",
        "g",
        "h",
        "hbar",
        "G",
        "e",
        "k",
        "k_B",
        "N_A",
        "R",
        "sigma",
        "epsilon_0",
        "mu_0",
        "alpha",
        "m_e",
        "m_p",
        "m_n",
        "atm",
        "eV",
    ]
    for alias in aliases:
        result = physical_constant(alias)
        assert result["ok"] is True, f"{alias}: {result}"
        payload = result["result"]
        assert "matches" not in payload
        assert isinstance(payload["name"], str)
        assert isinstance(payload["value"], float)
        assert isinstance(payload["unit"], str)
        assert isinstance(payload["uncertainty"], float)


def test_constant_negative_g_and_lowercase_g_differ() -> None:
    gravity = physical_constant("g")
    gravitational = physical_constant("G")
    assert gravity["ok"] is True and gravitational["ok"] is True
    assert gravity["result"]["value"] == pytest.approx(9.80665, rel=1e-12)
    assert gravity["result"]["name"] != gravitational["result"]["name"]


def test_calculate_rate() -> None:
    result = unit_calculate("3 m / 2 s")
    assert result["ok"] is True
    payload = result["result"]
    assert payload["value"] == pytest.approx(1.5, rel=1e-12)
    assert _UREG.Unit(payload["unit"]) == _UREG.meter / _UREG.second


def test_calculate_force() -> None:
    result = unit_calculate("9.81 m/s**2 * 2 kg")
    assert result["ok"] is True
    payload = result["result"]
    assert payload["value"] == pytest.approx(19.62, rel=1e-12)
    force = _UREG.Quantity(payload["value"], payload["unit"])
    assert force.check("[force]")
    assert force.to("N").magnitude == pytest.approx(19.62, rel=1e-12)


def test_calculate_speed_mph() -> None:
    result = unit_calculate("60 mph")
    assert result["ok"] is True
    payload = result["result"]
    speed = _UREG.Quantity(payload["value"], payload["unit"])
    assert speed.to("m/s").magnitude == pytest.approx(26.8224, rel=1e-9)


def test_convert_incompatible_units_fails() -> None:
    result = unit_convert(1, "m", "s")
    assert result["ok"] is False
    assert "error" in result


def test_convert_unknown_unit_fails() -> None:
    result = unit_convert(1, "not_a_unit", "m")
    assert result["ok"] is False
    assert "error" in result


def test_calculate_invalid_expression_fails() -> None:
    result = unit_calculate("3 m / / 2")
    assert result["ok"] is False
    assert "error" in result


def test_unknown_constant_fails() -> None:
    result = physical_constant("definitely-not-a-constant-xyz")
    assert result["ok"] is False
    assert "error" in result


def test_ambiguous_constant_query_returns_matches() -> None:
    result = physical_constant("mass")
    assert result["ok"] is True
    payload = result["result"]
    assert "matches" in payload
    matches = payload["matches"]
    assert 1 < len(matches) <= 20
    for match in matches:
        assert set(match) == {"name", "value", "unit", "uncertainty"}
        assert isinstance(match["name"], str)
        assert isinstance(match["value"], float)
        assert isinstance(match["unit"], str)
        assert isinstance(match["uncertainty"], float)
