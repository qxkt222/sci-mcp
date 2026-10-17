"""Physics constants and unit tools backed by scipy.constants and pint."""

from __future__ import annotations

import re

import pint
from mcp.server.mcpserver import MCPServer
from scipy.constants import physical_constants as _PHYSICAL_CONSTANTS

from ..core import err, ok, to_jsonable

# One registry per process: creating a UnitRegistry loads the unit definition file.
_ureg = pint.UnitRegistry()

_MAX_MATCHES = 20

# Common shorthands mapped to exact scipy.constants.physical_constants keys.
# Lookup is case-sensitive first so that "g" (standard gravity) and "G"
# (gravitational constant) stay distinct.
_ALIASES: dict[str, str] = {
    "c": "speed of light in vacuum",
    "c0": "speed of light in vacuum",
    "speed of light": "speed of light in vacuum",
    "g": "standard acceleration of gravity",
    "gravity": "standard acceleration of gravity",
    "standard gravity": "standard acceleration of gravity",
    "h": "Planck constant",
    "planck": "Planck constant",
    "planck constant": "Planck constant",
    "hbar": "reduced Planck constant",
    "h-bar": "reduced Planck constant",
    "reduced planck constant": "reduced Planck constant",
    "G": "Newtonian constant of gravitation",
    "gravitational constant": "Newtonian constant of gravitation",
    "newtonian constant of gravitation": "Newtonian constant of gravitation",
    "e": "elementary charge",
    "elementary charge": "elementary charge",
    "k": "Boltzmann constant",
    "k_B": "Boltzmann constant",
    "kb": "Boltzmann constant",
    "boltzmann": "Boltzmann constant",
    "boltzmann constant": "Boltzmann constant",
    "N_A": "Avogadro constant",
    "NA": "Avogadro constant",
    "avogadro": "Avogadro constant",
    "avogadro constant": "Avogadro constant",
    "R": "molar gas constant",
    "gas constant": "molar gas constant",
    "molar gas constant": "molar gas constant",
    "sigma": "Stefan-Boltzmann constant",
    "stefan-boltzmann": "Stefan-Boltzmann constant",
    "stefan boltzmann": "Stefan-Boltzmann constant",
    "stefan-boltzmann constant": "Stefan-Boltzmann constant",
    "epsilon_0": "vacuum electric permittivity",
    "epsilon0": "vacuum electric permittivity",
    "eps0": "vacuum electric permittivity",
    "vacuum permittivity": "vacuum electric permittivity",
    "vacuum electric permittivity": "vacuum electric permittivity",
    "electric constant": "vacuum electric permittivity",
    "mu_0": "vacuum mag. permeability",
    "mu0": "vacuum mag. permeability",
    "vacuum magnetic permeability": "vacuum mag. permeability",
    "vacuum mag. permeability": "vacuum mag. permeability",
    "magnetic constant": "vacuum mag. permeability",
    "alpha": "fine-structure constant",
    "fine structure constant": "fine-structure constant",
    "fine-structure constant": "fine-structure constant",
    "m_e": "electron mass",
    "me": "electron mass",
    "electron mass": "electron mass",
    "m_p": "proton mass",
    "mp": "proton mass",
    "proton mass": "proton mass",
    "m_n": "neutron mass",
    "mn": "neutron mass",
    "neutron mass": "neutron mass",
    "atm": "standard atmosphere",
    "atmosphere": "standard atmosphere",
    "standard atmosphere": "standard atmosphere",
    "eV": "electron volt",
    "electronvolt": "electron volt",
    "electron volt": "electron volt",
}

# Pint evaluates "3 m / 2 s" left-to-right as (3*m/2)*s. Physical notation means
# 3*m/(2*s), so wrap a numeric denominator and the unit factors that follow it.
_UNIT_TOKEN = r"[A-Za-z_µ°Ω%][A-Za-z0-9_µ°Ω]*"
_DIV_NUMBER_WITH_UNITS = re.compile(
    r"/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"((?:\s*" + _UNIT_TOKEN + r"(?:\*\*\s*[+-]?\d+)?)+)"
)


def _unit_text(unit: str) -> str:
    """Normalize a scipy unit string to compact pint syntax, e.g. "m s^-1" -> "m/s"."""
    if not unit:
        return ""
    try:
        return format(_ureg.Unit(unit), "~").replace(" ", "")
    except Exception:
        return unit


def _describe(name: str) -> dict:
    """Build the public record for one entry of scipy.constants.physical_constants."""
    value, unit, uncertainty = _PHYSICAL_CONSTANTS[name]
    return to_jsonable(
        {
            "name": name,
            "value": float(value),
            "unit": _unit_text(unit),
            "uncertainty": float(uncertainty),
        }
    )


def _lookup_alias(query: str) -> str | None:
    """Resolve a shorthand to a constant key; fall back to case-insensitive lookup."""
    if query in _ALIASES:
        return _ALIASES[query]
    lowered = query.lower()
    targets = {target for key, target in _ALIASES.items() if key.lower() == lowered}
    if len(targets) == 1:
        return targets.pop()
    return None


def _group_division_denominators(expression: str) -> str:
    """Rewrite a numeric denominator with units as an explicit parenthesized group."""
    return _DIV_NUMBER_WITH_UNITS.sub(
        lambda match: f"/ ({match.group(1)}{match.group(2)})", expression
    )


def physical_constant(query: str) -> dict:
    """Look up a physical constant by alias, exact name, or keyword; return value, unit and uncertainty."""
    try:
        if not isinstance(query, str) or not query.strip():
            return err("physical_constant failed: query must be a non-empty string")
        text = query.strip()

        alias_name = _lookup_alias(text)
        if alias_name is not None and alias_name in _PHYSICAL_CONSTANTS:
            return ok(_describe(alias_name))

        lowered = text.lower()
        exact = [name for name in _PHYSICAL_CONSTANTS if name.lower() == lowered]
        if len(exact) == 1:
            return ok(_describe(exact[0]))

        if exact:
            candidates = exact
        else:
            candidates = [name for name in _PHYSICAL_CONSTANTS if lowered in name.lower()]

        names = sorted(set(candidates), key=str.lower)[:_MAX_MATCHES]
        if not names:
            return err(f"physical_constant failed: no constant matches {query!r}")
        if len(names) == 1:
            return ok(_describe(names[0]))
        return ok({"matches": [_describe(name) for name in names]})
    except Exception as exc:
        return err(f"physical_constant failed: {exc}")


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a numeric value between units, including offset units such as degC."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return err("unit_convert failed: value must be a number")
        if not isinstance(from_unit, str) or not isinstance(to_unit, str):
            return err("unit_convert failed: from_unit and to_unit must be strings")

        # Quantity(value, unit) is required for offset units: value * ureg.degC
        # raises OffsetUnitCalculusError in pint.
        converted = _ureg.Quantity(float(value), from_unit).to(to_unit)
        return ok(
            {
                "value": float(converted.magnitude),
                "unit": str(converted.units),
                "expression": str(converted),
            }
        )
    except Exception as exc:
        return err(f"unit_convert failed: {exc}")


def unit_calculate(expression: str) -> dict:
    """Evaluate a unit-bearing arithmetic expression and return its value and unit."""
    try:
        if not isinstance(expression, str) or not expression.strip():
            return err("unit_calculate failed: expression must be a non-empty string")

        normalized = _group_division_denominators(expression)
        result = _ureg.parse_expression(normalized)

        if isinstance(result, _ureg.Quantity):
            return ok(
                {
                    "value": float(result.magnitude),
                    "unit": str(result.units),
                    "expression": str(result),
                }
            )
        return ok({"value": float(result), "unit": "", "expression": str(result)})
    except Exception as exc:
        return err(f"unit_calculate failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(physical_constant)
    mcp.tool()(unit_convert)
    mcp.tool()(unit_calculate)
