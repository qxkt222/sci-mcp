"""Symbolic ordinary differential equation tools: analytic solutions via sympy.dsolve."""

from __future__ import annotations

import re

from mcp.server.mcpserver import MCPServer
from sympy import Derivative, Function, Subs, Symbol, dsolve, sympify

from ..core import err, ok

# Initial-condition keys use the compact form "y(0)", "y'(0)", "y''(0)".
_ICS_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*('*)\(([^()]*)\)\s*$")


def _equation_order(equation: object, y_func: Function, x_sym: Symbol) -> int:
    """Return the highest derivative order of ``y`` with respect to ``x``."""
    target = y_func(x_sym)
    highest = 0
    for derivative in getattr(equation, "atoms", lambda _cls: set())(Derivative):
        dependent = derivative.expr
        if dependent != target and getattr(dependent, "func", None) != y_func:
            continue
        order = 0
        for variable, count in derivative.variable_count:
            if variable == x_sym:
                order += int(count)
        highest = max(highest, order)
    return highest


def _parse_initial_conditions(
    ics: dict[str, float], y_func: Function, x_sym: Symbol
) -> dict:
    """Convert {"y(0)": 0, "y'(0)": 1} into sympy's dsolve ``ics`` mapping."""
    if not isinstance(ics, dict):
        raise TypeError("ics must be a dict mapping e.g. \"y(0)\" to a value")

    parsed: dict = {}
    for raw_key, raw_value in ics.items():
        key = str(raw_key).strip()
        match = _ICS_KEY_PATTERN.match(key)
        if match is None:
            raise ValueError(
                f"invalid initial condition key '{raw_key}'; expected forms like "
                "\"y(0)\", \"y'(0)\", \"y''(0)\""
            )

        name, primes, argument_text = match.groups()
        if name != str(y_func.__name__):
            raise ValueError(
                f"initial condition '{raw_key}' refers to function '{name}', "
                f"but the solution function is '{y_func.__name__}'"
            )

        argument = sympify(argument_text, locals={str(x_sym): x_sym})
        derivative_order = len(primes)

        if derivative_order == 0:
            point_key = y_func(argument)
        else:
            point_key = Subs(
                Derivative(y_func(x_sym), x_sym, derivative_order),
                x_sym,
                argument,
            )
        parsed[point_key] = sympify(raw_value)

    return parsed


def dsolve_ode(
    equation: str,
    y: str = "y",
    x: str = "t",
    ics: dict[str, float] | None = None,
) -> dict:
    """Solve an ODE analytically with sympy.dsolve, optionally with initial conditions."""
    try:
        y_func = Function(str(y))
        x_sym = Symbol(str(x))
        local_dict = {str(y): y_func, str(x): x_sym}

        parsed = sympify(str(equation), locals=local_dict)
        if isinstance(parsed, bool):
            return err("dsolve_ode failed: equation did not parse into a sympy expression")

        # dsolve accepts a bare expression (implicitly == 0) or an Eq.
        order = _equation_order(parsed, y_func, x_sym)

        if ics is not None and not ics:
            ics = None

        initial_conditions = None
        if ics is not None:
            try:
                initial_conditions = _parse_initial_conditions(ics, y_func, x_sym)
            except Exception as exc:
                return err(
                    f"dsolve_ode failed to parse initial conditions: {exc}; "
                    "you can omit ics to obtain the general solution"
                )

        raw_solution = dsolve(parsed, y_func(x_sym), ics=initial_conditions)
        if not isinstance(raw_solution, (list, tuple)):
            raw_solution = [raw_solution]

        solutions = [str(solution) for solution in raw_solution]
        if not solutions:
            return err("dsolve_ode failed: dsolve returned no solutions")

        return ok(
            {
                "solutions": solutions,
                "order": order,
                "particular": initial_conditions is not None,
            }
        )
    except Exception as exc:  # never let an exception escape an MCP tool
        return err(f"dsolve_ode failed: {exc}")


def register(mcp: MCPServer) -> None:
    """Register the symbolic ODE tools with the MCP server."""
    mcp.tool()(dsolve_ode)
