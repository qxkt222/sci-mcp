"""Tests for sci_mcp.tools.ode_analytic (public tool-function interface)."""

from __future__ import annotations

import json

import sympy

from sci_mcp.tools.ode_analytic import dsolve_ode

t = sympy.Symbol("t")
y = sympy.Function("y")


def _solution_expr(result: dict) -> sympy.Expr:
    """Assert a successful single-solution envelope and return the right-hand side."""
    assert result["ok"] is True, result
    solutions = result["result"]["solutions"]
    assert len(solutions) == 1, result
    parsed = sympy.sympify(solutions[0])
    assert isinstance(parsed, sympy.Eq), solutions[0]
    assert parsed.lhs == y(t), solutions[0]
    return parsed.rhs


def test_second_order_homogeneous_general_solution_satisfies_ode() -> None:
    result = dsolve_ode("Derivative(y(t), t, 2) + y(t)")
    assert result["ok"] is True, result
    assert result["result"]["order"] == 2
    assert result["result"]["particular"] is False

    rhs = _solution_expr(result)
    # Independent check: substituting the solution must zero the ODE.
    assert sympy.simplify(sympy.diff(rhs, t, 2) + rhs) == 0
    # y'' + y = 0 has sin/cos (or the equivalent complex exponential) solutions.
    assert rhs.has(sympy.sin) or rhs.has(sympy.cos) or rhs.has(sympy.exp)


def test_first_order_solution_matches_independent_dsolve() -> None:
    result = dsolve_ode("Eq(Derivative(y(t), t), y(t))")
    assert result["ok"] is True, result
    assert result["result"]["order"] == 1

    rhs = _solution_expr(result)
    assert sympy.simplify(sympy.diff(rhs, t) - rhs) == 0

    # Compare against a direct sympy.dsolve call, not against our tool.
    expected = sympy.dsolve(
        sympy.Eq(sympy.Derivative(y(t), t), y(t)), y(t)
    ).rhs
    assert sympy.simplify(rhs - expected) == 0


def test_initial_value_problem_produces_sine() -> None:
    result = dsolve_ode(
        "Derivative(y(t), t, 2) + y(t)",
        ics={"y(0)": 0, "y'(0)": 1},
    )
    assert result["ok"] is True, result
    assert result["result"]["particular"] is True

    rhs = _solution_expr(result)
    assert sympy.simplify(rhs - sympy.sin(t)) == 0


def test_first_order_decay_matches_c1_exp_minus_2t() -> None:
    result = dsolve_ode("Derivative(y(t), t) + 2*y(t)")
    assert result["ok"] is True, result

    rhs = _solution_expr(result)
    c1 = sympy.Symbol("C1")
    assert sympy.simplify(rhs - c1 * sympy.exp(-2 * t)) == 0


def test_alternative_diff_syntax_is_accepted() -> None:
    result = dsolve_ode("y(t).diff(t, 2) + y(t)")
    assert result["ok"] is True, result
    rhs = _solution_expr(result)
    assert sympy.simplify(sympy.diff(rhs, t, 2) + rhs) == 0


def test_unparseable_equation_returns_error() -> None:
    result = dsolve_ode("Derivative(y(t), t, 2")
    assert result["ok"] is False
    assert "error" in result
    assert "dsolve_ode" in result["error"]


def test_invalid_initial_condition_key_returns_error() -> None:
    result = dsolve_ode("Derivative(y(t), t, 2) + y(t)", ics={"y_at_zero": 1})
    assert result["ok"] is False
    assert "error" in result
    assert "general solution" in result["error"]


def test_result_is_json_serializable() -> None:
    result = dsolve_ode(
        "Derivative(y(t), t, 2) + y(t)",
        ics={"y(0)": 0, "y'(0)": 1},
    )
    assert result["ok"] is True, result
    encoded = json.dumps(result)
    decoded = json.loads(encoded)
    assert decoded["result"]["particular"] is True
    assert decoded["result"]["order"] == 2
