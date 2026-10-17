"""Tests for sci_mcp.tools.symbolic (public tool-function interface)."""

from __future__ import annotations

import sympy

from sci_mcp.tools.symbolic import (
    symbolic_calculus,
    symbolic_evaluate,
    symbolic_solve,
)

x, y = sympy.symbols("x y")


def _result_expression(result: dict) -> sympy.Expr:
    """Assert a successful envelope and parse its expression string with sympy."""
    assert result["ok"] is True, result
    return sympy.sympify(result["result"]["expression"])


# ---------------------------------------------------------------------------
# symbolic_calculus
# ---------------------------------------------------------------------------


def test_diff_first_derivative() -> None:
    result = symbolic_calculus("x**2", "diff")
    assert result["ok"] is True
    assert result["result"]["operation"] == "diff"
    value = _result_expression(result)
    assert sympy.simplify(value - 2 * x) == 0


def test_diff_second_derivative_with_order() -> None:
    result = symbolic_calculus("x**3", "diff", order=2)
    value = _result_expression(result)
    assert sympy.simplify(value - 6 * x) == 0


def test_integrate_x_squared_roundtrip() -> None:
    result = symbolic_calculus("x**2", "integrate")
    antiderivative = _result_expression(result)
    # Independently verify: the derivative of the antiderivative is x**2.
    assert sympy.simplify(sympy.diff(antiderivative, x) - x**2) == 0


def test_limit_sin_over_x_at_zero() -> None:
    result = symbolic_calculus("sin(x)/x", "limit", point="0")
    value = _result_expression(result)
    assert sympy.simplify(value - 1) == 0


def test_limit_at_infinity() -> None:
    result = symbolic_calculus("1/x", "limit", point="oo")
    value = _result_expression(result)
    assert value == 0


def test_series_exp_x_order_4() -> None:
    result = symbolic_calculus("exp(x)", "series", order=4)
    truncated = _result_expression(result).removeO()
    expected = 1 + x + x**2 / 2 + x**3 / 6
    assert sympy.expand(truncated - expected) == 0
    # The x**4 and higher terms must be absent from the truncation.
    assert sympy.expand(truncated).as_poly(x).degree() == 3


def test_simplify_pythagorean_identity() -> None:
    result = symbolic_calculus("sin(x)**2 + cos(x)**2", "simplify")
    value = _result_expression(result)
    assert value == 1


def test_expand_product() -> None:
    result = symbolic_calculus("(x + 1)**2", "expand")
    value = _result_expression(result)
    assert sympy.expand(value - (x**2 + 2 * x + 1)) == 0


def test_factor_difference_of_squares() -> None:
    result = symbolic_calculus("x**2 - 1", "factor")
    value = _result_expression(result)
    assert sympy.simplify(value - (x - 1) * (x + 1)) == 0


# ---------------------------------------------------------------------------
# symbolic_solve
# ---------------------------------------------------------------------------


def test_solve_single_equation() -> None:
    result = symbolic_solve("x**2 - 1", "x")
    assert result["ok"] is True
    assert result["result"]["variables"] == ["x"]
    # Single-variable solutions are a flat list of numeric values.
    solutions = result["result"]["solutions"]
    assert sorted(solutions) == [-1, 1]


def test_solve_single_variable_solution_elements_are_numeric() -> None:
    result = symbolic_solve("x**2 - 1", "x")
    assert result["ok"] is True
    solutions = result["result"]["solutions"]
    assert sorted(solutions) == [-1, 1]
    for value in solutions:
        assert isinstance(value, (int, float))
        assert not isinstance(value, bool)
        assert not isinstance(value, (str, list))


def test_solve_linear_system() -> None:
    result = symbolic_solve(["x + y - 3", "x - y - 1"], ["x", "y"])
    assert result["ok"] is True
    assert result["result"]["variables"] == ["x", "y"]
    solutions = result["result"]["solutions"]
    # Multi-variable solutions are a list of {variable: numeric value} dicts.
    assert solutions == [{"x": 2, "y": 1}]
    for solution in solutions:
        for value in solution.values():
            assert isinstance(value, (int, float))
            assert not isinstance(value, (str, list))


def test_solve_no_solution_is_not_an_error() -> None:
    result = symbolic_solve("exp(x)", "x")
    assert result["ok"] is True
    assert result["result"]["solutions"] == []


# ---------------------------------------------------------------------------
# symbolic_evaluate
# ---------------------------------------------------------------------------


def test_evaluate_numeric_substitution() -> None:
    result = symbolic_evaluate("x**2 + y", {"x": 2, "y": 3})
    assert result["ok"] is True
    assert result["result"]["numeric"] == 7.0
    assert sympy.sympify(result["result"]["expression"]) == 7


def test_evaluate_symbolic_constant_string() -> None:
    result = symbolic_evaluate("sin(x)", {"x": "pi/2"})
    assert result["ok"] is True
    assert abs(result["result"]["numeric"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_operation_returns_error() -> None:
    result = symbolic_calculus("x**2", "taylor")
    assert result["ok"] is False
    assert "error" in result
    assert "unknown operation" in result["error"]
    assert "taylor" in result["error"]


def test_invalid_expression_returns_error() -> None:
    result = symbolic_calculus("x**", "diff")
    assert result["ok"] is False
    assert "error" in result
    assert "symbolic_calculus" in result["error"]


def test_solve_count_mismatch_returns_error() -> None:
    result = symbolic_solve(["x + y - 3"], ["x", "y"])
    assert result["ok"] is False
    assert "error" in result
    assert "mismatch" in result["error"]


def test_solve_invalid_expression_returns_error() -> None:
    result = symbolic_solve("x +* 1", "x")
    assert result["ok"] is False
    assert "error" in result


def test_evaluate_free_symbols_returns_error() -> None:
    result = symbolic_evaluate("x + y", {"x": 1})
    assert result["ok"] is False
    assert "error" in result
    assert "expression still contains free symbols" in result["error"]
    assert "y" in result["error"]
