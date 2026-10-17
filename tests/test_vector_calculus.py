"""Tests for sci_mcp.tools.vector_calculus (public tool-function interface)."""

from __future__ import annotations

import json

import sympy

from sci_mcp.tools.vector_calculus import (
    curl,
    divergence,
    gradient,
    laplacian,
)

x, y, z = sympy.symbols("x y z")


def _components(result: dict) -> list[sympy.Expr]:
    """Assert a successful envelope and parse the component strings."""
    assert result["ok"] is True, result
    return [sympy.sympify(component) for component in result["result"]["components"]]


def test_gradient_of_x_squared_plus_y_squared() -> None:
    result = gradient("x**2 + y**2", ["x", "y"])
    assert result["ok"] is True, result
    assert result["result"]["variables"] == ["x", "y"]

    components = _components(result)
    assert len(components) == 2
    assert sympy.simplify(components[0] - 2 * x) == 0
    assert sympy.simplify(components[1] - 2 * y) == 0


def test_gradient_of_product_xyz() -> None:
    components = _components(gradient("x*y*z", ["x", "y", "z"]))
    expected = [y * z, x * z, x * y]
    assert len(components) == 3
    for actual, want in zip(components, expected):
        assert sympy.simplify(actual - want) == 0


def test_divergence_of_position_field_is_three() -> None:
    result = divergence([x, y, z], ["x", "y", "z"])
    assert result["ok"] is True, result
    assert sympy.simplify(sympy.sympify(result["result"]["divergence"]) - 3) == 0


def test_divergence_of_mixed_field() -> None:
    result = divergence(["x*y", "y*z", "z*x"], ["x", "y", "z"])
    assert result["ok"] is True, result
    actual = sympy.sympify(result["result"]["divergence"])
    assert sympy.simplify(actual - (x + y + z)) == 0


def test_curl_of_rotation_field() -> None:
    result = curl(["-y", "x", "0"], ["x", "y", "z"])
    assert result["ok"] is True, result
    components = _components(result)
    expected = [sympy.Integer(0), sympy.Integer(0), sympy.Integer(2)]
    assert len(components) == 3
    for actual, want in zip(components, expected):
        assert sympy.simplify(actual - want) == 0


def test_curl_requires_three_dimensions() -> None:
    result = curl(["x", "y"], ["x", "y"])
    assert result["ok"] is False
    assert "error" in result
    assert "3" in result["error"]


def test_laplacian_of_x_squared_plus_y_squared() -> None:
    result = laplacian("x**2 + y**2", ["x", "y"])
    assert result["ok"] is True, result
    assert sympy.simplify(sympy.sympify(result["result"]["laplacian"]) - 4) == 0


def test_laplacian_of_sine_cosine_product() -> None:
    result = laplacian("sin(x)*cos(y)", ["x", "y"])
    assert result["ok"] is True, result
    actual = sympy.sympify(result["result"]["laplacian"])
    expected = -2 * sympy.sin(x) * sympy.cos(y)
    assert sympy.simplify(actual - expected) == 0


def test_invalid_expression_returns_error() -> None:
    result = gradient("x**", ["x"])
    assert result["ok"] is False
    assert "error" in result


def test_divergence_dimension_mismatch_returns_error() -> None:
    result = divergence(["x", "y"], ["x", "y", "z"])
    assert result["ok"] is False
    assert "error" in result
    assert "mismatch" in result["error"]


def test_results_are_json_serializable() -> None:
    for result in (
        gradient("x**2 + y**2", ["x", "y"]),
        divergence(["x", "y", "z"], ["x", "y", "z"]),
        curl(["-y", "x", "0"], ["x", "y", "z"]),
        laplacian("x**2 + y**2", ["x", "y"]),
    ):
        assert result["ok"] is True, result
        json.dumps(result)
