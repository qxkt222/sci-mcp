"""Tests for sci_mcp.tools.planning against independently known LP/MILP optima."""

from __future__ import annotations

import json
import math

import pytest

from sci_mcp.tools.planning import integer_program, linear_program


def test_linear_program_corner_optimum() -> None:
    """min -3x0-2x1 s.t. x0+x1<=4, x0<=2, x1<=3, x>=0 has optimum x=[2,2], fun=-10."""
    result = linear_program(
        [-3.0, -2.0],
        a_ub=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        b_ub=[4.0, 2.0, 3.0],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["status"] == 0
    assert payload["x"] == pytest.approx([2.0, 2.0])
    assert payload["fun"] == pytest.approx(-10.0)


def test_linear_program_equality_constraint_is_satisfied() -> None:
    """min x0+x1 s.t. x0+x1=1 returns a feasible point on the equality line."""
    result = linear_program([1.0, 1.0], a_eq=[[1.0, 1.0]], b_eq=[1.0])

    assert result["ok"] is True
    x = result["result"]["x"]
    assert math.isclose(x[0] + x[1], 1.0, abs_tol=1e-9)
    assert x[0] >= 0.0 and x[1] >= 0.0
    assert result["result"]["fun"] == pytest.approx(1.0)


def test_linear_program_infeasible_problem_returns_error() -> None:
    """x >= 1 together with x <= 0 is infeasible, so success is False."""
    result = linear_program([1.0], a_ub=[[1.0]], b_ub=[0.0], bounds=[[1.0, None]])

    assert result["ok"] is False
    assert "error" in result
    assert result["success"] is False
    assert result["status"] != 0


def test_integer_program_binary_knapsack_optimum() -> None:
    """max 5x0+4x1 s.t. 2x0+3x1<=6, x in {0,1} has optimum x=[1,1], fun=-9."""
    result = integer_program(
        [-5.0, -4.0],
        a_ub=[[2.0, 3.0]],
        b_ub=[6.0],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"] == pytest.approx([1.0, 1.0])
    assert payload["fun"] == pytest.approx(-9.0)
    assert all(float(value).is_integer() for value in payload["x"])


def test_integer_program_differs_from_continuous_relaxation() -> None:
    """min -x0-x1, 2x0+2x1<=3, 0<=x<=1: LP gives -1.5, integer optimum is -1."""
    lp = linear_program([-1.0, -1.0], a_ub=[[2.0, 2.0]], b_ub=[3.0], bounds=[[0.0, 1.0], [0.0, 1.0]])
    ip = integer_program(
        [-1.0, -1.0],
        integrality=[1, 1],
        a_ub=[[2.0, 2.0]],
        b_ub=[3.0],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
    )

    assert lp["ok"] is True and ip["ok"] is True
    assert lp["result"]["fun"] == pytest.approx(-1.5)
    assert ip["result"]["fun"] == pytest.approx(-1.0)
    assert all(float(value).is_integer() for value in ip["result"]["x"])
    assert ip["result"]["fun"] > lp["result"]["fun"]


def test_linear_program_dimension_mismatch_returns_error() -> None:
    """A constraint matrix whose column count disagrees with c is rejected."""
    result = linear_program([1.0, 2.0, 3.0], a_ub=[[1.0, 2.0]], b_ub=[1.0])

    assert result["ok"] is False
    assert "error" in result


def test_linear_program_bounds_count_mismatch_returns_error() -> None:
    """bounds must contain exactly one pair per variable."""
    result = linear_program([1.0, 2.0], bounds=[[0.0, None]])

    assert result["ok"] is False
    assert "error" in result


def test_integer_program_integrality_length_mismatch_returns_error() -> None:
    """integrality must contain exactly one flag per variable."""
    result = integer_program([1.0, 2.0], integrality=[1])

    assert result["ok"] is False
    assert "error" in result


def test_planning_results_are_json_serializable() -> None:
    """Both envelopes can be passed straight to json.dumps."""
    lp = linear_program([-3.0, -2.0], a_ub=[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], b_ub=[4.0, 2.0, 3.0])
    ip = integer_program(
        [-5.0, -4.0], a_ub=[[2.0, 3.0]], b_ub=[6.0], bounds=[[0.0, 1.0], [0.0, 1.0]]
    )

    for result in (lp, ip):
        assert result["ok"] is True
        decoded = json.loads(json.dumps(result))
        assert decoded["result"]["status"] == 0
        assert isinstance(decoded["result"]["x"], list)
        assert isinstance(decoded["result"]["fun"], float)
