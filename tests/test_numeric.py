"""Tests for sci_mcp.tools.numeric against independently known ground truth."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sci_mcp.tools.numeric import (
    fft_analysis,
    find_roots,
    global_optimize,
    interpolate_1d,
    matrix_compute,
    optimize_minimize,
    solve_ode,
)


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------


def test_solve_ode_exponential_decay() -> None:
    """y' = -y, y(0) = 1 has the exact solution y(t) = exp(-t)."""
    t_eval = np.linspace(0.0, 3.0, 20).tolist()
    result = solve_ode("-y", [1.0], [0.0, 3.0], t_eval=t_eval)

    assert result["ok"] is True
    t = np.asarray(result["result"]["t"], dtype=float)
    y = np.asarray(result["result"]["y"], dtype=float)
    assert t.shape == (20,)
    assert y.shape == (20,)
    max_relative_error = np.max(np.abs(y - np.exp(-t)) / np.exp(-t))
    assert max_relative_error < 1e-6


def test_solve_ode_harmonic_oscillator() -> None:
    """y0' = y1, y1' = -y0 with y0(0)=1, y1(0)=0 gives y0 = cos(t)."""
    t_eval = np.linspace(0.0, 2.0 * math.pi, 25).tolist()
    result = solve_ode(["y1", "-y0"], [1.0, 0.0], [0.0, 2.0 * math.pi], t_eval=t_eval)

    assert result["ok"] is True
    t = np.asarray(result["result"]["t"], dtype=float)
    y = np.asarray(result["result"]["y"], dtype=float)
    assert y.shape == (2, 25)
    assert np.max(np.abs(y[0] - np.cos(t))) < 1e-3
    assert np.max(np.abs(y[1] + np.sin(t))) < 1e-3


def test_solve_ode_parameters_are_injected() -> None:
    """y' = -k*y with k=2 has solution y(t) = exp(-2t)."""
    t_eval = np.linspace(0.0, 2.0, 15).tolist()
    result = solve_ode("-k*y", [1.0], [0.0, 2.0], t_eval=t_eval, params={"k": 2.0})

    assert result["ok"] is True
    t = np.asarray(result["result"]["t"], dtype=float)
    y = np.asarray(result["result"]["y"], dtype=float)
    assert np.max(np.abs(y - np.exp(-2.0 * t))) < 1e-8


def test_solve_ode_adaptive_sampling_when_t_eval_omitted() -> None:
    """Without t_eval the adaptive solver mesh is returned and is monotone."""
    result = solve_ode("-y", [1.0], [0.0, 2.0])

    assert result["ok"] is True
    t = np.asarray(result["result"]["t"], dtype=float)
    assert t.size >= 2
    assert np.all(np.diff(t) > 0)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def test_optimize_minimize_quadratic() -> None:
    """(x0-2)^2 + (x1+1)^2 is minimized at (2, -1) with value 0."""
    result = optimize_minimize("(x0 - 2)**2 + (x1 + 1)**2", [0.0, 0.0])

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"] == pytest.approx([2.0, -1.0], abs=1e-6)
    assert payload["fun"] == pytest.approx(0.0, abs=1e-10)
    assert payload["nfev"] > 0


def test_optimize_minimize_respects_bounds() -> None:
    """minimize x0^2 on [1, 2] attains its constrained minimum x0 = 1 (fun = 1)."""
    result = optimize_minimize("x0**2", [1.5], bounds=[[1.0, 2.0]])

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"][0] == pytest.approx(1.0, abs=1e-6)
    assert payload["fun"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Dense linear algebra
# ---------------------------------------------------------------------------


def test_matrix_det() -> None:
    result = matrix_compute("det", [[1.0, 2.0], [3.0, 4.0]])
    assert result["ok"] is True
    assert result["result"] == pytest.approx(-2.0, abs=1e-12)


def test_matrix_solve_linear_system() -> None:
    """[[3,1],[1,2]] x = [9,8] has the exact solution x = [2, 3]."""
    result = matrix_compute("solve", [[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
    assert result["ok"] is True
    assert result["result"] == pytest.approx([2.0, 3.0], abs=1e-12)


def test_matrix_eig_diagonal() -> None:
    result = matrix_compute("eig", [[2.0, 0.0], [0.0, 3.0]])
    assert result["ok"] is True
    eigenvalues = sorted(float(v) for v in result["result"]["eigenvalues"])
    assert eigenvalues == pytest.approx([2.0, 3.0], abs=1e-12)


def test_matrix_inv_rank_norm_transpose() -> None:
    """inv([[1,2],[3,4]]) = [[-2, 1],[1.5, -0.5]], rank 2, Frobenius norm sqrt(30)."""
    inverse = matrix_compute("inv", [[1.0, 2.0], [3.0, 4.0]])
    assert inverse["ok"] is True
    assert np.asarray(inverse["result"]) == pytest.approx(
        np.asarray([[-2.0, 1.0], [1.5, -0.5]]), abs=1e-12
    )

    rank = matrix_compute("rank", [[1.0, 2.0], [2.0, 4.0]])
    assert rank["ok"] is True
    assert rank["result"] == 1

    norm = matrix_compute("norm", [[1.0, 2.0], [3.0, 4.0]])
    assert norm["ok"] is True
    assert norm["result"] == pytest.approx(math.sqrt(30.0), rel=1e-12)

    transpose = matrix_compute("transpose", [[1.0, 2.0], [3.0, 4.0]])
    assert transpose["ok"] is True
    assert np.asarray(transpose["result"]) == pytest.approx(
        np.asarray([[1.0, 3.0], [2.0, 4.0]]), abs=1e-12
    )


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------


def test_find_roots_brentq() -> None:
    result = find_roots("x**2 - 4", bracket=[0.0, 3.0])
    assert result["ok"] is True
    assert result["result"]["method"] == "brentq"
    assert result["result"]["roots"] == pytest.approx([2.0], abs=1e-10)


def test_find_roots_newton() -> None:
    result = find_roots("x**2 - 4", x0=3.0)
    assert result["ok"] is True
    assert result["result"]["method"] == "newton"
    assert result["result"]["roots"] == pytest.approx([2.0], abs=1e-10)


def test_find_roots_custom_variable_and_params() -> None:
    """omega^2 * u^2 - 9 = 0 with omega=3 has root u=1 on [0, 2]."""
    result = find_roots("omega**2 * u**2 - 9", variable="u", bracket=[0.0, 2.0], params={"omega": 3.0})
    assert result["ok"] is True
    assert result["result"]["roots"] == pytest.approx([1.0], abs=1e-10)


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def test_fft_dominant_frequency() -> None:
    """A 1 Hz sine sampled at 8 Hz for 2 s has its dominant bin at 1 Hz."""
    sample_rate = 8.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * math.pi * 1.0 * t)

    result = fft_analysis(signal.tolist(), sample_rate=sample_rate)
    assert result["ok"] is True
    payload = result["result"]
    assert payload["dominant_frequency"] == pytest.approx(1.0, abs=0.1)
    assert len(payload["frequencies"]) == len(payload["amplitudes"])

    index = int(np.argmin(np.abs(np.asarray(payload["frequencies"]) - 1.0)))
    assert payload["amplitudes"][index] == pytest.approx(1.0, abs=1e-6)


def test_fft_nyquist_amplitude_is_not_doubled() -> None:
    """[1, -1, 1, -1] at 4 Hz is a pure Nyquist (2 Hz) component of amplitude 1.0."""
    result = fft_analysis([1.0, -1.0, 1.0, -1.0], sample_rate=4.0)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["frequencies"] == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)
    assert payload["dominant_frequency"] == pytest.approx(2.0, abs=1e-12)

    nyquist_index = int(np.argmin(np.abs(np.asarray(payload["frequencies"]) - 2.0)))
    # The old implementation doubled every non-DC bin and reported 2.0 here.
    assert payload["amplitudes"][nyquist_index] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def test_interpolate_1d_linear() -> None:
    result = interpolate_1d([0.0, 1.0], [0.0, 2.0], [0.5])
    assert result["ok"] is True
    assert result["result"]["xi"] == pytest.approx([0.5])
    assert result["result"]["yi"] == pytest.approx([1.0], abs=1e-12)


def test_interpolate_1d_quadratic_exact_on_parabola() -> None:
    """Quadratic interpolation reproduces x^2 exactly."""
    x = [-1.0, 0.0, 1.0]
    y = [1.0, 0.0, 1.0]
    result = interpolate_1d(x, y, [-0.5, 0.5], kind="quadratic")
    assert result["ok"] is True
    assert result["result"]["yi"] == pytest.approx([0.25, 0.25], abs=1e-10)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_matrix_unknown_operation_error() -> None:
    result = matrix_compute("diagonalize", [[1.0, 0.0], [0.0, 1.0]])
    assert result["ok"] is False
    assert "error" in result
    assert "unknown operation" in result["error"]


def test_matrix_singular_matrix_error() -> None:
    result = matrix_compute("inv", [[1.0, 2.0], [2.0, 4.0]])
    assert result["ok"] is False
    assert "error" in result


def test_matrix_dimension_mismatch_error() -> None:
    result = matrix_compute("solve", [[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0])
    assert result["ok"] is False
    assert "error" in result


def test_solve_ode_invalid_expression_error() -> None:
    result = solve_ode("sin(", [1.0], [0.0, 1.0])
    assert result["ok"] is False
    assert "error" in result


def test_solve_ode_state_length_mismatch_error() -> None:
    result = solve_ode(["y1", "-y0"], [1.0], [0.0, 1.0])
    assert result["ok"] is False
    assert "error" in result


def test_solve_ode_unknown_symbol_error() -> None:
    result = solve_ode("-k*y", [1.0], [0.0, 1.0])
    assert result["ok"] is False
    assert "error" in result
    assert "unknown symbol" in result["error"]


def test_find_roots_requires_bracket_or_x0_error() -> None:
    result = find_roots("x**2 - 4")
    assert result["ok"] is False
    assert "error" in result


def test_find_roots_no_sign_change_error() -> None:
    result = find_roots("x**2 + 1", bracket=[-1.0, 1.0])
    assert result["ok"] is False
    assert "error" in result


def test_fft_empty_signal_error() -> None:
    result = fft_analysis([], sample_rate=2.0)
    assert result["ok"] is False
    assert "error" in result


def test_fft_invalid_sample_rate_error() -> None:
    result = fft_analysis([1.0, 2.0, 3.0], sample_rate=0.0)
    assert result["ok"] is False
    assert "error" in result


def test_interpolate_length_mismatch_error() -> None:
    result = interpolate_1d([0.0, 1.0, 2.0], [0.0, 1.0], [0.5])
    assert result["ok"] is False
    assert "error" in result


def test_interpolate_out_of_bounds_error() -> None:
    result = interpolate_1d([0.0, 1.0], [0.0, 2.0], [1.5])
    assert result["ok"] is False
    assert "error" in result


def test_interpolate_unknown_kind_error() -> None:
    result = interpolate_1d([0.0, 1.0], [0.0, 2.0], [0.5], kind="spline")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Regression: parameter names that shadow sympy built-ins
# ---------------------------------------------------------------------------


def test_solve_ode_sir_with_builtin_named_params() -> None:
    """SIR rhs uses ``beta``, ``gamma`` and ``N``, all sympy built-in names.

    Those names must be parsed as parameter Symbols, not as sympy functions.
    With beta=0.3, gamma=0.1, N=1000 the infectious count peaks well above 100
    and has declined by the end of the 60-day window.
    """
    result = solve_ode(
        rhs=["-beta*y0*y1/N", "beta*y0*y1/N - gamma*y1", "gamma*y1"],
        y0=[999.0, 1.0, 0.0],
        t_span=[0.0, 60.0],
        params={"beta": 0.3, "gamma": 0.1, "N": 1000.0},
    )

    assert result["ok"] is True
    payload = result["result"]
    t = np.asarray(payload["t"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(60.0)
    assert y.shape == (3, t.size)

    infectious = y[1]
    peak = float(np.max(infectious))
    assert peak > 100.0
    # Recovery trend: the final infectious count is below the peak.
    assert infectious[-1] < peak


def test_optimize_minimize_param_named_beta() -> None:
    """``beta`` in the objective is a parameter, not sympy's beta function."""
    result = optimize_minimize("beta*(x0 - 1)**2", [0.0], params={"beta": 2.0})

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"][0] == pytest.approx(1.0, abs=1e-6)
    assert payload["fun"] == pytest.approx(0.0, abs=1e-10)


def test_find_roots_param_named_beta() -> None:
    """x**2 - beta with beta=4 has the root 2 on [0, 3]."""
    result = find_roots("x**2 - beta", bracket=[0.0, 3.0], params={"beta": 4.0})

    assert result["ok"] is True
    assert result["result"]["roots"] == pytest.approx([2.0], abs=1e-10)


# ---------------------------------------------------------------------------
# Optimization with constraints
# ---------------------------------------------------------------------------


def test_optimize_minimize_equality_constraint() -> None:
    """min x0^2 + x1^2 subject to x0 + x1 = 1 has the known solution (0.5, 0.5)."""
    result = optimize_minimize(
        "x0**2 + x1**2",
        [0.0, 0.0],
        constraints=[{"kind": "eq", "expr": "x0 + x1 - 1"}],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert payload["fun"] == pytest.approx(0.5, abs=1e-8)


def test_optimize_minimize_inequality_constraint() -> None:
    """min x0^2 + x1^2 subject to x0 + x1 >= 1 has the known solution (0.5, 0.5)."""
    result = optimize_minimize(
        "x0**2 + x1**2",
        [2.0, 2.0],
        constraints=[{"kind": "ineq", "expr": "x0 + x1 - 1"}],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert payload["fun"] == pytest.approx(0.5, abs=1e-8)


def test_optimize_minimize_infeasible_constraints_error() -> None:
    """x0 + x1 = 1 and x0 + x1 = 2 cannot hold simultaneously."""
    result = optimize_minimize(
        "x0**2 + x1**2",
        [0.0, 0.0],
        constraints=[
            {"kind": "eq", "expr": "x0 + x1 - 1"},
            {"kind": "eq", "expr": "x0 + x1 - 2"},
        ],
    )

    assert result["ok"] is False
    assert "error" in result


def test_optimize_minimize_bounds_and_constraints_together() -> None:
    """x0 >= 0 bounds and x0 + x1 = 1: the unconstrained optimum (0.5, 0.5) is feasible."""
    result = optimize_minimize(
        "x0**2 + x1**2",
        [1.0, 1.0],
        bounds=[[0.0, None], [None, None]],
        constraints=[{"kind": "eq", "expr": "x0 + x1 - 1"}],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    x0, x1 = payload["x"]
    assert x0 == pytest.approx(0.5, abs=1e-6)
    assert x1 == pytest.approx(0.5, abs=1e-6)
    assert x0 >= -1e-8
    assert x0 + x1 == pytest.approx(1.0, abs=1e-8)


def test_optimize_minimize_constraint_uses_params() -> None:
    """Constraint x0 - a = 0 with a=2 pins x0 to 2."""
    result = optimize_minimize(
        "x0**2 + x1**2",
        [0.0, 0.0],
        params={"a": 2.0},
        constraints=[{"kind": "eq", "expr": "x0 - a"}],
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"][0] == pytest.approx(2.0, abs=1e-6)
    assert payload["x"][1] == pytest.approx(0.0, abs=1e-6)


def test_optimize_minimize_unknown_constraint_kind_error() -> None:
    result = optimize_minimize(
        "x0**2",
        [0.0],
        constraints=[{"kind": "ge", "expr": "x0"}],
    )

    assert result["ok"] is False
    assert "error" in result
    assert "unknown constraint kind" in result["error"]


def test_optimize_minimize_invalid_constraint_expression_error() -> None:
    result = optimize_minimize(
        "x0**2",
        [0.0],
        constraints=[{"kind": "eq", "expr": "x0 +"}],
    )

    assert result["ok"] is False
    assert "error" in result


def test_optimize_minimize_without_constraints_still_works() -> None:
    """The pre-constraint call signature/behaviour is unchanged."""
    result = optimize_minimize("(x0 - 3)**2", [0.0])

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["x"][0] == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# FFT: degenerate spectra have no dominant frequency
# ---------------------------------------------------------------------------


def test_fft_all_zero_signal_has_no_dominant_frequency() -> None:
    """An all-zero signal has only zero amplitudes, so no dominant bin exists."""
    result = fft_analysis([0.0, 0.0, 0.0, 0.0])

    assert result["ok"] is True
    payload = result["result"]
    assert payload["dominant_frequency"] is None
    assert "reason" in payload
    assert "no dominant frequency" in payload["reason"]
    assert payload["amplitudes"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)


def test_fft_one_hz_regression_after_no_dominant_change() -> None:
    """Regression: a 1 Hz sine still reports 1 Hz (and no ``reason`` key)."""
    sample_rate = 16.0
    t = np.arange(0.0, 1.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * math.pi * 1.0 * t)

    result = fft_analysis(signal.tolist(), sample_rate=sample_rate)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["dominant_frequency"] == pytest.approx(1.0, abs=0.1)
    assert "reason" not in payload


# ---------------------------------------------------------------------------
# File references: "@<path>" array parameters
# ---------------------------------------------------------------------------


def _write_column(path, values) -> None:
    """Write a numeric one-column CSV using only the standard library."""
    path.write_text("".join(f"{float(value):.17g}\n" for value in values), encoding="utf-8")


def test_fft_analysis_reads_at_file_csv(tmp_path, monkeypatch) -> None:
    """``@sine.csv`` is resolved relative to SCI_MCP_OUTPUT_DIR and analysed."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 8.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    _write_column(tmp_path / "sine.csv", np.sin(2.0 * math.pi * 1.0 * t))

    result = fft_analysis("@sine.csv", sample_rate=sample_rate)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["dominant_frequency"] == pytest.approx(1.0, abs=0.1)
    assert len(payload["frequencies"]) == int(t.size // 2 + 1)


def test_interpolate_1d_reads_at_file_csv(tmp_path, monkeypatch) -> None:
    """x, y and xi can all be CSV file references; y = x**2 is exact."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    _write_column(tmp_path / "x.csv", [0.0, 1.0, 2.0])
    _write_column(tmp_path / "y.csv", [0.0, 1.0, 4.0])
    _write_column(tmp_path / "xi.csv", [0.5, 1.5])

    result = interpolate_1d("@x.csv", "@y.csv", "@xi.csv", kind="quadratic")

    assert result["ok"] is True
    payload = result["result"]
    assert payload["xi"] == pytest.approx([0.5, 1.5])
    assert payload["yi"] == pytest.approx([0.25, 2.25], abs=1e-10)


def test_matrix_compute_reads_at_file_csv(tmp_path, monkeypatch) -> None:
    """A two-dimensional CSV is loaded for the ``a`` matrix argument."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "a.csv").write_text("1,2\n3,4\n", encoding="utf-8")

    result = matrix_compute("det", "@a.csv")

    assert result["ok"] is True
    assert result["result"] == pytest.approx(-2.0, abs=1e-12)


def test_matrix_compute_reads_at_file_for_rhs(tmp_path, monkeypatch) -> None:
    """The optional ``b`` argument also accepts a file reference."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "a.csv").write_text("3,1\n1,2\n", encoding="utf-8")
    _write_column(tmp_path / "b.csv", [9.0, 8.0])

    result = matrix_compute("solve", "@a.csv", "@b.csv")

    assert result["ok"] is True
    assert result["result"] == pytest.approx([2.0, 3.0], abs=1e-12)


def test_at_file_missing_returns_error(tmp_path, monkeypatch) -> None:
    """A missing @file reference fails cleanly with the operation name in the error."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))

    result = fft_analysis("@does_not_exist.csv", sample_rate=2.0)

    assert result["ok"] is False
    assert "error" in result
    assert "fft_analysis" in result["error"]


# ---------------------------------------------------------------------------
# Global optimization
# ---------------------------------------------------------------------------


def test_global_optimize_de_finds_multimodal_minimum() -> None:
    """(x0**2 - 1)**2 + x1**2 has two global minima at (+-1, 0) with fun = 0."""
    result = global_optimize(
        "(x0**2 - 1)**2 + x1**2",
        bounds=[[-3.0, 3.0], [-3.0, 3.0]],
        method="differential_evolution",
        seed=42,
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["method"] == "differential_evolution"
    assert payload["fun"] < 0.01
    assert abs(abs(payload["x"][0]) - 1.0) < 0.1
    assert abs(payload["x"][1]) < 0.1


def test_global_optimize_de_rastrigin_2d() -> None:
    """The 2-D Rastrigin function has global minimum 0 at the origin."""
    result = global_optimize(
        "20 + x0**2 + x1**2 - 10*(cos(2*pi*x0) + cos(2*pi*x1))",
        bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        method="differential_evolution",
        maxiter=2000,
        seed=42,
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["fun"] < 0.5
    assert np.asarray(payload["x"]) == pytest.approx([0.0, 0.0], abs=1e-2)


def test_global_optimize_dual_annealing() -> None:
    """dual_annealing minimizes a shifted quadratic on the same bounds."""
    result = global_optimize(
        "(x0 - 1.5)**2 + (x1 + 2.5)**2",
        bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        method="dual_annealing",
        maxiter=500,
        seed=42,
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["method"] == "dual_annealing"
    assert payload["fun"] < 0.01
    assert np.asarray(payload["x"]) == pytest.approx([1.5, -2.5], abs=0.1)


def test_global_optimize_basinhopping() -> None:
    """basinhopping needs a starting point and finds the quadratic minimum."""
    result = global_optimize(
        "(x0 - 3.0)**2 + (x1 - 4.0)**2",
        method="basinhopping",
        x0=[0.0, 0.0],
        maxiter=50,
        seed=42,
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert payload["method"] == "basinhopping"
    assert payload["fun"] < 1e-6
    assert np.asarray(payload["x"]) == pytest.approx([3.0, 4.0], abs=1e-4)


def test_global_optimize_params_are_injected() -> None:
    """(x0 - a)**2 + (x1 - b)**2 with a=2, b=3 is minimized at (2, 3)."""
    result = global_optimize(
        "(x0 - a)**2 + (x1 - b)**2",
        bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        method="differential_evolution",
        seed=42,
        params={"a": 2.0, "b": 3.0},
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["success"] is True
    assert np.asarray(payload["x"]) == pytest.approx([2.0, 3.0], abs=1e-2)
    assert payload["fun"] < 1e-4


def test_global_optimize_result_is_json_serializable() -> None:
    """Every result field is a JSON-native type (numpy values are converted)."""
    result = global_optimize(
        "(x0 - 1.0)**2",
        bounds=[[-2.0, 2.0]],
        method="differential_evolution",
        maxiter=100,
        seed=42,
    )

    assert result["ok"] is True
    encoded = json.dumps(result)
    decoded = json.loads(encoded)
    assert decoded["ok"] is True
    assert decoded["result"]["method"] == "differential_evolution"
    assert isinstance(decoded["result"]["fun"], float)
    assert isinstance(decoded["result"]["nfev"], int)
    assert isinstance(decoded["result"]["success"], bool)


def test_global_optimize_unknown_method_error() -> None:
    result = global_optimize(
        "x0**2",
        bounds=[[-1.0, 1.0]],
        method="simulated_annealing",
        seed=42,
    )

    assert result["ok"] is False
    assert "error" in result
    assert "unknown method" in result["error"]


def test_global_optimize_de_requires_bounds_error() -> None:
    result = global_optimize("x0**2", method="differential_evolution", seed=42)

    assert result["ok"] is False
    assert "error" in result
    assert "requires bounds" in result["error"]


def test_global_optimize_dual_annealing_requires_bounds_error() -> None:
    result = global_optimize("x0**2", method="dual_annealing", x0=[0.0], seed=42)

    assert result["ok"] is False
    assert "error" in result
    assert "requires bounds" in result["error"]


def test_global_optimize_basinhopping_requires_x0_error() -> None:
    result = global_optimize(
        "x0**2",
        bounds=[[-1.0, 1.0]],
        method="basinhopping",
        seed=42,
    )

    assert result["ok"] is False
    assert "error" in result
    assert "requires x0" in result["error"]


def test_global_optimize_invalid_bounds_shape_error() -> None:
    result = global_optimize(
        "x0**2 + x1**2",
        bounds=[[-1.0, 1.0], [-1.0]],
        method="differential_evolution",
        seed=42,
    )

    assert result["ok"] is False
    assert "error" in result
    assert "bounds" in result["error"]


def test_global_optimize_bounds_x0_dimension_mismatch_error() -> None:
    result = global_optimize(
        "x0**2 + x1**2",
        bounds=[[-1.0, 1.0]],
        method="basinhopping",
        x0=[0.0, 0.0],
        seed=42,
    )

    assert result["ok"] is False
    assert "error" in result
    assert "bounds" in result["error"]


def test_global_optimize_maxiter_out_of_range_error() -> None:
    low = global_optimize(
        "x0**2", bounds=[[-1.0, 1.0]], method="differential_evolution", maxiter=0, seed=42
    )
    high = global_optimize(
        "x0**2", bounds=[[-1.0, 1.0]], method="differential_evolution", maxiter=100001, seed=42
    )

    for result in (low, high):
        assert result["ok"] is False
        assert "error" in result
        assert "maxiter" in result["error"]
