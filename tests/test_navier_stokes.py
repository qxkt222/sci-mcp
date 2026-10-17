"""Tests for sci_mcp.tools.navier_stokes against independently known behaviour."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.navier_stokes import solve_navier_stokes_2d

TWO_PI = 2.0 * math.pi


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write every .npz produced by these tests into a temporary directory."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def _payload(result: dict) -> dict:
    assert result["ok"] is True, result.get("error")
    return result["result"]


def _from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    t = np.asarray(payload["t"], dtype=float)
    omega = np.asarray(payload["omega"], dtype=float)
    return x, y, t, omega


# ---------------------------------------------------------------------------
# 1. Taylor-Green vortex against the analytic solution
# ---------------------------------------------------------------------------


def test_taylor_green_matches_analytic_decay() -> None:
    """omega = -2 sin(x) sin(y) exp(-2 nu t) is an exact solution."""
    nu = 0.05
    result = solve_navier_stokes_2d(
        nu=nu,
        nx=32,
        ny=32,
        t_end=0.5,
        nt=200,
        initial_vorticity="-2*sin(x)*sin(y)",
    )
    payload = _payload(result)
    x, y, t, omega = _from_payload(payload)

    # Several time levels from the returned (downsampled) series.
    checked = 0
    for t_index in (0, 5, 10, len(t) - 1):
        time = float(t[t_index])
        analytic = (
            -2.0
            * np.sin(x)[None, :]
            * np.sin(y)[:, None]
            * np.exp(-2.0 * nu * time)
        )
        numerical = omega[t_index]
        scale = float(np.max(np.abs(analytic)))
        error = float(np.max(np.abs(numerical - analytic))) / scale
        assert error < 5e-2, f"relative error {error:.3e} at t={time}"
        checked += 1
    assert checked == 4


# ---------------------------------------------------------------------------
# 2. Energy decays monotonically for the Taylor-Green vortex
# ---------------------------------------------------------------------------


def test_energy_is_monotonically_decreasing() -> None:
    payload = _payload(
        solve_navier_stokes_2d(
            nu=0.05, nx=32, ny=32, t_end=0.5, nt=100,
            initial_vorticity="-2*sin(x)*sin(y)",
        )
    )
    energy = np.asarray(payload["energy"], dtype=float)
    assert energy.size == len(payload["t"])
    assert np.all(np.isfinite(energy))
    assert np.all(energy >= 0.0)
    increments = np.diff(energy)
    assert np.all(increments <= 1e-12 * max(energy[0], 1.0))
    assert energy[-1] < energy[0]


# ---------------------------------------------------------------------------
# 3. Larger viscosity decays faster
# ---------------------------------------------------------------------------


def test_larger_viscosity_decays_faster() -> None:
    common = dict(nx=32, ny=32, t_end=2.0, nt=800,
                  initial_vorticity="-2*sin(x)*sin(y)")
    low = _payload(solve_navier_stokes_2d(nu=0.01, **common))
    high = _payload(solve_navier_stokes_2d(nu=0.05, **common))
    low_final = float(np.max(np.abs(np.asarray(low["omega"], dtype=float)[-1])))
    high_final = float(np.max(np.abs(np.asarray(high["omega"], dtype=float)[-1])))
    assert high_final < low_final
    # Both agree with the analytic amplitude 2*exp(-2 nu t).
    assert low_final == pytest.approx(2.0 * np.exp(-2.0 * 0.01 * 2.0), rel=5e-2)
    assert high_final == pytest.approx(2.0 * np.exp(-2.0 * 0.05 * 2.0), rel=5e-2)


# ---------------------------------------------------------------------------
# 4. Stability failure is reported as err
# ---------------------------------------------------------------------------


def test_stability_violation_returns_err() -> None:
    result = solve_navier_stokes_2d(
        nu=2.0, nx=8, ny=8, t_end=1.0, nt=1,
        initial_vorticity="-2*sin(x)*sin(y)",
    )
    assert result["ok"] is False
    assert "unstable" in result["error"].lower()
    assert "nt" in result["error"]


# ---------------------------------------------------------------------------
# 5. Grid limits
# ---------------------------------------------------------------------------


def test_grid_limits_are_enforced() -> None:
    too_small = solve_navier_stokes_2d(nu=0.01, nx=4, ny=8)
    assert too_small["ok"] is False
    assert "nx" in too_small["error"]

    too_large = solve_navier_stokes_2d(nu=0.01, nx=300, ny=8)
    assert too_large["ok"] is False
    assert "nx" in too_large["error"]

    # 257 x 256 exceeds nx*ny even if each axis were individually in range.
    too_many = solve_navier_stokes_2d(nu=0.01, nx=256, ny=257)
    assert too_many["ok"] is False


# ---------------------------------------------------------------------------
# 6. Invalid initial vorticity expressions
# ---------------------------------------------------------------------------


def test_invalid_initial_expression_returns_err() -> None:
    unknown = solve_navier_stokes_2d(nu=0.01, nx=16, ny=16, initial_vorticity="z + 1")
    assert unknown["ok"] is False
    assert "unknown symbol" in unknown["error"]

    # Undefined functions are rejected when the expression is evaluated.
    undefined = solve_navier_stokes_2d(
        nu=0.01, nx=16, ny=16, initial_vorticity="foo(x, y)"
    )
    assert undefined["ok"] is False
    assert "initial_vorticity" in undefined["error"]

    syntax = solve_navier_stokes_2d(nu=0.01, nx=16, ny=16, initial_vorticity="1 +* 2")
    assert syntax["ok"] is False

    non_finite = solve_navier_stokes_2d(
        nu=0.01, nx=16, ny=16, t_end=0.1, nt=10,
        initial_vorticity="sqrt(x - 100)",
    )
    assert non_finite["ok"] is False


# ---------------------------------------------------------------------------
# 7. npz contents and reported shapes
# ---------------------------------------------------------------------------


def test_npz_file_and_shapes() -> None:
    payload = _payload(
        solve_navier_stokes_2d(
            nu=0.02, nx=16, ny=16, t_end=0.1, nt=20,
            initial_vorticity="-2*sin(x)*sin(y)",
        )
    )
    path = Path(payload["path"])
    assert path.exists()
    assert path.suffix == ".npz"

    with np.load(path) as data:
        assert set(("x", "y", "t", "omega")).issubset(data.files)
        assert data["omega"].shape == (21, 16, 16)
        assert data["x"].shape == (16,)
        assert data["y"].shape == (16,)
        assert data["t"].shape == (21,)

    # The returned field is downsampled to at most 20 time levels.
    assert payload["original_shape"] == [21, 16, 16]
    assert payload["shape"] == [20, 16, 16]
    assert payload["downsampled"] is True
    assert len(payload["t"]) == 20
    assert len(payload["energy"]) == 20
    assert len(payload["enstrophy"]) == 20


# ---------------------------------------------------------------------------
# 8. Model / limitations metadata
# ---------------------------------------------------------------------------


def test_model_and_limitations_are_reported() -> None:
    payload = _payload(solve_navier_stokes_2d(nu=0.02, nx=8, ny=8, t_end=0.1, nt=10))
    assert payload["model"] == "2D incompressible vorticity-streamfunction, periodic"
    assert payload["limitations"] == (
        "not a general CFD solver; no 3D, turbulence models, or complex boundaries"
    )


# ---------------------------------------------------------------------------
# 9. JSON serialisability with strict NaN/Infinity rejection
# ---------------------------------------------------------------------------


def test_result_is_strictly_json_serialisable() -> None:
    result = solve_navier_stokes_2d(
        nu=0.05, nx=16, ny=16, t_end=0.2, nt=40,
        initial_vorticity="-2*sin(x)*sin(y)",
    )
    text = json.dumps(result, allow_nan=False)
    round_trip = json.loads(text)
    assert round_trip["ok"] is True


# ---------------------------------------------------------------------------
# 10. Zero initial vorticity stays identically zero
# ---------------------------------------------------------------------------


def test_zero_initial_vorticity_stays_zero() -> None:
    payload = _payload(
        solve_navier_stokes_2d(nu=0.1, nx=16, ny=16, t_end=0.5, nt=50,
                               initial_vorticity="0")
    )
    omega = np.asarray(payload["omega"], dtype=float)
    assert np.all(omega == 0.0)
    assert payload["max"] == 0.0
    assert payload["min"] == 0.0
    assert np.all(np.asarray(payload["energy"], dtype=float) == 0.0)
    assert np.all(np.asarray(payload["enstrophy"], dtype=float) == 0.0)
