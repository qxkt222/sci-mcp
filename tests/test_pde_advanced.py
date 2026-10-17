"""Tests for sci_mcp.tools.pde_advanced against independently known behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.pde_advanced import (
    solve_burgers_1d,
    solve_heat_3d,
    solve_laplace_3d,
)


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write every .npz produced by these tests into a temporary directory."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def _load(payload: dict) -> dict:
    """Load the saved full grid referenced by a tool payload."""
    return np.load(payload["path"])


# ---------------------------------------------------------------------------
# 3D heat equation
# ---------------------------------------------------------------------------


def test_heat3d_matches_analytic_solution() -> None:
    """u_t = Laplacian(u) with sin(pi x)sin(pi y)sin(pi z) data decays as exp(-3 pi^2 t)."""
    result = solve_heat_3d(
        alpha=1.0,
        nx=12,
        ny=12,
        nz=12,
        t_end=0.01,
        nt=40,
        initial="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    z = np.asarray(payload["z"], dtype=float)
    t = np.asarray(payload["t"], dtype=float)
    u = np.asarray(payload["u"], dtype=float)
    assert u.shape == (
        len(payload["t"]),
        len(payload["z"]),
        len(payload["y"]),
        len(payload["x"]),
    )
    assert t[-1] == pytest.approx(0.01)

    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")
    exact = np.exp(-3.0 * np.pi**2 * t[-1]) * (
        np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y) * np.sin(np.pi * grid_z)
    )
    error = float(np.max(np.abs(u[-1] - exact)))
    assert error < 1e-2


def test_heat3d_rejects_unstable_time_step() -> None:
    """r = alpha*dt/min(dx,dy,dz)^2 > 1/6 must be rejected with an nt hint."""
    result = solve_heat_3d(
        alpha=1.0,
        nx=12,
        ny=12,
        nz=12,
        t_end=0.1,
        nt=10,
        initial="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is False
    assert "error" in result
    message = result["error"].lower()
    assert "unstable" in message or "stability" in message
    assert "nt" in message


def test_heat3d_rejects_oversized_grid() -> None:
    """nx*ny*nz above 500,000 must be rejected before allocation."""
    result = solve_heat_3d(
        alpha=1.0,
        nx=100,
        ny=100,
        nz=60,
        t_end=0.001,
        nt=2,
        initial="1",
    )
    assert result["ok"] is False
    assert "grid too large" in result["error"].lower()


def test_heat3d_saves_full_grid_npz() -> None:
    """The complete (t, z, y, x) history must be persisted and shapes reported."""
    result = solve_heat_3d(
        alpha=0.5,
        nx=8,
        ny=8,
        nz=8,
        t_end=0.01,
        nt=10,
        initial="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    path = Path(payload["path"])
    assert path.exists()
    u = np.asarray(payload["u"], dtype=float)
    assert payload["shape"] == list(u.shape)
    assert payload["original_shape"] == [11, 9, 9, 9]
    assert payload["shape"] == payload["original_shape"]
    assert payload["downsampled"] is False

    with _load(payload) as data:
        assert set(data.files) == {"x", "y", "z", "t", "u"}
        assert data["x"].shape == (9,)
        assert data["y"].shape == (9,)
        assert data["z"].shape == (9,)
        assert data["t"].shape == (11,)
        assert data["u"].shape == (11, 9, 9, 9)


# ---------------------------------------------------------------------------
# 3D Laplace equation
# ---------------------------------------------------------------------------


def test_laplace3d_one_face_at_100() -> None:
    """Single face at 100 V: the centre potential is V/6 and Jacobi converges."""
    result = solve_laplace_3d(15, 15, 15, z1=100.0)
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["converged"] is True
    assert payload["iterations"] > 0
    centre = payload["center"]
    assert 15.0 <= centre <= 35.0
    # The cube-centre potential for one face at V and the other five
    # grounded is V/6; verify against that independently derived value.
    assert centre == pytest.approx(100.0 / 6.0, abs=1.0)


def test_laplace3d_zero_boundaries_give_zero_field() -> None:
    """With all six faces at zero the unique solution is identically zero."""
    result = solve_laplace_3d(10, 10, 10)
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["converged"] is True
    assert payload["max"] == pytest.approx(0.0, abs=1e-12)
    assert payload["min"] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.asarray(payload["u"], dtype=float), 0.0)


def test_laplace3d_rejects_oversized_grid() -> None:
    """nx*ny*nz above 500,000 must be rejected."""
    result = solve_laplace_3d(100, 100, 60)
    assert result["ok"] is False
    assert "grid too large" in result["error"].lower()


# ---------------------------------------------------------------------------
# 1D Burgers equation
# ---------------------------------------------------------------------------


def test_burgers_energy_decreases_and_boundaries_vanish() -> None:
    """Viscous Burgers flow must stay finite with decaying energy and zero ends."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=100, t_end=0.5, nt=400, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    dx = float(payload["dx"])
    with _load(payload) as data:
        u = np.asarray(data["u"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u[:, 0])) < 1e-12
    assert np.max(np.abs(u[:, -1])) < 1e-12

    energy = np.sum(u * u, axis=1) * dx
    assert np.all(np.diff(energy) <= 1e-12)
    assert energy[-1] < energy[0]
    assert x[-1] == pytest.approx(1.0)


def test_burgers_large_diffusion_smooths_peak() -> None:
    """A large viscosity must reduce the peak below the initial unit amplitude."""
    result = solve_burgers_1d(
        nu=0.5, length=1.0, nx=100, t_end=0.05, nt=1000, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    with _load(payload) as data:
        u = np.asarray(data["u"], dtype=float)
    assert np.all(np.isfinite(u))
    initial_peak = float(np.max(np.abs(u[0])))
    final_peak = float(np.max(np.abs(u[-1])))
    assert initial_peak == pytest.approx(1.0, abs=1e-3)
    assert final_peak < 0.9 * initial_peak


def test_burgers_rejects_unstable_time_step() -> None:
    """dt above dx^2/(2 nu) must be rejected with an nt hint."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=100, t_end=1.0, nt=10, initial="sin(pi*x)"
    )
    assert result["ok"] is False
    assert "error" in result
    message = result["error"].lower()
    assert "unstable" in message or "stability" in message
    assert "nt" in message


def test_burgers_saves_full_grid_npz() -> None:
    """The complete (t, x) history must be persisted with correct shapes."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=50, t_end=0.01, nt=50, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert Path(payload["path"]).exists()
    u = np.asarray(payload["u"], dtype=float)
    assert payload["shape"] == list(u.shape)
    assert payload["shape"] == [20, 20]
    assert payload["original_shape"] == [51, 51]
    assert payload["downsampled"] is True

    with _load(payload) as data:
        assert data["x"].shape == (51,)
        assert data["t"].shape == (51,)
        assert data["u"].shape == (51, 51)


def test_burgers_rejects_invalid_initial_expression() -> None:
    """An unparsable initial expression must surface as an error, not an exception."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=50, t_end=0.01, nt=10, initial="sin(pi*x"
    )
    assert result["ok"] is False
    assert "initial" in result["error"].lower()


def test_burgers_rejects_unknown_symbol_in_initial() -> None:
    """Free symbols other than x must be rejected explicitly."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=50, t_end=0.01, nt=10, initial="sin(pi*q)"
    )
    assert result["ok"] is False
    assert "unknown symbol" in result["error"].lower()


def test_burgers_rejects_too_few_points() -> None:
    """nx below the supported minimum must return an error."""
    result = solve_burgers_1d(
        nu=0.01, length=1.0, nx=5, t_end=0.01, nt=10, initial="sin(pi*x)"
    )
    assert result["ok"] is False
    assert "nx" in result["error"].lower()


def test_results_are_strict_json_serialisable() -> None:
    """Envelopes must contain no NaN/Infinity literals."""
    heat = solve_heat_3d(
        alpha=1.0, nx=6, ny=6, nz=6, t_end=0.005, nt=5, initial="sin(pi*x)*y*z"
    )
    laplace = solve_laplace_3d(6, 6, 6, z1=100.0, tol=1e-5)
    burgers = solve_burgers_1d(
        nu=0.05, length=1.0, nx=40, t_end=0.01, nt=20, initial="sin(pi*x)"
    )
    for result in (heat, laplace, burgers):
        assert result["ok"] is True, result.get("error")
        encoded = json.dumps(result, allow_nan=False)
        assert encoded
