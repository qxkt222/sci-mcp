"""Tests for sci_mcp.tools.pde_more against independently known behaviour."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.pde import solve_heat_1d
from sci_mcp.tools.pde_more import (
    solve_heat_1d_variable,
    solve_pde_1d,
    solve_wave_3d,
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
# Variable-coefficient heat equation
# ---------------------------------------------------------------------------


def test_variable_constant_alpha_matches_constant_coefficient_solver() -> None:
    """alpha(x) = 1 must reproduce solve_heat_1d on the same grid."""
    common = dict(
        length=1.0,
        nx=50,
        t_end=0.02,
        nt=200,
        initial="sin(pi*x)",
        left=0.0,
        right=0.0,
    )
    reference = solve_heat_1d(alpha=1.0, **common)
    variable = solve_heat_1d_variable(alpha_expr="1.0", **common)
    assert reference["ok"] is True, reference.get("error")
    assert variable["ok"] is True, variable.get("error")

    reference_u = _load(reference["result"])["u"]
    variable_u = _load(variable["result"])["u"]
    assert reference_u.shape == variable_u.shape
    assert float(np.max(np.abs(reference_u - variable_u))) < 1e-8


def test_variable_linear_alpha_is_finite_keeps_boundaries_and_decays() -> None:
    """alpha = 1 + 0.5x must stay finite, respect Dirichlet ends and decay."""
    result = solve_heat_1d_variable(
        alpha_expr="1 + 0.5*x",
        length=1.0,
        nx=40,
        t_end=0.02,
        nt=200,
        initial="sin(pi*x)",
        left=0.0,
        right=0.0,
    )
    assert result["ok"] is True, result.get("error")
    payload = result["result"]
    u = _load(payload)["u"]

    assert np.all(np.isfinite(u))
    assert np.allclose(u[:, 0], 0.0)
    assert np.allclose(u[:, -1], 0.0)
    assert float(np.max(u[-1])) < float(np.max(u[0]))
    assert float(np.max(u[0])) == pytest.approx(1.0, abs=1e-12)
    assert payload["alpha_max"] == pytest.approx(1.5, rel=1e-12)


def test_variable_negative_alpha_is_rejected() -> None:
    """A non-positive conductivity cannot be used by the flux scheme."""
    result = solve_heat_1d_variable(
        alpha_expr="-1.0",
        length=1.0,
        nx=20,
        t_end=0.01,
        nt=50,
        initial="sin(pi*x)",
    )
    assert result["ok"] is False
    assert "positive" in result["error"]


def test_variable_explicit_stability_is_enforced() -> None:
    """dt*max(alpha)/dx^2 > 0.5 must be reported as unstable."""
    result = solve_heat_1d_variable(
        alpha_expr="1.0",
        length=1.0,
        nx=10,
        t_end=1.0,
        nt=10,
    )
    assert result["ok"] is False
    assert "unstable" in result["error"]


# ---------------------------------------------------------------------------
# Semi-general 1D PDE
# ---------------------------------------------------------------------------


def test_semi_general_heat_matches_analytic_solution() -> None:
    """F = u_xx must reproduce exp(-pi^2 t) sin(pi x)."""
    result = solve_pde_1d(
        equation="1.0*u_xx",
        length=1.0,
        nx=40,
        t_end=0.02,
        nt=200,
        initial="sin(pi*x)",
        left=0.0,
        right=0.0,
    )
    assert result["ok"] is True, result.get("error")

    grid = _load(result["result"])
    x = grid["x"]
    t = grid["t"]
    expected = np.exp(-(math.pi**2) * t)[:, None] * np.sin(math.pi * x)[None, :]
    assert float(np.max(np.abs(grid["u"] - expected))) < 2e-3
    assert result["result"]["stability_note"].startswith("explicit scheme")
    assert "not guaranteed" in result["result"]["stability_note"]


def test_semi_general_nonlinear_reaction_diffusion_runs_stably() -> None:
    """F = u - u^3 + 0.1*u_xx with small dt must stay finite and report the note."""
    result = solve_pde_1d(
        equation="u - u**3 + 0.1*u_xx",
        length=1.0,
        nx=40,
        t_end=0.1,
        nt=400,
        initial="0.1*sin(pi*x)",
        left=0.0,
        right=0.0,
    )
    assert result["ok"] is True, result.get("error")
    payload = result["result"]
    u = _load(payload)["u"]
    assert np.all(np.isfinite(u))
    assert float(np.max(np.abs(u))) < 0.5
    assert payload["stability_note"] == (
        "explicit scheme; stability is not guaranteed for arbitrary F. "
        "Reduce dt/nx or use a specialized solver for stiff/complex problems."
    )


def test_semi_general_invalid_expression_is_rejected() -> None:
    """Unknown symbols and malformed expressions must return err, not raise."""
    unknown = solve_pde_1d(equation="u_xxx + 1", nx=20, nt=10)
    assert unknown["ok"] is False
    assert "unknown symbol" in unknown["error"]

    malformed = solve_pde_1d(equation="1 +* 2", nx=20, nt=10)
    assert malformed["ok"] is False


def test_semi_general_blowup_is_detected_early() -> None:
    """F = 1000*u with a large dt overflows; the solver must stop with err."""
    result = solve_pde_1d(
        equation="1000*u",
        length=1.0,
        nx=10,
        t_end=10.0,
        nt=1000,
        initial="1.0",
        left=0.0,
        right=0.0,
    )
    assert result["ok"] is False
    assert "non-finite" in result["error"]
    assert 1 <= result["step"] <= 1000


# ---------------------------------------------------------------------------
# 3D wave equation
# ---------------------------------------------------------------------------


def test_wave3d_matches_modal_solution() -> None:
    """The separable mode sin(pi x)sin(pi y)sin(pi z)cos(c*pi*sqrt(3)*t)."""
    c = 1.0
    nx = ny = nz = 12
    t_end = 0.2
    result = solve_wave_3d(
        c=c,
        nx=nx,
        ny=ny,
        nz=nz,
        t_end=t_end,
        nt=20,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
        initial_velocity="0",
        boundary=0.0,
    )
    assert result["ok"] is True, result.get("error")

    grid = _load(result["result"])
    x, y, z = grid["x"], grid["y"], grid["z"]
    expected = (
        np.sin(math.pi * x)[None, None, :]
        * np.sin(math.pi * y)[None, :, None]
        * np.sin(math.pi * z)[:, None, None]
        * math.cos(c * math.pi * math.sqrt(3.0) * t_end)
    )
    final = grid["u"][-1]
    assert final.shape == expected.shape
    assert float(np.max(np.abs(final - expected))) < 2e-2

    initial = grid["u"][0]
    assert np.allclose(initial, expected / math.cos(c * math.pi * math.sqrt(3.0) * t_end), atol=1e-12)


def test_wave3d_cfl_violation_is_rejected() -> None:
    """c*dt*sqrt(3)/min(dx,dy,dz) > 1 must return err."""
    result = solve_wave_3d(
        c=2.0,
        nx=8,
        ny=8,
        nz=8,
        t_end=0.5,
        nt=10,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is False
    assert "CFL" in result["error"]


def test_wave3d_grid_limit_is_enforced() -> None:
    """nx*ny*nz above 500,000 is rejected before allocation."""
    result = solve_wave_3d(
        c=1.0,
        nx=100,
        ny=100,
        nz=100,
        t_end=0.01,
        nt=2,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is False
    assert "grid too large" in result["error"]


def test_wave3d_storage_guard_is_enforced() -> None:
    """nx*ny*nz*(nt+1) above 20,000,000 is rejected before allocation."""
    result = solve_wave_3d(
        c=1.0,
        nx=10,
        ny=10,
        nz=10,
        t_end=1.0,
        nt=30_000,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert result["ok"] is False
    assert "stored points" in result["error"]


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------


def test_npz_contents_and_shapes_for_all_three_tools() -> None:
    """Every tool must save the full grid and report matching shapes."""
    variable = solve_heat_1d_variable(
        alpha_expr="1 + 0.5*x",
        length=1.0,
        nx=20,
        t_end=0.005,
        nt=20,
    )
    assert variable["ok"] is True, variable.get("error")
    variable_payload = variable["result"]
    variable_npz = _load(variable_payload)
    assert set(variable_npz.files) == {"x", "t", "u"}
    assert variable_npz["u"].shape == tuple(variable_payload["original_shape"])
    assert variable_payload["shape"] == [21, 21]
    assert variable_payload["downsampled"] is False
    assert Path(variable_payload["path"]).is_file()

    semi = solve_pde_1d(
        equation="1.0*u_xx",
        length=1.0,
        nx=20,
        t_end=0.02,
        nt=20,
    )
    assert semi["ok"] is True, semi.get("error")
    semi_payload = semi["result"]
    semi_npz = _load(semi_payload)
    assert set(semi_npz.files) == {"x", "t", "u"}
    assert semi_npz["u"].shape == tuple(semi_payload["original_shape"])
    assert semi_payload["shape"] == [21, 21]
    assert semi_payload["downsampled"] is False

    wave = solve_wave_3d(
        c=1.0,
        nx=6,
        ny=6,
        nz=6,
        t_end=0.1,
        nt=4,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert wave["ok"] is True, wave.get("error")
    wave_payload = wave["result"]
    wave_npz = _load(wave_payload)
    assert set(wave_npz.files) == {"x", "y", "z", "t", "u"}
    assert wave_npz["u"].shape == tuple(wave_payload["original_shape"])
    assert wave_payload["shape"] == [5, 7, 7, 7]
    assert wave_payload["downsampled"] is False


def test_downsampled_output_is_capped() -> None:
    """Returned arrays respect the 100x50 (1D) and 20^3x20 (3D) caps."""
    variable = solve_heat_1d_variable(
        alpha_expr="1.0",
        length=1.0,
        nx=150,
        t_end=0.001,
        nt=60,
    )
    assert variable["ok"] is True, variable.get("error")
    payload = variable["result"]
    assert payload["original_shape"] == [61, 151]
    assert payload["shape"] == [50, 100]
    assert payload["downsampled"] is True
    assert len(payload["x"]) == 100
    assert len(payload["t"]) == 50

    wave = solve_wave_3d(
        c=1.0,
        nx=22,
        ny=22,
        nz=22,
        t_end=0.01,
        nt=20,
        initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
    )
    assert wave["ok"] is True, wave.get("error")
    wave_payload = wave["result"]
    assert wave_payload["original_shape"] == [21, 23, 23, 23]
    assert wave_payload["shape"] == [20, 20, 20, 20]
    assert wave_payload["downsampled"] is True


def test_results_are_strict_json_serializable() -> None:
    """Successful and failing envelopes must survive allow_nan=False."""
    results = [
        solve_heat_1d_variable(alpha_expr="1 + 0.5*x", nx=20, nt=20, t_end=0.02),
        solve_pde_1d(equation="u - u**3 + 0.1*u_xx", nx=20, nt=20, t_end=0.02),
        solve_wave_3d(
            c=1.0,
            nx=6,
            ny=6,
            nz=6,
            t_end=0.1,
            nt=4,
            initial_displacement="sin(pi*x)*sin(pi*y)*sin(pi*z)",
        ),
        solve_heat_1d_variable(alpha_expr="-1.0", nx=20, nt=20),
        solve_pde_1d(equation="bad expression +*", nx=20, nt=20),
        solve_wave_3d(c=1.0, nx=100, ny=100, nz=100, t_end=0.01, nt=2,
                      initial_displacement="0"),
    ]
    for result in results:
        json.dumps(result, allow_nan=False)
