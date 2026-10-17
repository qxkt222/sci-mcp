"""Tests for sci_mcp.tools.pde against independently known analytic solutions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.pde import solve_heat_1d, solve_laplace_2d, solve_wave_1d


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write every .npz produced by these tests into a temporary directory."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Heat equation
# ---------------------------------------------------------------------------


def test_heat_matches_analytic_solution() -> None:
    """u_t = u_xx with u(0,t)=u(1,t)=0 and sin(pi*x) initial data.

    The separable solution is u = exp(-pi^2 t) sin(pi x); FTCS should match it
    to grid accuracy at t = 0.1.
    """
    result = solve_heat_1d(
        alpha=1.0, length=1.0, nx=50, t_end=0.1, nt=1000, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    x = np.asarray(payload["x"], dtype=float)
    t = np.asarray(payload["t"], dtype=float)
    u = np.asarray(payload["u"], dtype=float)
    assert t[-1] == pytest.approx(0.1)

    exact = np.exp(-np.pi**2 * 0.1) * np.sin(np.pi * x)
    coarse_error = float(np.max(np.abs(u[-1] - exact)))
    assert coarse_error < 2e-3

    # Refining the grid must reduce the error: verifies a genuine converging
    # discretisation rather than a coincidentally tuned one.
    refined = solve_heat_1d(
        alpha=1.0, length=1.0, nx=80, t_end=0.1, nt=2000, initial="sin(pi*x)"
    )
    assert refined["ok"] is True, refined.get("error")
    refined_payload = refined["result"]
    refined_x = np.asarray(refined_payload["x"], dtype=float)
    refined_u = np.asarray(refined_payload["u"], dtype=float)
    refined_exact = np.exp(-np.pi**2 * 0.1) * np.sin(np.pi * refined_x)
    refined_error = float(np.max(np.abs(refined_u[-1] - refined_exact)))
    assert refined_error < coarse_error


def test_heat_rejects_unstable_time_step() -> None:
    """r = alpha*dt/dx^2 > 0.5 must be rejected with a stability hint."""
    result = solve_heat_1d(
        alpha=1.0, length=1.0, nx=50, t_end=0.1, nt=10, initial="sin(pi*x)"
    )
    assert result["ok"] is False
    assert "error" in result
    message = result["error"].lower()
    assert "unstable" in message or "stability" in message
    assert "nt" in message


def test_heat_saves_full_grid_npz() -> None:
    """The complete (not downsampled) grid must be persisted to an .npz file."""
    result = solve_heat_1d(
        alpha=0.5, length=1.0, nx=20, t_end=0.01, nt=20, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    path = Path(payload["path"])
    assert path.exists()
    # No downsampling here, so the reported shape must match both the returned
    # array and the full computation grid.
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [21, 21]
    assert payload["shape"] == payload["original_shape"]
    assert payload["x_len"] == 21
    assert payload["t_len"] == 21
    assert payload["downsampled"] is False

    with np.load(path) as data:
        assert set(data.files) == {"x", "t", "u"}
        assert data["x"].shape == (21,)
        assert data["t"].shape == (21,)
        assert data["u"].shape == (21, 21)


def test_heat_downsampling_metadata() -> None:
    """Large grids are truncated to 100 space and 50 time points in the response."""
    result = solve_heat_1d(
        alpha=1.0, length=1.0, nx=300, t_end=0.001, nt=400, initial="sin(pi*x)"
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert len(payload["x"]) <= 100
    assert len(payload["t"]) <= 50
    assert np.asarray(payload["u"]).shape == (len(payload["t"]), len(payload["x"]))
    assert payload["downsampled"] is True
    assert payload["original_x_length"] == 301
    assert payload["original_t_length"] == 401
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [401, 301]
    assert payload["shape"] != payload["original_shape"]
    assert payload["x_len"] == len(payload["x"])
    assert payload["t_len"] == len(payload["t"])


# ---------------------------------------------------------------------------
# Wave equation
# ---------------------------------------------------------------------------


def test_wave_matches_analytic_solution() -> None:
    """u_tt = c^2 u_xx with u(x,0)=sin(pi*x), u_t(x,0)=0 has u=cos(pi t)sin(pi x)."""
    result = solve_wave_1d(
        c=1.0,
        length=1.0,
        nx=100,
        t_end=0.1,
        nt=20,
        initial_displacement="sin(pi*x)",
        initial_velocity="0",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    x = np.asarray(payload["x"], dtype=float)
    t = np.asarray(payload["t"], dtype=float)
    u = np.asarray(payload["u"], dtype=float)
    assert t[-1] == pytest.approx(0.1)

    exact = np.cos(np.pi * 0.1) * np.sin(np.pi * x)
    assert float(np.max(np.abs(u[-1] - exact))) < 2e-3


def test_wave_rejects_cfl_violation() -> None:
    """c*dt/dx > 1 must be rejected."""
    result = solve_wave_1d(
        c=2.0,
        length=1.0,
        nx=100,
        t_end=0.1,
        nt=10,
        initial_displacement="sin(pi*x)",
    )
    assert result["ok"] is False
    assert "error" in result
    assert "cfl" in result["error"].lower()


def test_wave_saves_full_grid_npz() -> None:
    """The full wave history (including the initial row) is saved."""
    result = solve_wave_1d(
        c=1.0,
        length=1.0,
        nx=20,
        t_end=0.05,
        nt=25,
        initial_displacement="sin(pi*x)",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    path = Path(payload["path"])
    assert path.exists()
    with np.load(path) as data:
        assert set(data.files) == {"x", "t", "u"}
        assert data["u"].shape == (26, 21)
        # Initial displacement is stored unchanged (apart from fixed ends).
        assert data["u"][0] == pytest.approx(np.sin(np.pi * data["x"]))


def test_wave_shape_fields_for_small_grid() -> None:
    """Without downsampling, shape must equal original_shape and the returned u."""
    result = solve_wave_1d(
        c=1.0,
        length=1.0,
        nx=20,
        t_end=0.05,
        nt=25,
        initial_displacement="sin(pi*x)",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [len(payload["t"]), len(payload["x"])]
    assert payload["shape"] == payload["original_shape"]
    assert payload["x_len"] == len(payload["x"])
    assert payload["t_len"] == len(payload["t"])
    assert payload["downsampled"] is False


def test_wave_shape_fields_when_downsampled() -> None:
    """A large grid reports the returned shape plus the full original shape."""
    result = solve_wave_1d(
        c=1.0,
        length=1.0,
        nx=300,
        t_end=0.001,
        nt=400,
        initial_displacement="sin(pi*x)",
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["downsampled"] is True
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [401, 301]
    assert payload["shape"] != payload["original_shape"]
    assert payload["x_len"] == len(payload["x"])
    assert payload["t_len"] == len(payload["t"])


# ---------------------------------------------------------------------------
# Laplace equation
# ---------------------------------------------------------------------------


def test_laplace_center_value() -> None:
    """Top edge at 100 and all other edges at 0 gives u(0.5,0.5) ~= 25."""
    result = solve_laplace_2d(
        nx=40, ny=40, left=0.0, right=0.0, top=100.0, bottom=0.0
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["converged"] is True
    assert payload["iterations"] > 0

    u = np.asarray(payload["u"], dtype=float)
    center = u[u.shape[0] // 2, u.shape[1] // 2]
    assert 20.0 <= center <= 30.0


def test_laplace_zero_boundaries_give_zero_solution() -> None:
    """With every Dirichlet value zero the discrete solution is identically zero."""
    result = solve_laplace_2d(
        nx=30, ny=20, left=0.0, right=0.0, top=0.0, bottom=0.0
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["converged"] is True
    assert float(np.max(np.abs(np.asarray(payload["u"], dtype=float)))) == 0.0
    assert payload["max"] == 0.0
    assert payload["min"] == 0.0


def test_laplace_shape_fields_for_small_grid() -> None:
    """Without downsampling, shape must equal original_shape and the returned u."""
    result = solve_laplace_2d(
        nx=30, ny=20, left=0.0, right=0.0, top=100.0, bottom=0.0
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [len(payload["y"]), len(payload["x"])]
    assert payload["shape"] == payload["original_shape"]
    assert payload["x_len"] == len(payload["x"])
    assert payload["y_len"] == len(payload["y"])
    assert payload["downsampled"] is False


def test_laplace_shape_fields_when_downsampled() -> None:
    """A large grid reports the returned shape plus the full original shape."""
    result = solve_laplace_2d(
        nx=150,
        ny=40,
        left=0.0,
        right=0.0,
        top=100.0,
        bottom=0.0,
        tol=1e-6,
        max_iter=20000,
    )
    assert result["ok"] is True, result.get("error")

    payload = result["result"]
    assert payload["downsampled"] is True
    assert payload["shape"] == [len(payload["u"]), len(payload["u"][0])]
    assert payload["original_shape"] == [41, 151]
    assert payload["shape"] != payload["original_shape"]
    assert payload["x_len"] == len(payload["x"])
    assert payload["y_len"] == len(payload["y"])


def test_laplace_reports_unconverged_after_max_iter() -> None:
    """Hitting max_iter returns ok with converged=False and the iteration count."""
    result = solve_laplace_2d(
        nx=10,
        ny=10,
        left=0.0,
        right=0.0,
        top=100.0,
        bottom=0.0,
        tol=1e-12,
        max_iter=1,
    )
    assert result["ok"] is True, result.get("error")
    assert result["result"]["converged"] is False
    assert result["result"]["iterations"] == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_heat_rejects_invalid_initial_expression() -> None:
    """A malformed sympy expression must return err instead of raising."""
    result = solve_heat_1d(
        alpha=1.0, length=1.0, nx=50, t_end=0.1, nt=1000, initial="sin("
    )
    assert result["ok"] is False
    assert "error" in result


def test_heat_rejects_out_of_range_nx() -> None:
    """nx below the documented minimum is rejected."""
    result = solve_heat_1d(
        alpha=1.0, length=1.0, nx=5, t_end=0.1, nt=1000, initial="sin(pi*x)"
    )
    assert result["ok"] is False
    assert "nx" in result["error"].lower()


# ---------------------------------------------------------------------------
# Boundary condition types (PB-)
# ---------------------------------------------------------------------------


def _load_full_grid(result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the full (non-downsampled) grid persisted by a solver."""
    assert result["ok"] is True, result.get("error")
    with np.load(Path(result["result"]["path"])) as data:
        return data["x"], data["t"], data["u"]


def test_heat_adiabatic_conserves_total_heat() -> None:
    """Both ends Neumann with flux=0: no heat may enter or leave the rod.

    For the mandated ghost-node stencil the unweighted node sum is conserved
    exactly for an antisymmetric profile; for a generic profile the exact
    discrete invariant uses half weights on the two boundary nodes (the
    trapezoidal rule), which is checked with an even-symmetric profile.
    """
    common = dict(alpha=0.7, length=1.0, nx=60, t_end=0.2, nt=2000)
    insulated = {"type": "neumann", "flux": 0.0}

    odd = solve_heat_1d(
        **common, initial="1 + 0.5*cos(pi*x)", left=insulated, right=insulated
    )
    x, _, u = _load_full_grid(odd)
    dx = float(x[1] - x[0])
    totals = np.sum(u, axis=1) * dx
    assert float(np.max(np.abs(totals - totals[0])) / abs(totals[0])) < 1e-6

    even = solve_heat_1d(
        **common, initial="0.5 + cos(2*pi*x)", left=insulated, right=insulated
    )
    x2, _, u2 = _load_full_grid(even)
    dx2 = float(x2[1] - x2[0])
    weights = np.full(x2.size, dx2)
    weights[0] = weights[-1] = 0.5 * dx2
    conserved = u2 @ weights
    assert (
        float(np.max(np.abs(conserved - conserved[0])) / abs(conserved[0])) < 1e-9
    )


def test_heat_neumann_left_dirichlet_right_reaches_linear_steady_state() -> None:
    """Left Neumann flux q and right Dirichlet V have steady state V + q*(x-L)."""
    flux = 5.0
    right_value = 100.0
    result = solve_heat_1d(
        alpha=1.0,
        length=1.0,
        nx=40,
        t_end=4.0,
        nt=20000,
        initial="100",
        left={"type": "neumann", "flux": flux},
        right=right_value,
    )
    x, _, u = _load_full_grid(result)
    steady = right_value + flux * (x - 1.0)
    final_error = float(np.max(np.abs(u[-1] - steady)))
    initial_error = float(np.max(np.abs(u[0] - steady)))
    assert final_error < 2e-3
    assert final_error < 0.01 * initial_error


def test_heat_robin_large_h_matches_dirichlet() -> None:
    """h -> infinity drives the Robin ends to the ambient Dirichlet values."""
    common = dict(alpha=1.0, length=1.0, nx=40, t_end=0.1, nt=400, initial="10")
    robin = solve_heat_1d(
        **common,
        left={"type": "robin", "h": 1.0e6, "ambient": 20.0},
        right={"type": "robin", "h": 1.0e6, "ambient": 5.0},
    )
    dirichlet = solve_heat_1d(**common, left=20.0, right=5.0)
    x_robin, _, u_robin = _load_full_grid(robin)
    x_dirichlet, _, u_dirichlet = _load_full_grid(dirichlet)
    assert np.array_equal(x_robin, x_dirichlet)
    scale = float(np.max(np.abs(u_dirichlet)))
    relative = float(np.max(np.abs(u_robin[-1] - u_dirichlet[-1]))) / scale
    assert relative < 1e-3


def test_heat_robin_differs_from_neumann() -> None:
    """A finite-h Robin end differs from an insulated Neumann end."""
    common = dict(alpha=1.0, length=1.0, nx=40, t_end=0.02, nt=80, initial="10")
    robin = solve_heat_1d(
        **common,
        left={"type": "robin", "h": 2.0, "ambient": 50.0},
        right={"type": "robin", "h": 2.0, "ambient": -50.0},
    )
    neumann = solve_heat_1d(
        **common,
        left={"type": "neumann", "flux": 0.0},
        right={"type": "neumann", "flux": 0.0},
    )
    _, _, u_robin = _load_full_grid(robin)
    _, _, u_neumann = _load_full_grid(neumann)
    assert float(np.max(np.abs(u_robin[-1] - u_neumann[-1]))) > 1e-2

    # h=0 reduces the Robin law to du/dx=0, i.e. the Neumann solution exactly.
    zero_h = solve_heat_1d(
        **common,
        left={"type": "robin", "h": 0.0, "ambient": 123.0},
        right={"type": "robin", "h": 0.0, "ambient": -7.0},
    )
    _, _, u_zero_h = _load_full_grid(zero_h)
    assert np.allclose(u_zero_h, u_neumann, rtol=0.0, atol=1e-12)


def test_wave_free_end_conserves_energy() -> None:
    """A free (Neumann) end and a fixed end must conserve the leapfrog energy."""
    result = solve_wave_1d(
        c=1.0,
        length=1.0,
        nx=100,
        t_end=0.5,
        nt=200,
        initial_displacement="cos(pi*x/2)",
        initial_velocity="0",
        left={"type": "neumann", "flux": 0.0},
        right=0.0,
    )
    x, t, u = _load_full_grid(result)
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    # Leapfrog's conserved energy: staggered kinetic term plus the potential
    # averaged over consecutive time levels.
    kinetic = 0.5 * ((u[1:] - u[:-1]) / dt) ** 2
    weights = np.full(x.size, dx)
    weights[0] = weights[-1] = 0.5 * dx
    kinetic_energy = kinetic @ weights
    gradient = (u[:, 1:] - u[:, :-1]) / dx
    potential = 0.5 * gradient**2
    potential_energy = 0.5 * (potential[1:] + potential[:-1]).sum(axis=1) * dx
    energy = kinetic_energy + potential_energy
    assert float(np.max(np.abs(energy - energy[0])) / abs(energy[0])) < 1e-3


def test_wave_robin_large_h_matches_dirichlet() -> None:
    """Wave displacement obeys the same large-h Robin -> Dirichlet limit."""
    common = dict(
        c=1.0,
        length=1.0,
        nx=50,
        t_end=0.1,
        nt=10,
        initial_displacement="sin(pi*x)",
    )
    robin = solve_wave_1d(
        **common,
        left={"type": "robin", "h": 1.0e6, "ambient": 0.0},
        right={"type": "robin", "h": 1.0e6, "ambient": 0.0},
    )
    dirichlet = solve_wave_1d(**common, left=0.0, right=0.0)
    _, _, u_robin = _load_full_grid(robin)
    _, _, u_dirichlet = _load_full_grid(dirichlet)
    scale = float(np.max(np.abs(u_dirichlet)))
    relative = float(np.max(np.abs(u_robin[-1] - u_dirichlet[-1]))) / scale
    assert relative < 1e-3


def test_float_and_dict_dirichlet_boundaries_agree() -> None:
    """A plain float must remain exactly equivalent to a dirichlet dict."""
    common = dict(alpha=0.8, length=1.0, nx=30, t_end=0.02, nt=40, initial="sin(pi*x)")
    floats = solve_heat_1d(**common, left=1.5, right=-0.5)
    dicts = solve_heat_1d(
        **common,
        left={"type": "dirichlet", "value": 1.5},
        right={"type": "dirichlet", "value": -0.5},
    )
    _, _, u_floats = _load_full_grid(floats)
    _, _, u_dicts = _load_full_grid(dicts)
    assert np.array_equal(u_floats, u_dicts)


def test_heat_rejects_unknown_boundary_type() -> None:
    """An unknown boundary condition type must return err."""
    result = solve_heat_1d(
        alpha=1.0,
        length=1.0,
        nx=20,
        t_end=0.01,
        nt=20,
        initial="0",
        left={"type": "banana", "value": 0.0},
    )
    assert result["ok"] is False
    assert "error" in result
    assert "banana" in result["error"].lower()


def test_heat_rejects_robin_missing_ambient() -> None:
    """A Robin condition without ambient is rejected."""
    result = solve_heat_1d(
        alpha=1.0,
        length=1.0,
        nx=20,
        t_end=0.01,
        nt=20,
        initial="0",
        right={"type": "robin", "h": 1.0},
    )
    assert result["ok"] is False
    assert "error" in result
    assert "ambient" in result["error"].lower()


def test_heat_rejects_non_finite_boundary_parameter() -> None:
    """Non-finite flux values are rejected."""
    result = solve_heat_1d(
        alpha=1.0,
        length=1.0,
        nx=20,
        t_end=0.01,
        nt=20,
        initial="0",
        left={"type": "neumann", "flux": float("inf")},
    )
    assert result["ok"] is False
    assert "error" in result
    assert "finite" in result["error"].lower()
