"""Additional PDE tools: variable-coefficient heat, semi-general 1D PDE, 3D wave.

Scope is intentionally limited:

* ``solve_heat_1d_variable`` treats ``alpha(x)`` as a finite positive
  conductivity and uses an explicit finite-volume flux form.
* ``solve_pde_1d`` is only *semi-general*: it evaluates an arbitrary
  ``F(x, t, u, u_x, u_xx)`` with an explicit Euler step. Stability cannot be
  established for arbitrary ``F`` and is therefore never promised.
* ``solve_wave_3d`` solves the constant-coefficient wave equation on the unit
  cube with Dirichlet faces.

All results are saved to ``output_dir()`` as ``.npz`` files; the returned
arrays are evenly downsampled. Non-finite values never escape a successful
result: the solvers stop and return an ``err(...)`` envelope instead.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from mcp.server.mcpserver import MCPServer

from sci_mcp.core import err, ok, output_dir

# Return-size limits (BRIEF section 11).
_MAX_1D_SPACE_POINTS = 100
_MAX_1D_TIME_POINTS = 50
_MAX_3D_SPACE_POINTS = 20
_MAX_3D_TIME_POINTS = 20

# Documented parameter and storage limits.
_MAX_1D_NX = 5_000
_MAX_1D_STEPS = 1_000_000
_MAX_3D_GRID_POINTS = 500_000
_MAX_STORED_POINTS = 20_000_000

# Stability caveat required for the semi-general solver (BRIEF section 15).
_STABILITY_NOTE = (
    "explicit scheme; stability is not guaranteed for arbitrary F. "
    "Reduce dt/nx or use a specialized solver for stiff/complex problems."
)

_OUTPUT_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Validation and expression helpers
# ---------------------------------------------------------------------------


def _float_param(value: Any, name: str) -> float:
    """Convert a JSON number to a finite float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float_param(value: Any, name: str) -> float:
    """Convert a JSON number to a finite, strictly positive float."""
    number = _float_param(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return number


def _int_param(value: Any, name: str, low: int, high: int) -> int:
    """Validate an integer parameter against an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high} (got {number})")
    return number


def _compile_expr(
    expression: Any, name: str, variables: tuple[str, ...]
) -> Callable[..., Any]:
    """Parse an expression in the given symbols and lambdify it for numpy.

    ``eval`` is never called directly; sympy performs parsing and code
    generation. Symbols outside ``variables`` are rejected, which also blocks
    accidental dependence on hidden names.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{name} must be a non-empty expression string")
    symbols = tuple(sp.Symbol(symbol) for symbol in variables)
    local_dict = dict(zip(variables, symbols))
    try:
        parsed = sp.sympify(expression, locals=local_dict)
    except Exception as exc:
        raise ValueError(f"could not parse {name} {expression!r}: {exc}") from exc
    unknown = parsed.free_symbols - set(symbols)
    if unknown:
        names = ", ".join(sorted(symbol.name for symbol in unknown))
        raise ValueError(f"unknown symbol(s) in {name}: {names}")
    try:
        if len(symbols) == 1:
            return sp.lambdify(symbols[0], parsed, "numpy")
        return sp.lambdify(symbols, parsed, "numpy")
    except Exception as exc:
        raise ValueError(f"could not compile {name}: {exc}") from exc


def _as_grid_values(
    raw: Any, shape: tuple[int, ...], name: str
) -> np.ndarray:
    """Normalise a lambdified result to ``shape`` and require finite values."""
    try:
        values = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    if values.ndim == 0:
        values = np.full(shape, float(values), dtype=float)
    else:
        try:
            values = np.array(np.broadcast_to(values, shape), dtype=float, copy=True)
        except ValueError as exc:
            raise ValueError(
                f"{name} must evaluate to a scalar or one value per grid point"
            ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to finite values on the grid")
    return values


def _evaluate_on_grid_1d(
    compiled: Callable[..., Any], grid: np.ndarray, name: str
) -> np.ndarray:
    """Evaluate a univariate expression on a 1D grid, broadcasting scalars."""
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = compiled(grid)
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    return _as_grid_values(raw, grid.shape, name)


def _evaluate_on_grid_3d(
    compiled: Callable[..., Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    name: str,
) -> np.ndarray:
    """Evaluate an expression in (x, y, z) on a (z, y, x) grid via broadcasting."""
    shape = (z.size, y.size, x.size)
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            raw = compiled(x[None, None, :], y[None, :, None], z[:, None, None])
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    return _as_grid_values(raw, shape, name)


def _downsample_indices(size: int, limit: int) -> np.ndarray:
    """Return evenly spaced indices covering the full range, at most ``limit``."""
    if size <= limit:
        return np.arange(size)
    return np.unique(np.rint(np.linspace(0.0, size - 1, limit)).astype(int))


def _new_output_path(prefix: str) -> Path:
    """Allocate a unique .npz path inside output_dir() without overwriting."""
    directory = output_dir()
    for _ in range(1000):
        candidate = directory / f"{prefix}_{next(_OUTPUT_COUNTER):06d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


def _saved_path(path: Path) -> str:
    """Return an absolute path string for reports."""
    return str(path.resolve())


def _require_storage(points: int, description: str) -> None:
    """Reject histories that would exceed the documented storage guard."""
    if points > _MAX_STORED_POINTS:
        raise ValueError(
            f"grid too large: {description} = {points} stored points exceeds "
            f"the limit of {_MAX_STORED_POINTS}; reduce nt or the grid size"
        )


def _require_cube_grid(nx: int, ny: int, nz: int) -> None:
    """Reject 3D grids above the documented total cell limit."""
    total = nx * ny * nz
    if total > _MAX_3D_GRID_POINTS:
        raise ValueError(
            f"grid too large: nx*ny*nz = {total} exceeds the limit of "
            f"{_MAX_3D_GRID_POINTS}; reduce nx, ny or nz"
        )


def _set_dirichlet_faces_3d(field: np.ndarray, value: float) -> None:
    """Set all six faces of a (z, y, x) array to a constant Dirichlet value."""
    field[0, :, :] = value
    field[-1, :, :] = value
    field[:, 0, :] = value
    field[:, -1, :] = value
    field[:, :, 0] = value
    field[:, :, -1] = value


# ---------------------------------------------------------------------------
# Variable-coefficient 1D heat equation
# ---------------------------------------------------------------------------


def solve_heat_1d_variable(
    alpha_expr: str,
    length: float = 1.0,
    nx: int = 100,
    t_end: float = 0.1,
    nt: int = 1000,
    initial: str = "sin(pi*x)",
    left: float = 0.0,
    right: float = 0.0,
) -> dict:
    """Solve u_t = d/dx(alpha(x) * du/dx) with an explicit flux scheme.

    ``alpha_expr`` is an expression in x and must be strictly positive on the
    spatial grid. Face conductivities ``alpha(x_{i+1/2})`` and
    ``alpha(x_{i-1/2})`` are evaluated on half nodes, giving the conservative
    flux form

        u_i^{n+1} = u_i^n + dt/dx^2 * (
            alpha_{i+1/2} (u_{i+1}^n - u_i^n)
            - alpha_{i-1/2} (u_i^n - u_{i-1}^n))

    The explicit scheme is rejected when ``dt * max(alpha) / dx^2 > 0.5``.
    ``nx`` is the number of intervals (nx+1 nodes), ``nt`` the number of time
    steps. The ends use constant Dirichlet values. The full (t, x) grid is
    saved to .npz; returned arrays are downsampled to at most 100 space and
    50 time points.
    """
    try:
        domain_length = _positive_float_param(length, "length")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", 10, _MAX_1D_NX)
        n_t = _int_param(nt, "nt", 10, _MAX_1D_STEPS)
        left_value = _float_param(left, "left")
        right_value = _float_param(right, "right")

        # Memory guardrail comes before any large allocation.
        _require_storage((n_x + 1) * (n_t + 1), "(nx+1)*(nt+1)")

        dx = domain_length / n_x
        dt = end_time / n_t

        alpha_compiled = _compile_expr(alpha_expr, "alpha_expr", ("x",))
        x = np.linspace(0.0, domain_length, n_x + 1)
        x_faces = 0.5 * (x[:-1] + x[1:])
        alpha_nodes = _evaluate_on_grid_1d(alpha_compiled, x, "alpha_expr")
        alpha_faces = _evaluate_on_grid_1d(alpha_compiled, x_faces, "alpha_expr")
        if np.any(alpha_nodes <= 0.0) or np.any(alpha_faces <= 0.0):
            raise ValueError(
                "alpha_expr must be strictly positive on the spatial grid; "
                "non-positive conductivity is not supported"
            )
        alpha_max = float(max(np.max(alpha_nodes), np.max(alpha_faces)))

        r = dt * alpha_max / (dx * dx)
        if r > 0.5:
            return err(
                "solve_heat_1d_variable failed: explicit scheme is unstable "
                f"because dt*max(alpha)/dx^2 = {r:.6g} exceeds 0.5; increase nt "
                f"(currently {n_t}) or reduce t_end"
            )

        initial_compiled = _compile_expr(initial, "initial", ("x",))
        initial_values = _evaluate_on_grid_1d(initial_compiled, x, "initial")
        initial_values[0] = left_value
        initial_values[-1] = right_value

        u = np.empty((n_t + 1, n_x + 1), dtype=float)
        u[0] = initial_values
        t = np.linspace(0.0, end_time, n_t + 1)
        flux_scale = dt / (dx * dx)
        for step in range(n_t):
            current = u[step]
            following = u[step + 1]
            centre = current[1:-1]
            updated = centre + flux_scale * (
                alpha_faces[1:] * (current[2:] - centre)
                - alpha_faces[:-1] * (centre - current[:-2])
            )
            following[1:-1] = updated
            following[0] = left_value
            following[-1] = right_value
            if not np.all(np.isfinite(updated)):
                return err(
                    "solve_heat_1d_variable failed: non-finite value (NaN or "
                    f"infinity) detected at step {step + 1} of {n_t}; reduce "
                    "dt or check alpha_expr"
                )

        path = _new_output_path("heat_1d_variable")
        np.savez(path, x=x, t=t, u=u)

        x_index = _downsample_indices(x.size, _MAX_1D_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_1D_TIME_POINTS)
        downsampled = bool(x_index.size < x.size or t_index.size < t.size)
        returned_u = u[np.ix_(t_index, x_index)]

        return ok(
            {
                "x": x[x_index].tolist(),
                "t": t[t_index].tolist(),
                "u": returned_u.tolist(),
                "path": _saved_path(path),
                "shape": [int(size) for size in returned_u.shape],
                "original_shape": [int(size) for size in u.shape],
                "downsampled": downsampled,
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "alpha_max": alpha_max,
                "alpha_min": float(min(np.min(alpha_nodes), np.min(alpha_faces))),
                "dx": float(dx),
                "dt": float(dt),
                "r": float(r),
                "left": float(left_value),
                "right": float(right_value),
            }
        )
    except Exception as exc:
        return err(f"solve_heat_1d_variable failed: {exc}")


# ---------------------------------------------------------------------------
# Semi-general explicit 1D PDE
# ---------------------------------------------------------------------------


def solve_pde_1d(
    equation: str,
    length: float = 1.0,
    nx: int = 100,
    t_end: float = 0.1,
    nt: int = 1000,
    initial: str = "sin(pi*x)",
    left: float = 0.0,
    right: float = 0.0,
) -> dict:
    """Solve u_t = F(x, t, u, u_x, u_xx) with an explicit Euler scheme.

    ``equation`` is a sympy expression for ``F`` in the symbols ``x``, ``t``,
    ``u``, ``u_x`` and ``u_xx``. Spatial derivatives are second-order centred
    differences evaluated on the interior nodes at every step; the ends keep
    the constant Dirichlet values ``left``/``right``.

    This is only a *semi-general* solver: no stability condition can be
    derived for arbitrary ``F``. The result therefore always contains a
    ``stability_note`` recommending smaller steps or a specialized solver.
    If a step produces a non-finite value the solver stops and returns
    ``err(...)`` with the offending step number.

    The full (t, x) grid is saved to .npz; returned arrays are downsampled to
    at most 100 space and 50 time points.
    """
    try:
        domain_length = _positive_float_param(length, "length")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", 3, _MAX_1D_NX)
        n_t = _int_param(nt, "nt", 1, _MAX_1D_STEPS)
        left_value = _float_param(left, "left")
        right_value = _float_param(right, "right")

        # Memory guardrail comes before any large allocation.
        _require_storage((n_x + 1) * (n_t + 1), "(nx+1)*(nt+1)")

        dx = domain_length / n_x
        dt = end_time / n_t

        right_hand_side = _compile_expr(
            equation, "equation", ("x", "t", "u", "u_x", "u_xx")
        )
        initial_compiled = _compile_expr(initial, "initial", ("x",))
        x = np.linspace(0.0, domain_length, n_x + 1)
        initial_values = _evaluate_on_grid_1d(initial_compiled, x, "initial")
        initial_values[0] = left_value
        initial_values[-1] = right_value

        u = np.empty((n_t + 1, n_x + 1), dtype=float)
        u[0] = initial_values
        t = np.linspace(0.0, end_time, n_t + 1)
        x_interior = x[1:-1]

        for step in range(n_t):
            current = u[step]
            following = u[step + 1]
            centre = current[1:-1]
            u_x = (current[2:] - current[:-2]) / (2.0 * dx)
            u_xx = (current[2:] - 2.0 * centre + current[:-2]) / (dx * dx)
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    raw = right_hand_side(
                        x_interior, t[step], centre, u_x, u_xx
                    )
                    rhs = np.asarray(raw, dtype=float)
                    if rhs.ndim == 0:
                        rhs = np.full(centre.shape, float(rhs), dtype=float)
                    else:
                        try:
                            rhs = np.broadcast_to(rhs, centre.shape)
                        except ValueError as exc:
                            raise ValueError(
                                "equation must evaluate to a scalar or one "
                                "value per grid point"
                            ) from exc
                    updated = centre + dt * rhs
            except Exception as exc:
                raise ValueError(
                    f"could not evaluate equation at step {step + 1}: {exc}"
                ) from exc

            following[1:-1] = updated
            following[0] = left_value
            following[-1] = right_value
            if not np.all(np.isfinite(updated)):
                return err(
                    "solve_pde_1d failed: non-finite value (NaN or infinity) "
                    f"detected at step {step + 1} of {n_t}; the explicit scheme "
                    "blew up. Reduce dt/nx or use a specialized solver for "
                    "stiff/complex problems.",
                    step=int(step + 1),
                    stability_note=_STABILITY_NOTE,
                )

        path = _new_output_path("pde_1d")
        np.savez(path, x=x, t=t, u=u)

        x_index = _downsample_indices(x.size, _MAX_1D_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_1D_TIME_POINTS)
        downsampled = bool(x_index.size < x.size or t_index.size < t.size)
        returned_u = u[np.ix_(t_index, x_index)]

        return ok(
            {
                "x": x[x_index].tolist(),
                "t": t[t_index].tolist(),
                "u": returned_u.tolist(),
                "path": _saved_path(path),
                "shape": [int(size) for size in returned_u.shape],
                "original_shape": [int(size) for size in u.shape],
                "downsampled": downsampled,
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "dx": float(dx),
                "dt": float(dt),
                "left": float(left_value),
                "right": float(right_value),
                "stability_note": _STABILITY_NOTE,
            }
        )
    except Exception as exc:
        return err(f"solve_pde_1d failed: {exc}")


# ---------------------------------------------------------------------------
# 3D wave equation
# ---------------------------------------------------------------------------


def solve_wave_3d(
    c: float,
    nx: int,
    ny: int,
    nz: int,
    t_end: float,
    nt: int,
    initial_displacement: str,
    initial_velocity: str = "0",
    boundary: float = 0.0,
) -> dict:
    """Solve u_tt = c^2 (u_xx + u_yy + u_zz) on the unit cube.

    ``nx``, ``ny``, ``nz`` are the number of intervals per axis and ``nt`` the
    number of time steps. The interior uses the explicit second-order central
    scheme; the first time level uses the Taylor expansion
    ``u(dt) = u0 + dt*v0 + c^2*dt^2/2 * Laplacian(u0)``. All six faces use the
    constant Dirichlet value ``boundary``.

    Stability requires ``c*dt*sqrt(3)/min(dx,dy,dz) <= 1``. Grids above
    500,000 cells and histories above 20,000,000 stored points are rejected.
    The full (t, z, y, x) grid is saved to .npz; returned arrays are
    downsampled to at most 20 points per spatial axis and 20 time points.
    """
    try:
        speed = _positive_float_param(c, "c")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", 2, 2_000)
        n_y = _int_param(ny, "ny", 2, 2_000)
        n_z = _int_param(nz, "nz", 2, 2_000)
        n_t = _int_param(nt, "nt", 1, _MAX_1D_STEPS)
        boundary_value = _float_param(boundary, "boundary")

        # Guardrails come before any array allocation.
        _require_cube_grid(n_x, n_y, n_z)
        _require_storage(
            n_x * n_y * n_z * (n_t + 1), "nx*ny*nz*(nt+1)"
        )

        dx = 1.0 / n_x
        dy = 1.0 / n_y
        dz = 1.0 / n_z
        dt = end_time / n_t
        spacing = min(dx, dy, dz)
        courant = speed * dt * math.sqrt(3.0) / spacing
        if courant > 1.0:
            return err(
                "solve_wave_3d failed: CFL condition violated because "
                f"c*dt*sqrt(3)/min(dx,dy,dz) = {courant:.6g} exceeds 1; "
                f"increase nt (currently {n_t}) or reduce t_end/c"
            )

        displacement = _compile_expr(
            initial_displacement, "initial_displacement", ("x", "y", "z")
        )
        velocity = _compile_expr(
            initial_velocity, "initial_velocity", ("x", "y", "z")
        )
        x = np.linspace(0.0, 1.0, n_x + 1)
        y = np.linspace(0.0, 1.0, n_y + 1)
        z = np.linspace(0.0, 1.0, n_z + 1)
        u0 = _evaluate_on_grid_3d(displacement, x, y, z, "initial_displacement")
        v0 = _evaluate_on_grid_3d(velocity, x, y, z, "initial_velocity")

        # Layout: u[t, z, y, x].
        u = np.empty((n_t + 1, n_z + 1, n_y + 1, n_x + 1), dtype=float)
        u[0] = u0
        _set_dirichlet_faces_3d(u[0], boundary_value)

        dx2 = dx * dx
        dy2 = dy * dy
        dz2 = dz * dz
        c2dt2 = speed * speed * dt * dt
        interior = (slice(1, -1), slice(1, -1), slice(1, -1))

        def laplacian(field: np.ndarray) -> np.ndarray:
            centre = field[interior]
            return (
                (
                    field[1:-1, 1:-1, 2:]
                    - 2.0 * centre
                    + field[1:-1, 1:-1, :-2]
                )
                / dx2
                + (
                    field[1:-1, 2:, 1:-1]
                    - 2.0 * centre
                    + field[1:-1, :-2, 1:-1]
                )
                / dy2
                + (
                    field[2:, 1:-1, 1:-1]
                    - 2.0 * centre
                    + field[:-2, 1:-1, 1:-1]
                )
                / dz2
            )

        # Taylor first step (second-order accurate in time).
        first = u[1]
        first[interior] = (
            u0[interior] + dt * v0[interior] + 0.5 * c2dt2 * laplacian(u0)
        )
        _set_dirichlet_faces_3d(first, boundary_value)
        if not np.all(np.isfinite(first[interior])):
            return err(
                "solve_wave_3d failed: non-finite value (NaN or infinity) "
                "detected at step 1; check initial_displacement, "
                "initial_velocity and the CFL condition"
            )

        for step in range(1, n_t):
            current = u[step]
            previous = u[step - 1]
            following = u[step + 1]
            following[interior] = (
                2.0 * current[interior]
                - previous[interior]
                + c2dt2 * laplacian(current)
            )
            _set_dirichlet_faces_3d(following, boundary_value)
            if not np.all(np.isfinite(following[interior])):
                return err(
                    "solve_wave_3d failed: non-finite value (NaN or infinity) "
                    f"detected at step {step + 1} of {n_t}; reduce dt or "
                    "check the initial data"
                )

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("wave_3d")
        np.savez(path, x=x, y=y, z=z, t=t, u=u)

        x_index = _downsample_indices(x.size, _MAX_3D_SPACE_POINTS)
        y_index = _downsample_indices(y.size, _MAX_3D_SPACE_POINTS)
        z_index = _downsample_indices(z.size, _MAX_3D_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_3D_TIME_POINTS)
        downsampled = bool(
            x_index.size < x.size
            or y_index.size < y.size
            or z_index.size < z.size
            or t_index.size < t.size
        )
        returned_u = u[np.ix_(t_index, z_index, y_index, x_index)]

        return ok(
            {
                "x": x[x_index].tolist(),
                "y": y[y_index].tolist(),
                "z": z[z_index].tolist(),
                "t": t[t_index].tolist(),
                "u": returned_u.tolist(),
                "path": _saved_path(path),
                "shape": [int(size) for size in returned_u.shape],
                "original_shape": [int(size) for size in u.shape],
                "downsampled": downsampled,
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "dx": float(dx),
                "dy": float(dy),
                "dz": float(dz),
                "dt": float(dt),
                "c": float(speed),
                "courant": float(courant),
                "boundary": float(boundary_value),
            }
        )
    except Exception as exc:
        return err(f"solve_wave_3d failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(solve_heat_1d_variable)
    mcp.tool()(solve_pde_1d)
    mcp.tool()(solve_wave_3d)
