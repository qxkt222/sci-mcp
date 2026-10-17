"""Advanced PDE tools: 3D heat, 3D Laplace and 1D Burgers equations."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from mcp.server.mcpserver import MCPServer

from sci_mcp.core import err, ok, output_dir

_MAX_SPACE_POINTS = 20
_MAX_TIME_POINTS = 20
# Documented per-equation limits from the upgrade brief.
_MAX_3D_GRID_POINTS = 500_000
_MAX_HEAT3D_STEPS = 100_000
_MAX_BURGERS_NX = 5_000
_MAX_BURGERS_STEPS = 1_000_000
# Guardrail against accidental multi-gigabyte allocations: the full history
# (including every time level) is kept in memory and written to .npz.
_MAX_STORED_POINTS = 20_000_000

_OUTPUT_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Validation and conversion helpers
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
    """Parse an expression in the given symbols and lambdify it for numpy."""
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


def _evaluate_on_grid(
    compiled: Callable[..., Any], grid: np.ndarray, name: str
) -> np.ndarray:
    """Evaluate a univariate expression on a 1D grid, broadcasting scalars."""
    try:
        values = np.asarray(compiled(grid), dtype=float)
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    if values.ndim == 0:
        values = np.full(grid.shape, float(values), dtype=float)
    else:
        try:
            values = np.array(
                np.broadcast_to(values, grid.shape), dtype=float, copy=True
            )
        except ValueError as exc:
            raise ValueError(
                f"{name} must evaluate to a scalar or one value per grid point"
            ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to finite values on the grid")
    return values


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
        values = compiled(
            x[None, None, :], y[None, :, None], z[:, None, None]
        )
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    try:
        values = np.asarray(values, dtype=float)
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


def _require_cube_limits(nx: int, ny: int, nz: int) -> None:
    """Reject 3D grids above the documented total point limit."""
    total = nx * ny * nz
    if total > _MAX_3D_GRID_POINTS:
        raise ValueError(
            f"grid too large: nx*ny*nz = {total} exceeds the limit of "
            f"{_MAX_3D_GRID_POINTS} points; reduce nx, ny or nz"
        )


def _centre_index_set(intervals: int) -> np.ndarray:
    """Grid indices around the geometric centre of ``intervals`` cells.

    Even interval counts put the centre exactly on a node; odd counts put it
    between two nodes and return both, so an interpolated centre value can be
    formed. Averaging over the Cartesian product of the per-axis sets is the
    multilinear interpolation of the field at the geometric centre.
    """
    half = intervals // 2
    if intervals % 2 == 0:
        return np.array([half], dtype=int)
    return np.array([half, half + 1], dtype=int)


# ---------------------------------------------------------------------------
# 3D heat equation
# ---------------------------------------------------------------------------


def solve_heat_3d(
    alpha: float,
    nx: int,
    ny: int,
    nz: int,
    t_end: float,
    nt: int,
    initial: str,
    boundary: float = 0.0,
) -> dict:
    """Solve u_t = alpha*(u_xx + u_yy + u_zz) on the unit cube with explicit FTCS.

    ``nx``, ``ny``, ``nz`` are the number of intervals per axis (dx = 1/nx,
    ...) and ``nt`` the number of time steps. ``initial`` is an expression in
    x, y and z; all six faces use the constant Dirichlet value ``boundary``.
    The scheme requires r = alpha*dt/min(dx,dy,dz)^2 <= 1/6; larger values are
    rejected as unstable. Grids above 500,000 cells are rejected. The full
    (t, z, y, x) grid is saved to .npz; returned arrays are downsampled to at
    most 20 points per dimension and 20 time points.
    """
    try:
        a = _positive_float_param(alpha, "alpha")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", 2, 2_000)
        n_y = _int_param(ny, "ny", 2, 2_000)
        n_z = _int_param(nz, "nz", 2, 2_000)
        n_t = _int_param(nt, "nt", 1, _MAX_HEAT3D_STEPS)
        boundary_value = _float_param(boundary, "boundary")

        # Memory guardrails come before any array allocation.
        _require_cube_limits(n_x, n_y, n_z)
        stored = (n_x + 1) * (n_y + 1) * (n_z + 1) * (n_t + 1)
        if stored > _MAX_STORED_POINTS:
            raise ValueError(
                f"grid too large: (nx+1)*(ny+1)*(nz+1)*(nt+1) = {stored} "
                f"stored points exceeds the limit of {_MAX_STORED_POINTS}; "
                "reduce nt, nx, ny or nz"
            )

        dx = 1.0 / n_x
        dy = 1.0 / n_y
        dz = 1.0 / n_z
        dt = end_time / n_t
        spacing = min(dx, dy, dz)
        r = a * dt / (spacing * spacing)
        if r > 1.0 / 6.0:
            return err(
                "solve_heat_3d failed: explicit FTCS is unstable because "
                f"r = alpha*dt/min(dx,dy,dz)^2 = {r:.6g} exceeds 1/6; "
                f"increase nt (currently {n_t}) or reduce t_end"
            )

        compiled = _compile_expr(initial, "initial", ("x", "y", "z"))
        x = np.linspace(0.0, 1.0, n_x + 1)
        y = np.linspace(0.0, 1.0, n_y + 1)
        z = np.linspace(0.0, 1.0, n_z + 1)
        initial_values = _evaluate_on_grid_3d(compiled, x, y, z, "initial")

        # Layout: u[t, z, y, x].
        u = np.empty((n_t + 1, n_z + 1, n_y + 1, n_x + 1), dtype=float)
        u[0] = initial_values
        u[0, 0, :, :] = boundary_value
        u[0, -1, :, :] = boundary_value
        u[0, :, 0, :] = boundary_value
        u[0, :, -1, :] = boundary_value
        u[0, :, :, 0] = boundary_value
        u[0, :, :, -1] = boundary_value

        ax = a * dt / (dx * dx)
        ay = a * dt / (dy * dy)
        az = a * dt / (dz * dz)
        for step in range(n_t):
            current = u[step]
            following = u[step + 1]
            centre = current[1:-1, 1:-1, 1:-1]
            laplacian = (
                ax * (current[1:-1, 1:-1, 2:] - 2.0 * centre + current[1:-1, 1:-1, :-2])
                + ay * (current[1:-1, 2:, 1:-1] - 2.0 * centre + current[1:-1, :-2, 1:-1])
                + az * (current[2:, 1:-1, 1:-1] - 2.0 * centre + current[:-2, 1:-1, 1:-1])
            )
            following[1:-1, 1:-1, 1:-1] = centre + laplacian
            following[0, :, :] = boundary_value
            following[-1, :, :] = boundary_value
            following[:, 0, :] = boundary_value
            following[:, -1, :] = boundary_value
            following[:, :, 0] = boundary_value
            following[:, :, -1] = boundary_value

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("heat_3d")
        np.savez(path, x=x, y=y, z=z, t=t, u=u)

        x_index = _downsample_indices(x.size, _MAX_SPACE_POINTS)
        y_index = _downsample_indices(y.size, _MAX_SPACE_POINTS)
        z_index = _downsample_indices(z.size, _MAX_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_TIME_POINTS)
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
                "r": float(r),
                "boundary": float(boundary_value),
            }
        )
    except Exception as exc:
        return err(f"solve_heat_3d failed: {exc}")


# ---------------------------------------------------------------------------
# 3D Laplace equation
# ---------------------------------------------------------------------------


def solve_laplace_3d(
    nx: int,
    ny: int,
    nz: int,
    x0: float = 0.0,
    x1: float = 0.0,
    y0: float = 0.0,
    y1: float = 0.0,
    z0: float = 0.0,
    z1: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 20000,
) -> dict:
    """Solve u_xx + u_yy + u_zz = 0 on the unit cube with Dirichlet faces.

    ``x0``/``x1`` are the values on the faces x=0 and x=1 (likewise for y and
    z). A vectorised Jacobi iteration is run until max|u_new-u_old| < tol; if
    ``max_iter`` is reached first the call still succeeds with
    ``converged = False`` and the achieved ``iterations``. Grids above 500,000
    cells are rejected. The full (z, y, x) grid is saved to .npz and returned
    downsampled to at most 20 points per axis.
    """
    try:
        n_x = _int_param(nx, "nx", 2, 2_000)
        n_y = _int_param(ny, "ny", 2, 2_000)
        n_z = _int_param(nz, "nz", 2, 2_000)
        tolerance = _positive_float_param(tol, "tol")
        max_steps = _int_param(max_iter, "max_iter", 1, 1_000_000)
        boundary_x0 = _float_param(x0, "x0")
        boundary_x1 = _float_param(x1, "x1")
        boundary_y0 = _float_param(y0, "y0")
        boundary_y1 = _float_param(y1, "y1")
        boundary_z0 = _float_param(z0, "z0")
        boundary_z1 = _float_param(z1, "z1")

        # Memory guardrail comes before any array allocation.
        _require_cube_limits(n_x, n_y, n_z)

        x = np.linspace(0.0, 1.0, n_x + 1)
        y = np.linspace(0.0, 1.0, n_y + 1)
        z = np.linspace(0.0, 1.0, n_z + 1)

        # Layout: u[z, y, x]. Faces are applied in x, y, z order, so shared
        # edges keep the value of the last face that touches them.
        u = np.zeros((n_z + 1, n_y + 1, n_x + 1), dtype=float)
        u[:, :, 0] = boundary_x0
        u[:, :, -1] = boundary_x1
        u[:, 0, :] = boundary_y0
        u[:, -1, :] = boundary_y1
        u[0, :, :] = boundary_z0
        u[-1, :, :] = boundary_z1

        interior = (slice(1, -1), slice(1, -1), slice(1, -1))
        converged = False
        iterations = 0
        for iteration in range(1, max_steps + 1):
            iterations = iteration
            following = u.copy()
            following[interior] = (
                u[:-2, 1:-1, 1:-1]
                + u[2:, 1:-1, 1:-1]
                + u[1:-1, :-2, 1:-1]
                + u[1:-1, 2:, 1:-1]
                + u[1:-1, 1:-1, :-2]
                + u[1:-1, 1:-1, 2:]
            ) / 6.0
            delta = float(np.max(np.abs(following[interior] - u[interior])))
            u, following = following, u
            if delta < tolerance:
                converged = True
                break

        path = _new_output_path("laplace_3d")
        np.savez(path, x=x, y=y, z=z, u=u)

        x_index = _downsample_indices(x.size, _MAX_SPACE_POINTS)
        y_index = _downsample_indices(y.size, _MAX_SPACE_POINTS)
        z_index = _downsample_indices(z.size, _MAX_SPACE_POINTS)
        downsampled = bool(
            x_index.size < x.size
            or y_index.size < y.size
            or z_index.size < z.size
        )
        returned_u = u[np.ix_(z_index, y_index, x_index)]
        centre_z = _centre_index_set(n_z)
        centre_y = _centre_index_set(n_y)
        centre_x = _centre_index_set(n_x)
        # Multilinear interpolation of the geometric centre (0.5, 0.5, 0.5).
        centre_value = float(u[np.ix_(centre_z, centre_y, centre_x)].mean())

        return ok(
            {
                "x": x[x_index].tolist(),
                "y": y[y_index].tolist(),
                "z": z[z_index].tolist(),
                "u": returned_u.tolist(),
                "path": _saved_path(path),
                "shape": [int(size) for size in returned_u.shape],
                "original_shape": [int(size) for size in u.shape],
                "downsampled": downsampled,
                "centre": centre_value,
                "center": centre_value,
                "centre_index": [
                    centre_z.tolist(),
                    centre_y.tolist(),
                    centre_x.tolist(),
                ],
                "centre_coordinates": [0.5, 0.5, 0.5],
                "iterations": int(iterations),
                "converged": bool(converged),
                "tol": float(tolerance),
                "max_iter": int(max_steps),
                "max": float(np.max(u)),
                "min": float(np.min(u)),
            }
        )
    except Exception as exc:
        return err(f"solve_laplace_3d failed: {exc}")


# ---------------------------------------------------------------------------
# 1D Burgers equation
# ---------------------------------------------------------------------------


def solve_burgers_1d(
    nu: float,
    length: float = 1.0,
    nx: int = 200,
    t_end: float = 1.0,
    nt: int = 1000,
    initial: str = "sin(pi*x)",
    left: float = 0.0,
    right: float = 0.0,
) -> dict:
    """Solve u_t + u*u_x = nu*u_xx with explicit finite differences.

    ``u_x`` uses a centred difference and ``u_xx`` a second-order centred
    difference. ``initial`` is an expression in x; the ends use the constant
    Dirichlet values ``left`` and ``right``. The explicit scheme requires
    dt <= dx^2/(2*nu) (diffusion) and dt <= dx/max|u| (CFL); a violation is
    reported as an error with a hint to increase nt. The full (t, x) grid is
    saved to .npz and returned downsampled to at most 20 space and 20 time
    points.
    """
    try:
        viscosity = _float_param(nu, "nu")
        if viscosity < 0.0:
            raise ValueError("nu must be greater than or equal to 0")
        domain_length = _positive_float_param(length, "length")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", 10, _MAX_BURGERS_NX)
        n_t = _int_param(nt, "nt", 1, _MAX_BURGERS_STEPS)
        left_value = _float_param(left, "left")
        right_value = _float_param(right, "right")

        # Memory guardrail comes before any array allocation.
        stored = (n_x + 1) * (n_t + 1)
        if stored > _MAX_STORED_POINTS:
            raise ValueError(
                f"grid too large: (nx+1)*(nt+1) = {stored} stored points "
                f"exceeds the limit of {_MAX_STORED_POINTS}; reduce nt or nx"
            )

        dx = domain_length / n_x
        dt = end_time / n_t

        compiled = _compile_expr(initial, "initial", ("x",))
        x = np.linspace(0.0, domain_length, n_x + 1)
        initial_values = _evaluate_on_grid(compiled, x, "initial")
        initial_values[0] = left_value
        initial_values[-1] = right_value
        speed = float(np.max(np.abs(initial_values)))

        if viscosity > 0.0:
            diffusion_limit = dx * dx / (2.0 * viscosity)
            if dt > diffusion_limit:
                return err(
                    "solve_burgers_1d failed: explicit scheme is unstable "
                    f"because dt = {dt:.6g} exceeds the diffusion limit "
                    f"dx^2/(2*nu) = {diffusion_limit:.6g}; increase nt "
                    f"(currently {n_t}) or reduce t_end"
                )
        if speed > 0.0:
            cfl_limit = dx / speed
            if dt > cfl_limit:
                return err(
                    "solve_burgers_1d failed: CFL condition violated because "
                    f"dt = {dt:.6g} exceeds dx/max|u| = {cfl_limit:.6g}; "
                    f"increase nt (currently {n_t}) or reduce t_end"
                )

        u = np.empty((n_t + 1, n_x + 1), dtype=float)
        u[0] = initial_values
        for step in range(n_t):
            current = u[step]
            following = u[step + 1]
            following[1:-1] = (
                current[1:-1]
                - dt
                * current[1:-1]
                * (current[2:] - current[:-2])
                / (2.0 * dx)
                + viscosity
                * dt
                * (current[2:] - 2.0 * current[1:-1] + current[:-2])
                / (dx * dx)
            )
            following[0] = left_value
            following[-1] = right_value

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("burgers_1d")
        np.savez(path, x=x, t=t, u=u)

        x_index = _downsample_indices(x.size, _MAX_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_TIME_POINTS)
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
                "nu": float(viscosity),
                "max_initial_speed": float(speed),
                "left": float(left_value),
                "right": float(right_value),
            }
        )
    except Exception as exc:
        return err(f"solve_burgers_1d failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(solve_heat_3d)
    mcp.tool()(solve_laplace_3d)
    mcp.tool()(solve_burgers_1d)
