"""PDE tools: explicit finite-difference solvers for 1D heat/wave and 2D Laplace equations."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp
from mcp.server.mcpserver import MCPServer

from sci_mcp.core import err, ok, output_dir

_MAX_SPACE_POINTS = 100
_MAX_TIME_POINTS = 50
# Guardrail against accidental multi-gigabyte allocations: full history is kept.
_MAX_GRID_POINTS = 20_000_000
_MAX_LAPLACE_POINTS = 4_000_000

_OUTPUT_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Validation and conversion helpers
# ---------------------------------------------------------------------------


def _float_param(value: Any, name: str, *, positive: bool = False) -> float:
    """Convert a JSON number to a finite float, optionally requiring positivity."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
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


def _compile_expr(expression: Any, name: str) -> Callable[[np.ndarray], Any]:
    """Parse a scalar expression in the variable ``x`` and lambdify it for numpy."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{name} must be a non-empty expression string")
    x = sp.Symbol("x")
    try:
        parsed = sp.sympify(expression, locals={"x": x})
    except Exception as exc:
        raise ValueError(f"could not parse {name} {expression!r}: {exc}") from exc
    unknown = parsed.free_symbols - {x}
    if unknown:
        names = ", ".join(sorted(symbol.name for symbol in unknown))
        raise ValueError(f"unknown symbol(s) in {name}: {names}")
    try:
        return sp.lambdify(x, parsed, "numpy")
    except Exception as exc:
        raise ValueError(f"could not compile {name}: {exc}") from exc


def _evaluate_on_grid(compiled: Callable[[np.ndarray], Any], grid: np.ndarray, name: str) -> np.ndarray:
    """Evaluate a compiled expression on a grid, broadcasting scalar results."""
    try:
        values = np.asarray(compiled(grid), dtype=float)
    except Exception as exc:
        raise ValueError(f"could not evaluate {name} on the grid: {exc}") from exc
    if values.ndim == 0:
        values = np.full(grid.shape, float(values), dtype=float)
    else:
        try:
            values = np.array(np.broadcast_to(values, grid.shape), dtype=float, copy=True)
        except ValueError as exc:
            raise ValueError(f"{name} must evaluate to a scalar or one value per grid point") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must evaluate to finite values on the grid")
    return values


def _downsample_indices(size: int, limit: int) -> np.ndarray:
    """Return evenly spaced indices covering the full range, at most ``limit`` of them."""
    if size <= limit:
        return np.arange(size)
    return np.unique(np.rint(np.linspace(0.0, size - 1, limit)).astype(int))


def _new_output_path(prefix: str) -> Path:
    """Allocate a unique .npz path inside output_dir() without overwriting existing files."""
    directory = output_dir()
    for _ in range(1000):
        candidate = directory / f"{prefix}_{next(_OUTPUT_COUNTER):06d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


def _saved_path(path: Path) -> str:
    """Return an absolute path string for reports."""
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
#
# ``flux`` is the x-derivative du/dx at the boundary, NOT the outward-normal
# derivative.  The left end has outward normal -x, so its outward-normal
# derivative is -flux; the right end has outward normal +x, so it is +flux.
#
# The Robin condition uses the same x-derivative convention at both ends:
# ``h*(u - ambient) + du/dx = 0`` (unit conductivity k).  Since du/dn is
# -du/dx on the left and +du/dx on the right, the outward-normal form is
# ``h*(u - ambient) - du/dn = 0`` on the left and
# ``h*(u - ambient) + du/dn = 0`` on the right.
#
# Neumann/Robin ends use a ghost node.  For an explicit update the ghost value
# follows from central differences:
#   left:  ghost = u[1]   - 2*dx*flux      (or u[1]   + 2*dx*h*(u[0]-ambient))
#   right: ghost = u[-2]  + 2*dx*flux      (or u[-2]  - 2*dx*h*(u[-1]-ambient))
# Robin is stiff when h*dx is large, so for the new boundary value the ghost
# relation is evaluated at the new time level and the resulting scalar equation
# is solved for the boundary node (semi-implicit boundary treatment).  The
# interior scheme, r/CFL limits and output layout are unchanged.


def _parse_boundary(spec: Any, name: str) -> dict[str, Any]:
    """Normalise a float (Dirichlet value) or boundary-condition dict."""
    if isinstance(spec, dict):
        kind = spec.get("type")
        if not isinstance(kind, str):
            raise ValueError(f"{name}.type must be a string")
        kind = kind.strip().lower()
        if kind == "dirichlet":
            if "value" not in spec:
                raise ValueError(f"{name} Dirichlet condition requires a 'value' field")
            return {
                "type": "dirichlet",
                "value": _float_param(spec["value"], f"{name}.value"),
            }
        if kind == "neumann":
            if "flux" not in spec:
                raise ValueError(f"{name} Neumann condition requires a 'flux' field")
            return {
                "type": "neumann",
                "flux": _float_param(spec["flux"], f"{name}.flux"),
            }
        if kind == "robin":
            missing = [key for key in ("h", "ambient") if key not in spec]
            if missing:
                raise ValueError(
                    f"{name} Robin condition requires the field(s) {', '.join(missing)}"
                )
            return {
                "type": "robin",
                "h": _float_param(spec["h"], f"{name}.h"),
                "ambient": _float_param(spec["ambient"], f"{name}.ambient"),
            }
        raise ValueError(
            f"{name} has unknown boundary condition type {kind!r}; "
            "expected 'dirichlet', 'neumann' or 'robin'"
        )
    return {"type": "dirichlet", "value": _float_param(spec, name)}


def _checked_denominator(denominator: float, name: str) -> float:
    """Reject the singular Robin parameter combination before dividing."""
    if abs(denominator) < 1e-12:
        raise ValueError(
            f"{name} Robin condition is singular for this grid; adjust h, nx or nt"
        )
    return denominator


def _heat_left_next(
    bc: dict[str, Any], u0: float, u1: float, r: float, dx: float
) -> float:
    """New left-boundary value for the heat equation."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = u1 - 2.0 * dx * bc["flux"]
        return u0 + r * (u1 - 2.0 * u0 + ghost)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = 2.0 * r * dx * h
    denominator = _checked_denominator(1.0 - stiffness, "left")
    return ambient + (u0 + 2.0 * r * (u1 - u0) - ambient) / denominator


def _heat_right_next(
    bc: dict[str, Any], un: float, unm1: float, r: float, dx: float
) -> float:
    """New right-boundary value for the heat equation."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = unm1 + 2.0 * dx * bc["flux"]
        return un + r * (ghost - 2.0 * un + unm1)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = 2.0 * r * dx * h
    denominator = _checked_denominator(1.0 + stiffness, "right")
    return ambient + (un + 2.0 * r * (unm1 - un) - ambient) / denominator


def _wave_left_first(
    bc: dict[str, Any],
    u0: float,
    v0: float,
    u1: float,
    c2: float,
    dx: float,
    dt: float,
) -> float:
    """Left-boundary value for the Taylor first wave step."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = u1 - 2.0 * dx * bc["flux"]
        return u0 + dt * v0 + 0.5 * c2 * (u1 - 2.0 * u0 + ghost)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = c2 * dx * h
    denominator = _checked_denominator(1.0 - stiffness, "left")
    return ambient + (u0 + dt * v0 + c2 * (u1 - u0) - ambient) / denominator


def _wave_right_first(
    bc: dict[str, Any],
    un: float,
    vn: float,
    unm1: float,
    c2: float,
    dx: float,
    dt: float,
) -> float:
    """Right-boundary value for the Taylor first wave step."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = unm1 + 2.0 * dx * bc["flux"]
        return un + dt * vn + 0.5 * c2 * (ghost - 2.0 * un + unm1)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = c2 * dx * h
    denominator = _checked_denominator(1.0 + stiffness, "right")
    return ambient + (un + dt * vn + c2 * (unm1 - un) - ambient) / denominator


def _wave_left_next(
    bc: dict[str, Any],
    u0: float,
    up0: float,
    u1: float,
    c2: float,
    dx: float,
) -> float:
    """New left-boundary displacement for the wave equation."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = u1 - 2.0 * dx * bc["flux"]
        return 2.0 * u0 - up0 + c2 * (u1 - 2.0 * u0 + ghost)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = 2.0 * c2 * dx * h
    denominator = _checked_denominator(1.0 - stiffness, "left")
    correction = 2.0 * u0 - up0 + 2.0 * c2 * (u1 - u0)
    return ambient + (correction - ambient) / denominator


def _wave_right_next(
    bc: dict[str, Any],
    un: float,
    upn: float,
    unm1: float,
    c2: float,
    dx: float,
) -> float:
    """New right-boundary displacement for the wave equation."""
    kind = bc["type"]
    if kind == "dirichlet":
        return bc["value"]
    if kind == "neumann":
        ghost = unm1 + 2.0 * dx * bc["flux"]
        return 2.0 * un - upn + c2 * (ghost - 2.0 * un + unm1)
    h = bc["h"]
    ambient = bc["ambient"]
    stiffness = 2.0 * c2 * dx * h
    denominator = _checked_denominator(1.0 + stiffness, "right")
    correction = 2.0 * un - upn + 2.0 * c2 * (unm1 - un)
    return ambient + (correction - ambient) / denominator


# ---------------------------------------------------------------------------
# Heat equation
# ---------------------------------------------------------------------------


def solve_heat_1d(
    alpha: float,
    length: float,
    nx: int,
    t_end: float,
    nt: int,
    initial: str,
    left: float | dict = 0.0,
    right: float | dict = 0.0,
) -> dict:
    """Solve u_t = alpha*u_xx with explicit FTCS and configurable ends.

    ``nx`` is the number of spatial intervals (nx+1 grid points) and ``nt`` the
    number of time steps. The scheme requires r = alpha*dt/dx^2 <= 0.5; larger
    values are rejected as unstable. Each of ``left``/``right`` is either a
    float (Dirichlet value) or a dict:
    ``{"type": "dirichlet", "value": v}``,
    ``{"type": "neumann", "flux": q}`` or
    ``{"type": "robin", "h": h, "ambient": a}``.

    Sign conventions: ``flux`` is the x-derivative du/dx at the boundary, not
    the outward-normal derivative. At the left end the outward normal is -x,
    so the outward-normal derivative there is -flux; at the right end it is
    +flux. The Robin condition uses the same x-derivative convention at both
    ends: ``h*(u - ambient) + du/dx = 0`` (unit conductivity k), i.e.
    ``h*(u - ambient) - du/dn = 0`` on the left and
    ``h*(u - ambient) + du/dn = 0`` on the right. Neumann/Robin ends use ghost
    nodes; Robin is applied at the new time level so that large h remains
    stable. The full grid is saved as .npz, while the returned arrays are
    downsampled to at most 100 spatial and 50 time points.
    """
    try:
        a = _float_param(alpha, "alpha", positive=True)
        domain_length = _float_param(length, "length", positive=True)
        end_time = _float_param(t_end, "t_end", positive=True)
        n_x = _int_param(nx, "nx", 10, 2000)
        n_t = _int_param(nt, "nt", 10, 1_000_000)
        left_bc = _parse_boundary(left, "left")
        right_bc = _parse_boundary(right, "right")

        if (n_x + 1) * (n_t + 1) > _MAX_GRID_POINTS:
            raise ValueError(
                f"grid too large: (nx+1)*(nt+1) = {(n_x + 1) * (n_t + 1)} exceeds "
                f"the limit of {_MAX_GRID_POINTS} stored points; reduce nt or nx"
            )

        dx = domain_length / n_x
        dt = end_time / n_t
        r = a * dt / (dx * dx)
        if r > 0.5:
            return err(
                "solve_heat_1d failed: explicit FTCS is unstable because "
                f"r = alpha*dt/dx^2 = {r:.6g} exceeds 0.5; increase nt "
                f"(currently {n_t}) or reduce t_end"
            )

        compiled = _compile_expr(initial, "initial")
        x = np.linspace(0.0, domain_length, n_x + 1)
        initial_values = _evaluate_on_grid(compiled, x, "initial")

        u = np.empty((n_t + 1, n_x + 1), dtype=float)
        u[0] = initial_values
        if left_bc["type"] == "dirichlet":
            u[0, 0] = left_bc["value"]
        if right_bc["type"] == "dirichlet":
            u[0, -1] = right_bc["value"]
        for step in range(n_t):
            current = u[step]
            following = u[step + 1]
            following[1:-1] = current[1:-1] + r * (
                current[2:] - 2.0 * current[1:-1] + current[:-2]
            )
            following[0] = _heat_left_next(left_bc, current[0], current[1], r, dx)
            following[-1] = _heat_right_next(
                right_bc, current[-1], current[-2], r, dx
            )

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("heat_1d")
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
                "shape": [int(returned_u.shape[0]), int(returned_u.shape[1])],
                "original_shape": [int(u.shape[0]), int(u.shape[1])],
                "x_len": int(x_index.size),
                "t_len": int(t_index.size),
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "downsampled": downsampled,
                "original_x_length": int(x.size),
                "original_t_length": int(t.size),
                "dx": float(dx),
                "dt": float(dt),
                "r": float(r),
            }
        )
    except Exception as exc:
        return err(f"solve_heat_1d failed: {exc}")


# ---------------------------------------------------------------------------
# Wave equation
# ---------------------------------------------------------------------------


def solve_wave_1d(
    c: float,
    length: float,
    nx: int,
    t_end: float,
    nt: int,
    initial_displacement: str,
    initial_velocity: str = "0",
    left: float | dict = 0.0,
    right: float | dict = 0.0,
) -> dict:
    """Solve u_tt = c^2*u_xx with explicit central differences.

    ``initial_displacement`` and ``initial_velocity`` are expressions in ``x``.
    The Courant condition c*dt/dx <= 1 is enforced. Each of ``left``/``right``
    is either a float (Dirichlet displacement) or a dict:
    ``{"type": "dirichlet", "value": v}``,
    ``{"type": "neumann", "flux": q}`` or
    ``{"type": "robin", "h": h, "ambient": a}``; boundary conditions act on the
    displacement. ``flux`` is du/dx at the boundary (not the outward-normal
    derivative): the left outward normal is -x so its outward-normal derivative
    is -flux, while the right end's is +flux. Robin means
    ``h*(u - ambient) + du/dx = 0`` at both ends with the same x-derivative
    convention (unit conductivity k). Neumann/Robin ends use ghost
    nodes, with Robin applied at the new time level for stability at large h.
    The full grid is saved as .npz, while the returned arrays are downsampled
    to at most 100 spatial and 50 time points.
    """
    try:
        wave_speed = _float_param(c, "c", positive=True)
        domain_length = _float_param(length, "length", positive=True)
        end_time = _float_param(t_end, "t_end", positive=True)
        n_x = _int_param(nx, "nx", 10, 2000)
        n_t = _int_param(nt, "nt", 10, 1_000_000)
        left_bc = _parse_boundary(left, "left")
        right_bc = _parse_boundary(right, "right")

        if (n_x + 1) * (n_t + 1) > _MAX_GRID_POINTS:
            raise ValueError(
                f"grid too large: (nx+1)*(nt+1) = {(n_x + 1) * (n_t + 1)} exceeds "
                f"the limit of {_MAX_GRID_POINTS} stored points; reduce nt or nx"
            )

        dx = domain_length / n_x
        dt = end_time / n_t
        courant = wave_speed * dt / dx
        if courant > 1.0:
            return err(
                "solve_wave_1d failed: CFL condition violated because "
                f"c*dt/dx = {courant:.6g} exceeds 1; increase nt "
                f"(currently {n_t}) or reduce t_end"
            )

        displacement_compiled = _compile_expr(initial_displacement, "initial_displacement")
        velocity_compiled = _compile_expr(initial_velocity, "initial_velocity")
        x = np.linspace(0.0, domain_length, n_x + 1)
        displacement = _evaluate_on_grid(displacement_compiled, x, "initial_displacement")
        velocity = _evaluate_on_grid(velocity_compiled, x, "initial_velocity")

        c2 = courant * courant
        u = np.empty((n_t + 1, n_x + 1), dtype=float)
        u[0] = displacement
        if left_bc["type"] == "dirichlet":
            u[0, 0] = left_bc["value"]
        if right_bc["type"] == "dirichlet":
            u[0, -1] = right_bc["value"]

        # Taylor first step: u^1 = u^0 + dt*v + (dt^2/2)*c^2*u_xx^0.
        u[1, 1:-1] = (
            u[0, 1:-1]
            + dt * velocity[1:-1]
            + 0.5 * c2 * (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2])
        )
        u[1, 0] = _wave_left_first(
            left_bc, u[0, 0], velocity[0], u[0, 1], c2, dx, dt
        )
        u[1, -1] = _wave_right_first(
            right_bc, u[0, -1], velocity[-1], u[0, -2], c2, dx, dt
        )

        for step in range(1, n_t):
            current = u[step]
            previous = u[step - 1]
            following = u[step + 1]
            following[1:-1] = (
                2.0 * current[1:-1]
                - previous[1:-1]
                + c2 * (current[2:] - 2.0 * current[1:-1] + current[:-2])
            )
            following[0] = _wave_left_next(
                left_bc, current[0], previous[0], current[1], c2, dx
            )
            following[-1] = _wave_right_next(
                right_bc, current[-1], previous[-1], current[-2], c2, dx
            )

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("wave_1d")
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
                "shape": [int(returned_u.shape[0]), int(returned_u.shape[1])],
                "original_shape": [int(u.shape[0]), int(u.shape[1])],
                "x_len": int(x_index.size),
                "t_len": int(t_index.size),
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "downsampled": downsampled,
                "original_x_length": int(x.size),
                "original_t_length": int(t.size),
                "dx": float(dx),
                "dt": float(dt),
                "courant": float(courant),
            }
        )
    except Exception as exc:
        return err(f"solve_wave_1d failed: {exc}")


# ---------------------------------------------------------------------------
# Laplace equation
# ---------------------------------------------------------------------------


def solve_laplace_2d(
    nx: int,
    ny: int,
    left: float,
    right: float,
    top: float,
    bottom: float,
    tol: float = 1e-6,
    max_iter: int = 20000,
) -> dict:
    """Solve u_xx + u_yy = 0 on the unit square with constant Dirichlet edges.

    Red-black Gauss-Seidel is iterated until max|u_new-u_old| < tol. If
    ``max_iter`` is reached first the call still succeeds, with
    ``converged = False`` and the achieved ``iterations`` count. The full grid
    is saved as .npz; the returned grid is downsampled to at most 100x100.
    """
    try:
        n_x = _int_param(nx, "nx", 2, 2000)
        n_y = _int_param(ny, "ny", 2, 2000)
        if (n_x + 1) * (n_y + 1) > _MAX_LAPLACE_POINTS:
            raise ValueError(
                f"grid too large: (nx+1)*(ny+1) = {(n_x + 1) * (n_y + 1)} exceeds "
                f"the limit of {_MAX_LAPLACE_POINTS} points; reduce nx or ny"
            )
        tolerance = _float_param(tol, "tol", positive=True)
        max_steps = _int_param(max_iter, "max_iter", 1, 1_000_000)
        left_value = _float_param(left, "left")
        right_value = _float_param(right, "right")
        top_value = _float_param(top, "top")
        bottom_value = _float_param(bottom, "bottom")

        x = np.linspace(0.0, 1.0, n_x + 1)
        y = np.linspace(0.0, 1.0, n_y + 1)

        u = np.zeros((n_y + 1, n_x + 1), dtype=float)
        u[0, :] = bottom_value
        u[-1, :] = top_value
        u[:, 0] = left_value
        u[:, -1] = right_value
        # Corners belong to two edges; use the average so the initial guess is
        # consistent with both adjacent Dirichlet values.
        u[0, 0] = 0.5 * (bottom_value + left_value)
        u[0, -1] = 0.5 * (bottom_value + right_value)
        u[-1, 0] = 0.5 * (top_value + left_value)
        u[-1, -1] = 0.5 * (top_value + right_value)

        interior = u[1:-1, 1:-1]
        rows = np.arange(1, n_y)
        columns = np.arange(1, n_x)
        red_mask = (rows[:, None] + columns[None, :]) % 2 == 0
        black_mask = ~red_mask

        converged = False
        iterations = 0
        for iteration in range(1, max_steps + 1):
            iterations = iteration
            neighbours = 0.25 * (
                u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:]
            )

            delta = 0.0
            if red_mask.any():
                delta = float(np.max(np.abs(neighbours[red_mask] - interior[red_mask])))
                interior[red_mask] = neighbours[red_mask]

            if black_mask.any():
                neighbours = 0.25 * (
                    u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:]
                )
                delta = max(
                    delta,
                    float(np.max(np.abs(neighbours[black_mask] - interior[black_mask]))),
                )
                interior[black_mask] = neighbours[black_mask]

            if delta < tolerance:
                converged = True
                break

        path = _new_output_path("laplace_2d")
        np.savez(path, x=x, y=y, u=u)

        x_index = _downsample_indices(x.size, _MAX_SPACE_POINTS)
        y_index = _downsample_indices(y.size, _MAX_SPACE_POINTS)
        downsampled = bool(x_index.size < x.size or y_index.size < y.size)
        returned_u = u[np.ix_(y_index, x_index)]

        return ok(
            {
                "x": x[x_index].tolist(),
                "y": y[y_index].tolist(),
                "u": returned_u.tolist(),
                "path": _saved_path(path),
                "shape": [int(returned_u.shape[0]), int(returned_u.shape[1])],
                "original_shape": [int(u.shape[0]), int(u.shape[1])],
                "x_len": int(x_index.size),
                "y_len": int(y_index.size),
                "iterations": int(iterations),
                "converged": bool(converged),
                "tol": float(tolerance),
                "max_iter": int(max_steps),
                "max": float(np.max(u)),
                "min": float(np.min(u)),
                "downsampled": downsampled,
                "original_x_length": int(x.size),
                "original_y_length": int(y.size),
            }
        )
    except Exception as exc:
        return err(f"solve_laplace_2d failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(solve_heat_1d)
    mcp.tool()(solve_wave_1d)
    mcp.tool()(solve_laplace_2d)
