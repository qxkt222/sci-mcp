"""2D incompressible Navier-Stokes on a periodic box (vorticity-streamfunction).

Scope, stated honestly: this module solves the *two-dimensional, incompressible*
Navier-Stokes equations in vorticity form on the square ``[0, 2*pi]^2`` with
periodic boundaries.  It is not a general CFD solver: there is no 3D support,
no turbulence modelling, no immersed/solid boundaries and no pressure solve.

Method
------
The vorticity-streamfunction formulation is used::

    omega_t + u*omega_x + v*omega_y = nu * laplacian(omega)
    laplacian(psi) = -omega,   u = psi_y,   v = -psi_x

Every right-hand-side evaluation solves the Poisson equation for ``psi``
spectrally (FFT), so the velocity field is divergence-free to machine
precision.  Spatial derivatives and the nonlinear advection term are evaluated
with the Fourier spectral method (no dealiasing).  Time integration is
**explicit RK2 (midpoint method)**.

Stability
---------
The documented sufficient (heuristic) checks are applied before stepping::

    nu * dt / dx^2 <= 0.5        (viscous / diffusion limit)
    max|u| * dt / dx <= 0.5      (convective CFL limit)

with ``dx = min(2*pi/nx, 2*pi/ny)`` and ``dt = t_end/nt``.  A violation returns
``err(...)`` asking for a larger ``nt``.  Note that a spectral diffusion
operator has the stricter mode-wise limit ``nu*dt*k_max^2 <= 2``; the two
checks above are the project-wide convention and are sufficient for the
moderate Reynolds numbers this tool targets.
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

_TWO_PI = 2.0 * math.pi
_MAX_SPACE_POINTS = 50
_MAX_TIME_POINTS = 20
_MIN_GRID = 8
_MAX_GRID = 256
_MAX_GRID_POINTS = 65_536
_MAX_STEPS = 100_000
# Guardrail against multi-gigabyte history arrays: (nt+1)*ny*nx float64 points.
_MAX_STORED_POINTS = 20_000_000

_OUTPUT_COUNTER = itertools.count(1)


# ---------------------------------------------------------------------------
# Validation helpers
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


def _non_negative_float_param(value: Any, name: str) -> float:
    """Convert a JSON number to a finite, non-negative float."""
    number = _float_param(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be greater than or equal to 0")
    return number


def _int_param(value: Any, name: str, low: int, high: int) -> int:
    """Validate an integer parameter against an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high} (got {number})")
    return number


def _compile_expr(expression: Any, name: str) -> Callable[..., Any]:
    """Parse an expression in the symbols x and y and lambdify it for numpy."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError(f"{name} must be a non-empty expression string")
    x, y = sp.symbols("x y")
    try:
        parsed = sp.sympify(expression, locals={"x": x, "y": y})
    except Exception as exc:
        raise ValueError(f"could not parse {name} {expression!r}: {exc}") from exc
    unknown = parsed.free_symbols - {x, y}
    if unknown:
        names = ", ".join(sorted(symbol.name for symbol in unknown))
        raise ValueError(f"unknown symbol(s) in {name}: {names}")
    try:
        return sp.lambdify((x, y), parsed, "numpy")
    except Exception as exc:
        raise ValueError(f"could not compile {name}: {exc}") from exc


def _evaluate_on_grid_2d(
    compiled: Callable[..., Any], x: np.ndarray, y: np.ndarray, name: str
) -> np.ndarray:
    """Evaluate f(x, y) on the (ny, nx) tensor grid, broadcasting scalars."""
    shape = (y.size, x.size)
    try:
        values = compiled(x[None, :], y[:, None])
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


# ---------------------------------------------------------------------------
# Spectral operators
# ---------------------------------------------------------------------------


def _wavenumbers(size: int) -> np.ndarray:
    """Integer Fourier wavenumbers for the periodic interval [0, 2*pi]."""
    return np.fft.fftfreq(size) * size


def _poisson_psi_hat(omega_hat: np.ndarray, k2_safe: np.ndarray) -> np.ndarray:
    """Solve laplacian(psi) = -omega in Fourier space: psi_hat = omega_hat/k^2."""
    psi_hat = omega_hat / k2_safe
    psi_hat[0, 0] = 0.0  # mean of psi is arbitrary; the zero mode is undetermined
    return psi_hat


def _velocity(psi_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple:
    """Return (u, v) on the grid from psi_hat using spectral derivatives."""
    ky_col = ky[:, None]
    kx_row = kx[None, :]
    u = np.fft.ifft2(1j * ky_col * psi_hat).real
    v = np.fft.ifft2(-1j * kx_row * psi_hat).real
    return u, v


def _rhs(
    omega: np.ndarray,
    nu: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
    k2_safe: np.ndarray,
) -> np.ndarray:
    """Right-hand side -u.grad(omega) + nu*laplacian(omega), spectrally."""
    kx_row = kx[None, :]
    ky_col = ky[:, None]
    omega_hat = np.fft.fft2(omega)
    psi_hat = _poisson_psi_hat(omega_hat, k2_safe)
    u, v = _velocity(psi_hat, kx, ky)
    omega_x = np.fft.ifft2(1j * kx_row * omega_hat).real
    omega_y = np.fft.ifft2(1j * ky_col * omega_hat).real
    advection = u * omega_x + v * omega_y
    diffusion = np.fft.ifft2(-nu * k2 * omega_hat).real
    return diffusion - advection


def _diagnostics(
    omega: np.ndarray, kx: np.ndarray, ky: np.ndarray, k2_safe: np.ndarray
) -> tuple[float, float]:
    """Kinetic energy and enstrophy per unit area via Parseval's identity."""
    omega_hat = np.fft.fft2(omega)
    psi_hat = _poisson_psi_hat(omega_hat, k2_safe)
    u_hat = 1j * ky[:, None] * psi_hat
    v_hat = -1j * kx[None, :] * psi_hat
    cell_count = float(omega.size)
    energy = 0.5 * float(np.sum(np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)) / (
        cell_count * cell_count
    )
    enstrophy = 0.5 * float(np.sum(np.abs(omega_hat) ** 2)) / (
        cell_count * cell_count
    )
    return energy, enstrophy


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def solve_navier_stokes_2d(
    nu: float,
    nx: int = 64,
    ny: int = 64,
    t_end: float = 1.0,
    nt: int = 1000,
    initial_vorticity: str = "-2*sin(x)*sin(y)",
) -> dict:
    """Solve 2D incompressible Navier-Stokes on [0, 2*pi]^2 with periodic boundaries.

    Vorticity-streamfunction form: ``omega_t + u*omega_x + v*omega_y =
    nu*laplacian(omega)``, with ``laplacian(psi) = -omega``, ``u = psi_y``,
    ``v = -psi_x``.  The Poisson solve and all spatial derivatives use FFT
    (Fourier spectral method); time integration is explicit RK2 (midpoint).
    ``initial_vorticity`` is a sympy expression in x and y (no Python eval).

    The grid is periodic with ``nx`` x ``ny`` points and spacing
    ``2*pi/nx``, ``2*pi/ny``.  Before stepping, the tool requires
    ``nu*dt/dx^2 <= 0.5`` and ``max|u|*dt/dx <= 0.5`` (``dx`` = smaller
    spacing, ``dt = t_end/nt``); otherwise it returns ``err`` and asks for a
    larger ``nt``.  ``nx``/``ny`` are limited to 8..256 with
    ``nx*ny <= 65536``, and ``nt <= 100000``.

    The full (t, y, x) vorticity history, grid and energy/enstrophy series are
    saved to .npz.  Returned arrays are downsampled to at most 50 x 50 spatial
    points and 20 time levels.  This is not a general CFD solver.
    """
    try:
        viscosity = _non_negative_float_param(nu, "nu")
        end_time = _positive_float_param(t_end, "t_end")
        n_x = _int_param(nx, "nx", _MIN_GRID, _MAX_GRID)
        n_y = _int_param(ny, "ny", _MIN_GRID, _MAX_GRID)
        n_t = _int_param(nt, "nt", 1, _MAX_STEPS)
        if n_x * n_y > _MAX_GRID_POINTS:
            raise ValueError(
                f"grid too large: nx*ny = {n_x * n_y} exceeds the limit of "
                f"{_MAX_GRID_POINTS} points; reduce nx or ny"
            )
        stored = (n_t + 1) * n_y * n_x
        if stored > _MAX_STORED_POINTS:
            raise ValueError(
                f"history too large: (nt+1)*ny*nx = {stored} stored points "
                f"exceeds the limit of {_MAX_STORED_POINTS}; reduce nt, nx or ny"
            )

        dx = _TWO_PI / n_x
        dy = _TWO_PI / n_y
        spacing = min(dx, dy)
        dt = end_time / n_t

        compiled = _compile_expr(initial_vorticity, "initial_vorticity")
        x = _TWO_PI * np.arange(n_x) / n_x
        y = _TWO_PI * np.arange(n_y) / n_y
        omega = _evaluate_on_grid_2d(compiled, x, y, "initial_vorticity")

        kx = _wavenumbers(n_x)
        ky = _wavenumbers(n_y)
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        k2_safe = k2.copy()
        k2_safe[0, 0] = 1.0

        # Velocity magnitude of the initial field drives the convective CFL check.
        omega_hat = np.fft.fft2(omega)
        psi_hat = _poisson_psi_hat(omega_hat, k2_safe)
        u0, v0 = _velocity(psi_hat, kx, ky)
        max_speed = float(np.max(np.sqrt(u0 * u0 + v0 * v0)))
        if not math.isfinite(max_speed):
            raise ValueError("initial vorticity produced a non-finite velocity field")

        viscous_number = viscosity * dt / (spacing * spacing)
        if viscous_number > 0.5:
            return err(
                "solve_navier_stokes_2d failed: explicit stepping is unstable "
                f"because nu*dt/dx^2 = {viscous_number:.6g} exceeds 0.5; "
                f"increase nt (currently {n_t}) or reduce t_end"
            )
        cfl = max_speed * dt / spacing
        if cfl > 0.5:
            return err(
                "solve_navier_stokes_2d failed: explicit stepping is unstable "
                f"because max|u|*dt/dx = {cfl:.6g} exceeds 0.5; "
                f"increase nt (currently {n_t}) or reduce t_end"
            )

        history = np.empty((n_t + 1, n_y, n_x), dtype=float)
        energy = np.empty(n_t + 1, dtype=float)
        enstrophy = np.empty(n_t + 1, dtype=float)
        history[0] = omega
        energy[0], enstrophy[0] = _diagnostics(omega, kx, ky, k2_safe)

        stopped_early = False
        completed_steps = 0
        for step in range(n_t):
            k1 = _rhs(omega, viscosity, kx, ky, k2, k2_safe)
            omega_mid = omega + 0.5 * dt * k1
            k2v = _rhs(omega_mid, viscosity, kx, ky, k2, k2_safe)
            omega = omega + dt * k2v
            if not np.all(np.isfinite(omega)):
                stopped_early = True
                break
            completed_steps = step + 1
            history[completed_steps] = omega
            energy[completed_steps], enstrophy[completed_steps] = _diagnostics(
                omega, kx, ky, k2_safe
            )

        if stopped_early:
            return err(
                "solve_navier_stokes_2d failed: non-finite vorticity appeared at "
                f"step {completed_steps + 1} of {n_t}; the explicit scheme blew "
                "up; increase nt or lower nu/t_end"
            )
        if completed_steps != n_t:
            return err(
                "solve_navier_stokes_2d failed: integration stopped at step "
                f"{completed_steps} of {n_t}"
            )
        if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(enstrophy))):
            return err(
                "solve_navier_stokes_2d failed: non-finite energy or enstrophy "
                "was produced; increase nt or lower nu/t_end"
            )

        t = np.linspace(0.0, end_time, n_t + 1)
        path = _new_output_path("navier_stokes_2d")
        np.savez(
            path,
            x=x,
            y=y,
            t=t,
            omega=history,
            energy=energy,
            enstrophy=enstrophy,
        )

        x_index = _downsample_indices(x.size, _MAX_SPACE_POINTS)
        y_index = _downsample_indices(y.size, _MAX_SPACE_POINTS)
        t_index = _downsample_indices(t.size, _MAX_TIME_POINTS)
        downsampled = bool(
            x_index.size < x.size or y_index.size < y.size or t_index.size < t.size
        )
        returned_omega = history[np.ix_(t_index, y_index, x_index)]

        return ok(
            {
                "x": x[x_index].tolist(),
                "y": y[y_index].tolist(),
                "t": t[t_index].tolist(),
                "omega": returned_omega.tolist(),
                "energy": energy[t_index].tolist(),
                "enstrophy": enstrophy[t_index].tolist(),
                "path": _saved_path(path),
                "shape": [int(size) for size in returned_omega.shape],
                "original_shape": [int(size) for size in history.shape],
                "downsampled": downsampled,
                "max": float(np.max(history)),
                "min": float(np.min(history)),
                "dx": float(dx),
                "dy": float(dy),
                "dt": float(dt),
                "nu": float(viscosity),
                "nx": n_x,
                "ny": n_y,
                "nt": n_t,
                "t_end": float(end_time),
                "scheme": "explicit RK2 (midpoint), Fourier spectral spatial operators",
                "stability": {
                    "viscous_number": float(viscous_number),
                    "cfl": float(cfl),
                    "max_initial_speed": float(max_speed),
                },
                "model": "2D incompressible vorticity-streamfunction, periodic",
                "limitations": (
                    "not a general CFD solver; no 3D, turbulence models, "
                    "or complex boundaries"
                ),
            }
        )
    except Exception as exc:
        return err(f"solve_navier_stokes_2d failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(solve_navier_stokes_2d)
