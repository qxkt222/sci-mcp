"""Linear and mixed-integer programming with scipy.optimize (linprog / milp)."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from sci_mcp.core import err, ok

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


ArrayLike = list[float] | tuple[float, ...] | np.ndarray
MatrixLike = list[list[float]] | tuple[tuple[float, ...], ...] | np.ndarray


def _as_vector(values: Any, name: str) -> np.ndarray:
    """Convert a JSON-style numeric list into a non-empty one-dimensional float array."""
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
    else:
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} must be a list of numbers")
        if not values:
            raise ValueError(f"{name} must not be empty")
        for item in values:
            if isinstance(item, bool) or not isinstance(item, numbers.Real):
                raise ValueError(f"{name} contains a non-numeric value: {item!r}")
        array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional list")
    return array


def _as_matrix(values: Any, name: str) -> np.ndarray:
    """Convert a JSON-style nested list into a two-dimensional float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-dimensional numeric array") from exc
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional nested list")
    return array


def _parse_constraints(
    a: Any, b: Any, n_vars: int, name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate one constraint block (matrix + right-hand side) against ``c``'s length."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError(f"{name} and the matching b must be provided together")
    matrix = _as_matrix(a, f"a_{name}")
    rhs = _as_vector(b, f"b_{name}")
    if matrix.shape[1] != n_vars:
        raise ValueError(
            f"a_{name} dimension mismatch: it has {matrix.shape[1]} columns "
            f"but c has {n_vars} entries"
        )
    if matrix.shape[0] != rhs.size:
        raise ValueError(
            f"a_{name} dimension mismatch: it has {matrix.shape[0]} rows "
            f"but b_{name} has {rhs.size} entries"
        )
    return matrix, rhs


def _parse_bounds(bounds: Any, n_vars: int) -> list[tuple[float | None, float | None]] | None:
    """Validate bounds into the [(lower, upper), ...] form used by linprog."""
    if bounds is None:
        return None
    if not isinstance(bounds, (list, tuple)):
        raise ValueError("bounds must be a list of [lower, upper] pairs or null")
    if len(bounds) != n_vars:
        raise ValueError(
            f"bounds must contain one [lower, upper] pair per variable "
            f"(got {len(bounds)} pairs for {n_vars} variables)"
        )
    parsed: list[tuple[float | None, float | None]] = []
    for index, pair in enumerate(bounds):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"bounds[{index}] must be a [lower, upper] pair")
        lower, upper = pair
        if lower is not None:
            if isinstance(lower, bool) or not isinstance(lower, numbers.Real):
                raise ValueError(f"bounds[{index}] lower bound must be a number or null")
            lower = float(lower)
        if upper is not None:
            if isinstance(upper, bool) or not isinstance(upper, numbers.Real):
                raise ValueError(f"bounds[{index}] upper bound must be a number or null")
            upper = float(upper)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"bounds[{index}] lower bound exceeds upper bound")
        parsed.append((lower, upper))
    return parsed


def _solver_payload(result: Any) -> dict[str, Any]:
    """Extract the common linprog/milp result fields into JSON-safe values."""
    raw_x = getattr(result, "x", None)
    x_values = (
        [] if raw_x is None else np.asarray(raw_x, dtype=float).reshape(-1).tolist()
    )
    raw_fun = getattr(result, "fun", None)
    fun = None if raw_fun is None else float(raw_fun)
    status = getattr(result, "status", -1)
    return {
        "x": x_values,
        "fun": fun,
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "status": int(status) if status is not None else -1,
    }


def _failure(operation: str, payload: dict[str, Any]) -> dict:
    """Build an ``err`` envelope that still carries the solver result fields."""
    # ``err`` already uses ``message`` for the error text, hence ``solver_message``.
    return err(
        f"{operation} failed: {payload['message']}",
        x=payload["x"],
        fun=payload["fun"],
        success=payload["success"],
        status=payload["status"],
        solver_message=payload["message"],
    )


def linear_program(
    c: list[float],
    a_ub: list[list[float]] | None = None,
    b_ub: list[float] | None = None,
    a_eq: list[list[float]] | None = None,
    b_eq: list[float] | None = None,
    bounds: list[list[float | None]] | None = None,
    method: str = "highs",
) -> dict:
    """Minimize ``c @ x`` subject to linear inequality/equality constraints.

    ``fun`` is the minimum objective value; negate ``c`` to maximize. Default
    bounds are scipy's ``(0, None)`` (all variables non-negative).
    """
    try:
        objective = _as_vector(c, "c")
        n_vars = int(objective.size)

        inequality = _parse_constraints(a_ub, b_ub, n_vars, "ub")
        equality = _parse_constraints(a_eq, b_eq, n_vars, "eq")
        parsed_bounds = _parse_bounds(bounds, n_vars)

        if not isinstance(method, str) or not method.strip():
            return err("linear_program failed: method must be a non-empty string")
        selected_method = method.strip().lower()

        result = optimize.linprog(
            objective,
            A_ub=None if inequality is None else inequality[0],
            b_ub=None if inequality is None else inequality[1],
            A_eq=None if equality is None else equality[0],
            b_eq=None if equality is None else equality[1],
            bounds=parsed_bounds,
            method=selected_method,
        )

        payload = _solver_payload(result)
        if not payload["success"]:
            return _failure("linear_program", payload)
        return ok(payload)
    except Exception as exc:
        return err(f"linear_program failed: {exc}")


def _parse_integrality(integrality: Any, n_vars: int) -> np.ndarray:
    """Validate integrality flags: 1 = integer variable, 0 = continuous."""
    if integrality is None:
        return np.ones(n_vars, dtype=int)
    if not isinstance(integrality, (list, tuple, np.ndarray)):
        raise ValueError("integrality must be a list of 0/1 flags or null")
    if len(integrality) != n_vars:
        raise ValueError(
            f"integrality must contain one flag per variable "
            f"(got {len(integrality)} flags for {n_vars} variables)"
        )
    flags: list[int] = []
    for index, item in enumerate(integrality):
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise ValueError(f"integrality[{index}] must be 0 (continuous) or 1 (integer)")
        flag = int(item)
        if flag not in (0, 1):
            raise ValueError(f"integrality[{index}] must be 0 (continuous) or 1 (integer)")
        flags.append(flag)
    return np.asarray(flags, dtype=int)


def integer_program(
    c: list[float],
    integrality: list[int] | None = None,
    a_ub: list[list[float]] | None = None,
    b_ub: list[float] | None = None,
    a_eq: list[list[float]] | None = None,
    b_eq: list[float] | None = None,
    bounds: list[list[float | None]] | None = None,
    method: str = "highs",
) -> dict:
    """Minimize ``c @ x`` with scipy.optimize.milp (mixed-integer linear programming).

    ``integrality`` uses 1 for integer and 0 for continuous variables and
    defaults to all-integer. ``fun`` is the minimum objective value; negate
    ``c`` to maximize. Default bounds are ``0 <= x <= inf``.
    """
    try:
        objective = _as_vector(c, "c")
        n_vars = int(objective.size)

        inequality = _parse_constraints(a_ub, b_ub, n_vars, "ub")
        equality = _parse_constraints(a_eq, b_eq, n_vars, "eq")
        parsed_bounds = _parse_bounds(bounds, n_vars)
        flags = _parse_integrality(integrality, n_vars)

        if not isinstance(method, str) or not method.strip():
            return err("integer_program failed: method must be a non-empty string")
        selected_method = method.strip().lower()
        if selected_method != "highs":
            return err(
                f"integer_program failed: unsupported method {method!r}; "
                "scipy.optimize.milp only provides the 'highs' solver"
            )

        constraints = []
        if inequality is not None:
            constraints.append(
                optimize.LinearConstraint(inequality[0], -np.inf, inequality[1])
            )
        if equality is not None:
            constraints.append(
                optimize.LinearConstraint(equality[0], equality[1], equality[1])
            )

        milp_bounds = None
        if parsed_bounds is not None:
            lower = np.asarray(
                [-np.inf if pair[0] is None else pair[0] for pair in parsed_bounds],
                dtype=float,
            )
            upper = np.asarray(
                [np.inf if pair[1] is None else pair[1] for pair in parsed_bounds],
                dtype=float,
            )
            milp_bounds = optimize.Bounds(lower, upper)

        # milp has no ``method`` argument (HiGHS is the only backend), so the
        # selected method is validated above and not forwarded.
        result = optimize.milp(
            objective,
            integrality=flags,
            bounds=milp_bounds,
            constraints=tuple(constraints) if constraints else None,
        )

        payload = _solver_payload(result)
        if not payload["success"]:
            return _failure("integer_program", payload)
        return ok(payload)
    except Exception as exc:
        return err(f"integer_program failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the linear and integer programming tools on the MCP server."""
    mcp.tool()(linear_program)
    mcp.tool()(integer_program)
