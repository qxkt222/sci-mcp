"""Numerical tools: ODE integration, optimization, linear algebra, root finding, FFT, interpolation."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from mcp.server.mcpserver import MCPServer
from scipy import integrate, interpolate, optimize

from ..core import load_array
from sci_mcp.core import err, ok

# Tight tolerances so that the returned trajectory is close to machine precision
# for well-conditioned problems (the scipy defaults, rtol=1e-3, are too coarse
# for e.g. reproducing exp(-t) with a 1e-6 relative-error check).
_ODE_RTOL = 1e-10
_ODE_ATOL = 1e-12

_SUPPORTED_MATRIX_OPS = (
    "det",
    "inv",
    "eig",
    "eigvals",
    "svd",
    "solve",
    "lstsq",
    "matmul",
    "rank",
    "norm",
    "transpose",
)

_SUPPORTED_INTERP_KINDS = ("linear", "quadratic", "cubic")


def _param_symbols(params: dict[str, float] | None) -> tuple[list[sp.Symbol], list[float]]:
    """Turn a JSON parameter dict into an ordered list of sympy symbols and float values."""
    if not params:
        return [], []
    symbols: list[sp.Symbol] = []
    values: list[float] = []
    for name, value in params.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"invalid parameter name: {name!r}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"parameter {name!r} must be a real number") from exc
        symbols.append(sp.Symbol(name))
        values.append(numeric)
    return symbols, values


def _call_scalar(func: Any, args: tuple[Any, ...], context: str) -> float:
    """Evaluate a lambdified expression and coerce the value to a finite-ish float."""
    value = func(*args)
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{context} must evaluate to a single real number")
    return float(array.reshape(-1)[0])


def solve_ode(
    rhs: str | list[str],
    y0: list[float],
    t_span: list[float],
    t_eval: list[float] | None = None,
    params: dict[str, float] | None = None,
    method: str = "RK45",
) -> dict:
    """Solve an initial-value ODE system with scipy solve_ivp.

    ``rhs`` is one expression (state variable ``y``) or a list of expressions
    (state variables ``y0, y1, ...`` in the same order as ``y0``). The time
    variable is ``t``; keys of ``params`` are injected as extra constants.
    """
    try:
        if isinstance(rhs, str):
            expressions = [rhs]
        elif isinstance(rhs, (list, tuple)):
            expressions = list(rhs)
        else:
            return err("solve_ode failed: rhs must be a string or a list of strings")

        if not expressions:
            return err("solve_ode failed: rhs must contain at least one expression")
        if any(not isinstance(item, str) for item in expressions):
            return err("solve_ode failed: every rhs entry must be a string")

        n_states = len(expressions)
        if not isinstance(y0, (list, tuple)) or len(y0) != n_states:
            return err(
                "solve_ode failed: y0 length must match the number of rhs expressions "
                f"(got {len(y0) if isinstance(y0, (list, tuple)) else 'non-list'} for {n_states})"
            )
        if not isinstance(t_span, (list, tuple)) or len(t_span) != 2:
            return err("solve_ode failed: t_span must be a two-element list")

        t = sp.Symbol("t")
        if n_states == 1:
            state_symbols = [sp.Symbol("y")]
        else:
            state_symbols = [sp.Symbol(f"y{i}") for i in range(n_states)]
        param_symbols, param_values = _param_symbols(params)

        reserved = {symbol.name for symbol in state_symbols} | {"t"}
        clashes = sorted({symbol.name for symbol in param_symbols} & reserved)
        if clashes:
            return err(f"solve_ode failed: parameter name clashes with a reserved symbol: {clashes}")

        # Parse with an explicit local namespace so that parameter names which
        # collide with sympy's built-in functions/constants (``beta``,
        # ``gamma``, ``N``, ...) resolve to Symbols instead of FunctionClass
        # objects. ``locals`` takes precedence over sympy's global namespace.
        local_dict: dict[str, sp.Symbol] = {
            symbol.name: symbol for symbol in [t, *state_symbols, *param_symbols]
        }

        try:
            parsed = [sp.sympify(item, locals=local_dict) for item in expressions]
        except Exception as exc:
            return err(f"solve_ode failed: could not parse rhs expression: {exc}")

        allowed = {t, *state_symbols, *param_symbols}
        for original, expression in zip(expressions, parsed):
            unknown = expression.free_symbols - allowed
            if unknown:
                names = ", ".join(sorted(symbol.name for symbol in unknown))
                return err(f"solve_ode failed: unknown symbol(s) in expression {original!r}: {names}")

        arguments = [t, *state_symbols, *param_symbols]
        try:
            compiled = sp.lambdify(arguments, parsed[0] if n_states == 1 else parsed, "numpy")
        except Exception as exc:
            return err(f"solve_ode failed: could not compile rhs expression: {exc}")

        if n_states == 1:

            def fun(time: float, state: np.ndarray) -> np.ndarray:
                value = compiled(time, state[0], *param_values)
                return np.atleast_1d(np.asarray(value, dtype=float))

        else:

            def fun(time: float, state: np.ndarray) -> np.ndarray:
                values = compiled(time, *state, *param_values)
                return np.asarray(values, dtype=float)

        y0_array = np.asarray(y0, dtype=float)
        if y0_array.ndim != 1:
            return err("solve_ode failed: y0 must be a one-dimensional list of numbers")

        evaluation_points = None
        if t_eval is not None:
            if not isinstance(t_eval, (list, tuple)):
                return err("solve_ode failed: t_eval must be a list of numbers or null")
            if len(t_eval) > 0:
                evaluation_points = np.asarray(t_eval, dtype=float)

        solution = integrate.solve_ivp(
            fun,
            (float(t_span[0]), float(t_span[1])),
            y0_array,
            method=method,
            t_eval=evaluation_points,
            rtol=_ODE_RTOL,
            atol=_ODE_ATOL,
        )

        if not solution.success:
            return err(
                f"solve_ode failed: solver did not converge: {solution.message}",
                solver_message=str(solution.message),
                status=int(solution.status),
            )

        if n_states == 1:
            y_result: Any = solution.y[0].tolist()
        else:
            y_result = solution.y.tolist()

        return ok({"t": solution.t.tolist(), "y": y_result})
    except Exception as exc:
        return err(f"solve_ode failed: {exc}")


def optimize_minimize(
    objective: str,
    x0: list[float],
    bounds: list[list[float]] | None = None,
    method: str | None = None,
    params: dict[str, float] | None = None,
    constraints: list[dict] | None = None,
) -> dict:
    """Minimize a scalar expression with scipy.optimize.minimize.

    Variables are named ``x0, x1, ...`` in the order of ``x0``. When ``method``
    is omitted, SLSQP is used if ``constraints`` is given, otherwise L-BFGS-B
    is used if ``bounds`` is given, otherwise scipy's default (BFGS).

    Each constraint is ``{"kind": "ineq"|"eq", "expr": "..."}`` where ``expr``
    is a sympy expression parsed with the variables and ``params``; ``ineq``
    means ``expr >= 0`` and ``eq`` means ``expr == 0``.
    """
    try:
        if not isinstance(x0, (list, tuple)) or len(x0) == 0:
            return err("optimize_minimize failed: x0 must be a non-empty list of numbers")

        variables = [sp.Symbol(f"x{i}") for i in range(len(x0))]
        param_symbols, param_values = _param_symbols(params)

        clashes = sorted({symbol.name for symbol in param_symbols} & {symbol.name for symbol in variables})
        if clashes:
            return err(f"optimize_minimize failed: parameter name clashes with a variable: {clashes}")

        # Explicit locals override sympy built-ins named like the parameters
        # (e.g. ``beta``), so they parse as Symbols rather than functions.
        local_dict: dict[str, sp.Symbol] = {
            symbol.name: symbol for symbol in [*variables, *param_symbols]
        }

        try:
            expression = sp.sympify(objective, locals=local_dict)
        except Exception as exc:
            return err(f"optimize_minimize failed: could not parse objective: {exc}")

        allowed = {*variables, *param_symbols}
        unknown = expression.free_symbols - allowed
        if unknown:
            names = ", ".join(sorted(symbol.name for symbol in unknown))
            return err(f"optimize_minimize failed: unknown symbol(s) in objective: {names}")

        try:
            compiled = sp.lambdify([*variables, *param_symbols], expression, "numpy")
        except Exception as exc:
            return err(f"optimize_minimize failed: could not compile objective: {exc}")

        start = np.asarray(x0, dtype=float)
        if start.ndim != 1:
            return err("optimize_minimize failed: x0 must be a one-dimensional list")

        parsed_bounds = None
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != start.size:
                return err(
                    "optimize_minimize failed: bounds must have one [lower, upper] pair per variable"
                )
            parsed_bounds = []
            for pair in bounds:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    return err("optimize_minimize failed: each bounds entry must be [lower, upper]")
                lower, upper = pair
                parsed_bounds.append(
                    (
                        None if lower is None else float(lower),
                        None if upper is None else float(upper),
                    )
                )

        parsed_constraints: list[dict[str, Any]] | None = None
        if constraints is not None:
            if not isinstance(constraints, (list, tuple)):
                return err("optimize_minimize failed: constraints must be a list of constraint dicts")
            parsed_constraints = []
            for index, item in enumerate(constraints):
                if not isinstance(item, dict):
                    return err(f"optimize_minimize failed: constraint {index} must be a dict")
                kind = item.get("kind")
                if kind not in ("ineq", "eq"):
                    return err(
                        f"optimize_minimize failed: unknown constraint kind {kind!r} "
                        f"for constraint {index}; expected 'ineq' or 'eq'"
                    )
                raw_expr = item.get("expr")
                if not isinstance(raw_expr, str) or not raw_expr.strip():
                    return err(
                        f"optimize_minimize failed: constraint {index} must have a non-empty "
                        "string 'expr'"
                    )
                try:
                    constraint_expr = sp.sympify(raw_expr, locals=local_dict)
                except Exception as exc:
                    return err(
                        f"optimize_minimize failed: could not parse constraint {index}: {exc}"
                    )
                constraint_unknown = constraint_expr.free_symbols - allowed
                if constraint_unknown:
                    names = ", ".join(sorted(symbol.name for symbol in constraint_unknown))
                    return err(
                        "optimize_minimize failed: unknown symbol(s) in "
                        f"constraint {index}: {names}"
                    )
                try:
                    compiled_constraint = sp.lambdify(
                        [*variables, *param_symbols], constraint_expr, "numpy"
                    )
                except Exception as exc:
                    return err(
                        f"optimize_minimize failed: could not compile constraint {index}: {exc}"
                    )

                def constraint_fun(
                    vector: np.ndarray,
                    _compiled: Any = compiled_constraint,
                    _context: str = f"constraint {index}",
                ) -> float:
                    return _call_scalar(_compiled, (*vector, *param_values), _context)

                parsed_constraints.append({"type": kind, "fun": constraint_fun})

        selected_method = method
        if selected_method is None or selected_method == "":
            if parsed_constraints:
                selected_method = "SLSQP"
            elif parsed_bounds is not None:
                selected_method = "L-BFGS-B"
            else:
                selected_method = None

        def fun(vector: np.ndarray) -> float:
            return _call_scalar(compiled, (*vector, *param_values), "objective")

        result = optimize.minimize(
            fun,
            start,
            method=selected_method,
            bounds=parsed_bounds,
            constraints=parsed_constraints or None,
        )

        payload = {
            "x": np.asarray(result.x, dtype=float).tolist(),
            "fun": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev) if result.nfev is not None else 0,
        }

        if not result.success:
            # ``err`` already owns the ``message`` keyword, so expose the solver
            # message under ``solver_message`` alongside the other result fields.
            return err(
                f"optimize_minimize failed: {result.message}",
                x=payload["x"],
                fun=payload["fun"],
                success=payload["success"],
                nfev=payload["nfev"],
                solver_message=payload["message"],
            )

        return ok(payload)
    except Exception as exc:
        return err(f"optimize_minimize failed: {exc}")


def _as_2d_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional nested list")
    return matrix


def _as_rhs(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required for this operation")
    vector = np.asarray(value, dtype=float)
    if vector.ndim not in (1, 2):
        raise ValueError(f"{name} must be a vector or a matrix")
    return vector


def matrix_compute(
    operation: str,
    a: list[list[float]] | str,
    b: list[list[float]] | list[float] | str | None = None,
) -> dict:
    """Perform a dense linear-algebra operation on matrix ``a`` (and optional ``b``)."""
    try:
        if not isinstance(operation, str):
            return err("matrix_compute failed: operation must be a string")
        op = operation.strip().lower()
        if op not in _SUPPORTED_MATRIX_OPS:
            return err(
                f"matrix_compute failed: unknown operation {operation!r}; "
                f"supported operations are {list(_SUPPORTED_MATRIX_OPS)}"
            )

        matrix = _as_2d_matrix(load_array(a), "a")
        rhs_input = None if b is None else load_array(b)

        if op == "det":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("det requires a square matrix")
            return ok(float(np.linalg.det(matrix)))

        if op == "inv":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("inv requires a square matrix")
            return ok(np.linalg.inv(matrix).tolist())

        if op == "eig":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("eig requires a square matrix")
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            # numpy 2.x may return a complex dtype even when every eigenvalue is
            # real; collapse to real floats in that case instead of emitting
            # {"real": ..., "imag": 0.0} pairs for a real spectrum.
            if np.iscomplexobj(eigenvalues) and np.all(eigenvalues.imag == 0):
                eigenvalues = eigenvalues.real
                eigenvectors = eigenvectors.real
            # Each row of the transpose is the eigenvector paired with eigenvalues[i].
            return ok(
                {
                    "eigenvalues": eigenvalues.tolist(),
                    "eigenvectors": eigenvectors.T.tolist(),
                }
            )

        if op == "eigvals":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("eigvals requires a square matrix")
            eigenvalues = np.linalg.eigvals(matrix)
            if np.iscomplexobj(eigenvalues) and np.all(eigenvalues.imag == 0):
                eigenvalues = eigenvalues.real
            return ok(eigenvalues.tolist())

        if op == "svd":
            u, singular_values, vh = np.linalg.svd(matrix)
            return ok({"u": u.tolist(), "s": singular_values.tolist(), "vh": vh.tolist()})

        if op == "solve":
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("solve requires a square coefficient matrix")
            rhs = _as_rhs(rhs_input, "b")
            if rhs.shape[0] != matrix.shape[0]:
                raise ValueError(
                    f"solve dimension mismatch: a has {matrix.shape[0]} rows but b has {rhs.shape[0]}"
                )
            return ok(np.linalg.solve(matrix, rhs).tolist())

        if op == "lstsq":
            rhs = _as_rhs(rhs_input, "b")
            solution, residuals, rank, singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
            return ok(
                {
                    "x": solution.tolist(),
                    "residuals": residuals.tolist(),
                    "rank": int(rank),
                    "singular_values": singular_values.tolist(),
                }
            )

        if op == "matmul":
            rhs = _as_rhs(rhs_input, "b")
            if matrix.shape[1] != rhs.shape[0]:
                raise ValueError(
                    f"matmul dimension mismatch: a has {matrix.shape[1]} columns "
                    f"but b has {rhs.shape[0]} rows"
                )
            return ok(np.matmul(matrix, rhs).tolist())

        if op == "rank":
            return ok(int(np.linalg.matrix_rank(matrix)))

        if op == "norm":
            return ok(float(np.linalg.norm(matrix)))

        # transpose
        return ok(np.transpose(matrix).tolist())
    except Exception as exc:
        return err(f"matrix_compute failed: {exc}")


def find_roots(
    expr: str,
    variable: str = "x",
    bracket: list[float] | None = None,
    x0: float | None = None,
    params: dict[str, float] | None = None,
) -> dict:
    """Find a scalar root of an expression using brentq (bracket) or Newton's method (x0)."""
    try:
        if not isinstance(expr, str):
            return err("find_roots failed: expr must be a string")
        if not isinstance(variable, str) or not variable.isidentifier():
            return err("find_roots failed: variable must be a valid identifier")

        symbol = sp.Symbol(variable)
        param_symbols, param_values = _param_symbols(params)

        clashes = sorted({item.name for item in param_symbols} & {symbol.name})
        if clashes:
            return err(f"find_roots failed: parameter name clashes with the variable: {clashes}")

        # Include the variable and every parameter explicitly so names that
        # shadow sympy built-ins (``beta``, ``N``, ...) become Symbols.
        local_dict: dict[str, sp.Symbol] = {
            item.name: item for item in [symbol, *param_symbols]
        }

        try:
            expression = sp.sympify(expr, locals=local_dict)
        except Exception as exc:
            return err(f"find_roots failed: could not parse expression: {exc}")

        allowed = {symbol, *param_symbols}
        unknown = expression.free_symbols - allowed
        if unknown:
            names = ", ".join(sorted(item.name for item in unknown))
            return err(f"find_roots failed: unknown symbol(s) in expression: {names}")

        try:
            compiled = sp.lambdify([symbol, *param_symbols], expression, "numpy")
        except Exception as exc:
            return err(f"find_roots failed: could not compile expression: {exc}")

        def fun(value: float) -> float:
            return _call_scalar(compiled, (value, *param_values), "expression")

        if bracket is not None:
            if not isinstance(bracket, (list, tuple)) or len(bracket) != 2:
                return err("find_roots failed: bracket must be a two-element [lower, upper] list")
            lower, upper = float(bracket[0]), float(bracket[1])
            root = optimize.brentq(fun, lower, upper)
            return ok({"roots": [float(root)], "method": "brentq"})

        if x0 is not None:
            root = optimize.newton(fun, float(x0))
            return ok({"roots": [float(root)], "method": "newton"})

        return err("find_roots failed: provide bracket or x0")
    except Exception as exc:
        return err(f"find_roots failed: {exc}")


def fft_analysis(signal: list[float] | str, sample_rate: float = 1.0) -> dict:
    """Compute the single-sided amplitude spectrum of a real signal via numpy rfft."""
    try:
        samples = load_array(signal)
        try:
            rate = float(sample_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_rate must be a positive number") from exc
        if rate <= 0:
            return err("fft_analysis failed: sample_rate must be greater than 0")

        if samples.ndim != 1 or samples.size == 0:
            return err("fft_analysis failed: signal must be a non-empty one-dimensional list")
        if not np.all(np.isfinite(samples)):
            return err("fft_analysis failed: signal must contain only finite values")

        count = samples.size
        spectrum = np.fft.rfft(samples)
        frequencies = np.fft.rfftfreq(count, d=1.0 / rate)
        amplitudes = np.abs(spectrum) * 2.0 / count
        amplitudes[0] /= 2.0  # DC component is not doubled
        if count % 2 == 0:
            # Even-length signals have a Nyquist bin at the last rfft index;
            # like DC it is not doubled in the single-sided amplitude spectrum.
            amplitudes[-1] /= 2.0

        payload: dict[str, Any] = {
            "frequencies": frequencies.tolist(),
            "amplitudes": amplitudes.tolist(),
        }

        # A purely zero (or otherwise non-finite) spectrum has no meaningful
        # dominant bin; returning 0.0 would falsely suggest a DC component.
        reason: str | None = None
        if amplitudes.size == 0:
            reason = "no dominant frequency (empty spectrum)"
        else:
            peak = float(np.max(amplitudes))
            if not np.isfinite(peak):
                reason = "no dominant frequency (amplitudes are not finite)"
            elif peak <= 0.0:
                reason = "no dominant frequency (all amplitudes are zero)"

        if reason is None:
            payload["dominant_frequency"] = float(frequencies[int(np.argmax(amplitudes))])
        else:
            payload["dominant_frequency"] = None
            payload["reason"] = reason
        return ok(payload)
    except Exception as exc:
        return err(f"fft_analysis failed: {exc}")


def interpolate_1d(
    x: list[float] | str,
    y: list[float] | str,
    xi: list[float] | str,
    kind: str = "linear",
) -> dict:
    """Interpolate y(x) at points xi using scipy interp1d (linear/quadratic/cubic)."""
    try:
        if not isinstance(kind, str) or kind.lower() not in _SUPPORTED_INTERP_KINDS:
            return err(
                f"interpolate_1d failed: unknown kind {kind!r}; "
                f"supported kinds are {list(_SUPPORTED_INTERP_KINDS)}"
            )
        selected_kind = kind.lower()

        x_array = load_array(x)
        y_array = load_array(y)
        xi_array = load_array(xi)
        if x_array.ndim != 1 or y_array.ndim != 1 or xi_array.ndim != 1:
            return err("interpolate_1d failed: x, y and xi must be one-dimensional lists")
        if x_array.size != y_array.size:
            return err(
                f"interpolate_1d failed: x and y must have the same length "
                f"(got {x_array.size} and {y_array.size})"
            )
        if x_array.size < 2:
            return err("interpolate_1d failed: at least two data points are required")
        if xi_array.size == 0:
            return err("interpolate_1d failed: xi must be a non-empty list of numbers")

        interpolator = interpolate.interp1d(
            x_array,
            y_array,
            kind=selected_kind,
            bounds_error=True,
        )
        yi = interpolator(xi_array)
        return ok({"xi": xi_array.tolist(), "yi": np.asarray(yi, dtype=float).tolist()})
    except Exception as exc:
        return err(f"interpolate_1d failed: {exc}")


_SUPPORTED_GLOBAL_METHODS = (
    "differential_evolution",
    "dual_annealing",
    "basinhopping",
)


def _parse_global_bounds(
    bounds: Any, n_variables: int | None
) -> tuple[list[tuple[float, float]], int]:
    """Validate and convert bounds into a list of finite (lower, upper) tuples."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) == 0:
        raise ValueError("bounds must be a non-empty list of [lower, upper] pairs")
    if n_variables is not None and len(bounds) != n_variables:
        raise ValueError(
            f"bounds must have one [lower, upper] pair per variable "
            f"(got {len(bounds)} pairs for {n_variables} variables)"
        )
    parsed: list[tuple[float, float]] = []
    for index, pair in enumerate(bounds):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"bounds entry {index} must be [lower, upper]")
        lower, upper = pair
        try:
            lower_value = float(lower)
            upper_value = float(upper)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounds entry {index} must contain real numbers") from exc
        if not (np.isfinite(lower_value) and np.isfinite(upper_value)):
            raise ValueError(f"bounds entry {index} must contain finite values")
        if lower_value > upper_value:
            raise ValueError(
                f"bounds entry {index} has lower > upper ({lower_value} > {upper_value})"
            )
        parsed.append((lower_value, upper_value))
    return parsed, len(parsed)


def global_optimize(
    objective: str,
    bounds: list[list[float]] | None = None,
    method: str = "differential_evolution",
    x0: list[float] | None = None,
    maxiter: int = 1000,
    seed: int | None = None,
    params: dict[str, float] | None = None,
) -> dict:
    """Minimize a scalar expression with a global optimizer.

    Variables are named ``x0, x1, ...`` in the order of ``bounds`` (or ``x0``).
    ``method`` is ``differential_evolution`` or ``dual_annealing`` (both require
    finite ``bounds``) or ``basinhopping`` (requires a starting point ``x0``).
    Keys of ``params`` are injected as additional symbolic constants.
    """
    try:
        if not isinstance(method, str) or method not in _SUPPORTED_GLOBAL_METHODS:
            return err(
                f"global_optimize failed: unknown method {method!r}; "
                f"supported methods are {list(_SUPPORTED_GLOBAL_METHODS)}"
            )
        selected_method = method

        if isinstance(maxiter, bool) or not isinstance(maxiter, int):
            return err("global_optimize failed: maxiter must be an integer")
        if maxiter < 1 or maxiter > 100000:
            return err("global_optimize failed: maxiter must be between 1 and 100000")

        if selected_method in ("differential_evolution", "dual_annealing") and bounds is None:
            return err(f"global_optimize failed: {selected_method} requires bounds")
        if selected_method == "basinhopping" and x0 is None:
            return err("global_optimize failed: basinhopping requires x0")

        start: np.ndarray | None = None
        if x0 is not None:
            if not isinstance(x0, (list, tuple)) or len(x0) == 0:
                return err("global_optimize failed: x0 must be a non-empty list of numbers")
            try:
                start = np.asarray(x0, dtype=float)
            except (TypeError, ValueError) as exc:
                return err(f"global_optimize failed: x0 must contain real numbers: {exc}")
            if start.ndim != 1:
                return err("global_optimize failed: x0 must be a one-dimensional list")
            if not np.all(np.isfinite(start)):
                return err("global_optimize failed: x0 must contain finite values")

        parsed_bounds: list[tuple[float, float]] | None = None
        if bounds is not None:
            try:
                parsed_bounds, n_variables = _parse_global_bounds(
                    bounds, None if start is None else int(start.size)
                )
            except ValueError as exc:
                return err(f"global_optimize failed: {exc}")
        elif start is not None:
            n_variables = int(start.size)
        else:  # pragma: no cover - guarded by the method requirement above
            return err("global_optimize failed: provide bounds or x0")

        if selected_method == "basinhopping" and start is None:  # pragma: no cover
            return err("global_optimize failed: basinhopping requires x0")

        variables = [sp.Symbol(f"x{i}") for i in range(n_variables)]
        param_symbols, param_values = _param_symbols(params)

        clashes = sorted(
            {symbol.name for symbol in param_symbols} & {symbol.name for symbol in variables}
        )
        if clashes:
            return err(f"global_optimize failed: parameter name clashes with a variable: {clashes}")

        # Explicit locals make parameter names that shadow sympy built-ins
        # (``beta``, ``N``, ...) parse as Symbols rather than functions.
        local_dict: dict[str, sp.Symbol] = {
            symbol.name: symbol for symbol in [*variables, *param_symbols]
        }

        try:
            expression = sp.sympify(objective, locals=local_dict)
        except Exception as exc:
            return err(f"global_optimize failed: could not parse objective: {exc}")

        allowed = {*variables, *param_symbols}
        unknown = expression.free_symbols - allowed
        if unknown:
            names = ", ".join(sorted(symbol.name for symbol in unknown))
            return err(f"global_optimize failed: unknown symbol(s) in objective: {names}")

        try:
            compiled = sp.lambdify([*variables, *param_symbols], expression, "numpy")
        except Exception as exc:
            return err(f"global_optimize failed: could not compile objective: {exc}")

        def fun(vector: Any) -> float:
            values = np.asarray(vector, dtype=float).reshape(-1)
            return _call_scalar(compiled, (*values, *param_values), "objective")

        if selected_method == "differential_evolution":
            # A deliberately larger population than scipy's default (15) makes
            # the search robust on rugged landscapes such as Rastrigin, where
            # the default population converges to a local minimum under a
            # fixed seed.
            result = optimize.differential_evolution(
                fun, parsed_bounds, maxiter=maxiter, seed=seed, popsize=30
            )
        elif selected_method == "dual_annealing":
            result = optimize.dual_annealing(
                fun, parsed_bounds, maxiter=maxiter, seed=seed
            )
        else:
            result = optimize.basinhopping(fun, start, niter=maxiter, seed=seed)

        payload = {
            "x": np.asarray(result.x, dtype=float).tolist(),
            "fun": float(result.fun),
            "success": bool(getattr(result, "success", True)),
            "message": str(getattr(result, "message", "")),
            "nfev": int(getattr(result, "nfev", 0) or 0),
            "method": selected_method,
        }

        if not payload["success"]:
            # ``err`` already owns the ``message`` keyword, so expose the solver
            # message under ``solver_message`` alongside the result fields.
            return err(
                f"global_optimize failed: {payload['message']}",
                x=payload["x"],
                fun=payload["fun"],
                success=payload["success"],
                nfev=payload["nfev"],
                method=payload["method"],
                solver_message=payload["message"],
            )

        return ok(payload)
    except Exception as exc:
        return err(f"global_optimize failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(solve_ode)
    mcp.tool()(optimize_minimize)
    mcp.tool()(matrix_compute)
    mcp.tool()(find_roots)
    mcp.tool()(fft_analysis)
    mcp.tool()(interpolate_1d)
    mcp.tool()(global_optimize)
