"""Nonlinear mixed models: penalized least-squares approximation (NM-).

This module provides a deliberately limited approximation to a nonlinear
mixed-effects model with a single random intercept per group. It is *not* a
full NLME marginal-likelihood fit: there is no integration over the random
effects, no variance-component estimation, and uncertainty is not propagated.
Every public function returns an ``ok()``/``err()`` envelope and never raises.
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sympy
from scipy.optimize import least_squares

from ..core import err, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_TABLE_ROWS = 1_000_000

_NOTE = (
    "penalized least-squares approximation; not a full NLME "
    "marginal-likelihood fit"
)

_RESERVED_SYMBOL = "x"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_input_path(raw: Any) -> Path:
    """Resolve an input file path (relative paths are anchored at output_dir())."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"file not found: {candidate}")
    if candidate.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"file exceeds the 50 MB limit ({candidate.stat().st_size} bytes)"
        )
    return candidate


def _load_table(data: Any) -> pd.DataFrame:
    """Build a DataFrame from a ``{column: list}`` mapping or ``@<path>.csv``."""
    if isinstance(data, dict):
        if not data:
            raise ValueError("data must not be empty")
        columns: dict[str, list[Any]] = {}
        lengths: set[int] = set()
        for name, values in data.items():
            if not isinstance(name, str) or not name:
                raise ValueError("data column names must be non-empty strings")
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"data[{name!r}] must be a list of values")
            lengths.add(len(values))
            columns[name] = list(values)
        if len(lengths) != 1:
            raise ValueError("all data columns must have the same length")
        if lengths == {0}:
            raise ValueError("data must not be empty")
        frame = pd.DataFrame(columns)
    elif isinstance(data, str):
        if not data.startswith("@"):
            raise ValueError("data string values must be '@<path>.csv' references")
        path = _resolve_input_path(data[1:])
        if path.suffix.lower() != ".csv":
            raise ValueError("data file references must point to a .csv file")
        frame = pd.read_csv(path)
        if len(frame) > _MAX_TABLE_ROWS:
            raise ValueError(
                f"CSV has {len(frame)} rows; the limit is {_MAX_TABLE_ROWS}"
            )
    else:
        raise ValueError("data must be a dict of columns or an '@<path>.csv' reference")

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("data must contain at least one row and one column")
    return frame


def _parse_initial(p0: Any) -> tuple[list[str], np.ndarray]:
    """Validate ``p0`` and return ordered parameter names plus initial values."""
    if not isinstance(p0, dict) or not p0:
        raise ValueError("p0 must be a non-empty dict of parameter names to values")
    names: list[str] = []
    values: list[float] = []
    for name, raw_value in p0.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("p0 parameter names must be non-empty strings")
        if not name.isidentifier():
            raise ValueError(f"p0 parameter name {name!r} is not a valid identifier")
        if name == _RESERVED_SYMBOL:
            raise ValueError(
                "p0 must not define 'x'; the independent variable is named x"
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Real):
            raise ValueError(f"p0[{name!r}] must be a real number")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"p0[{name!r}] must be finite")
        names.append(name)
        values.append(value)
    return names, np.asarray(values, dtype=float)


def _compile_model(model: Any, param_names: list[str]) -> Any:
    """Parse the model with sympy and return a numpy lambdified callable."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty expression string")
    try:
        expression = sympy.sympify(model)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"model could not be parsed as a sympy expression: {exc}")
    if not isinstance(expression, sympy.Expr):
        raise ValueError("model must be a scalar expression")

    x_symbol = sympy.Symbol(_RESERVED_SYMBOL)
    allowed = {x_symbol}
    allowed.update(sympy.Symbol(name) for name in param_names)
    unknown = sorted(str(symbol) for symbol in expression.free_symbols - allowed)
    if unknown:
        raise ValueError(
            "model contains unknown symbol(s): "
            + ", ".join(unknown)
            + f"; allowed symbols are x and {', '.join(param_names)}"
        )

    arguments = (x_symbol,) + tuple(sympy.Symbol(name) for name in param_names)
    try:
        return sympy.lambdify(arguments, expression, modules="numpy")
    except Exception as exc:  # pragma: no cover - defensive only
        raise ValueError(f"model could not be compiled: {exc}")


def _predict(func: Any, x_values: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate the model on all observations, broadcasting constant models."""
    try:
        raw = func(x_values, *params)
    except Exception as exc:
        raise ValueError(f"model evaluation failed: {exc}")
    array = np.asarray(raw, dtype=float)
    if array.ndim == 0:
        array = np.full(x_values.shape, float(array), dtype=float)
    try:
        return np.broadcast_to(array, x_values.shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(f"model output has an incompatible shape: {exc}")


def _overall_cost(
    func: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_index: np.ndarray,
    theta: np.ndarray,
    n_fixed: int,
    penalty: float,
    random_effect: bool,
) -> float:
    """Return RSS + penalty * sum(b_g^2) for a parameter vector."""
    params = theta[:n_fixed]
    residuals = y_values - _predict(func, x_values, params)
    if random_effect:
        random_terms = theta[n_fixed:]
        residuals = residuals - random_terms[group_index]
        return float(np.sum(residuals**2) + penalty * np.sum(random_terms**2))
    return float(np.sum(residuals**2))


def _result_fields(
    func: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_index: np.ndarray,
    group_labels: list[str],
    theta: np.ndarray,
    n_fixed: int,
    param_names: list[str],
    penalty: float,
    random_effect: bool,
) -> dict[str, Any]:
    """Assemble the public result fields from a parameter vector."""
    params = theta[:n_fixed]
    residuals = y_values - _predict(func, x_values, params)
    n = int(y_values.size)

    fields: dict[str, Any] = {
        "fixed_params": {
            name: float(value) for name, value in zip(param_names, params)
        }
    }

    if random_effect:
        random_terms = theta[n_fixed:]
        residuals = residuals - random_terms[group_index]
        fields["random_effects"] = {
            label: float(value)
            for label, value in zip(group_labels, random_terms)
        }
        n_estimated = n_fixed + len(group_labels)
    else:
        n_estimated = n_fixed

    rss = float(np.sum(residuals**2))
    degrees_of_freedom = n - n_estimated
    if degrees_of_freedom > 0:
        residual_std = float(np.sqrt(rss / degrees_of_freedom))
    else:
        residual_std = float(np.sqrt(rss / n)) if n > 0 else float("nan")

    fields.update(
        {
            "residual_std": residual_std,
            "cost": _overall_cost(
                func,
                x_values,
                y_values,
                group_index,
                theta,
                n_fixed,
                penalty,
                random_effect,
            ),
            "n": n,
            "n_groups": len(group_labels),
            "approximation": True,
            "note": _NOTE,
        }
    )
    return fields


# ---------------------------------------------------------------------------
# Nonlinear mixed model (penalized least squares)
# ---------------------------------------------------------------------------


def nonlinear_mixed_lm(
    model: str,
    data: dict[str, list] | str,
    groups: str,
    x: str,
    y: str,
    p0: dict[str, float],
    random_effect: bool = True,
    penalty: float = 1.0,
    maxiter: int = 2000,
) -> dict:
    """Fit a nonlinear model with one random intercept per group, approximately.

    The model is ``y = f(x; params) + b_g + error`` where ``b_g`` is a random
    intercept shared by the observations of group ``g``. ``model`` is a sympy
    expression in ``x`` (the independent-variable name is always ``x``, taken
    from the column named by the ``x`` argument) and the parameter names given
    by the keys of ``p0``.

    ``data`` is either a ``{column: list}`` mapping or an ``"@<path>.csv"``
    reference (relative paths resolve against ``output_dir()``). ``groups``,
    ``x`` and ``y`` are column names. All fixed parameters and every ``b_g``
    are optimized together with ``scipy.optimize.least_squares``, minimizing
    ``sum((y - f(x; params) - b_g)^2) + penalty * sum(b_g^2)``. Setting
    ``random_effect=False`` drops the ``b_g`` terms and returns no
    ``random_effects`` field.

    This is a **penalized least-squares approximation**, not a full nonlinear
    mixed-effects marginal-likelihood fit: random-effect variance components
    are not estimated, there is no integration over random effects, and no
    standard errors are produced. The response always carries
    ``approximation: true`` and an explanatory ``note``.
    """
    try:
        if not isinstance(groups, str) or not groups.strip():
            return err("nonlinear_mixed_lm failed: groups must be a non-empty column name")
        if not isinstance(x, str) or not x.strip():
            return err("nonlinear_mixed_lm failed: x must be a non-empty column name")
        if not isinstance(y, str) or not y.strip():
            return err("nonlinear_mixed_lm failed: y must be a non-empty column name")
        if not isinstance(random_effect, bool):
            return err("nonlinear_mixed_lm failed: random_effect must be a boolean")
        if isinstance(penalty, bool) or not isinstance(penalty, numbers.Real):
            return err("nonlinear_mixed_lm failed: penalty must be a real number")
        penalty_value = float(penalty)
        if not np.isfinite(penalty_value) or penalty_value < 0:
            return err("nonlinear_mixed_lm failed: penalty must be finite and >= 0")
        if isinstance(maxiter, bool) or not isinstance(maxiter, numbers.Integral):
            return err("nonlinear_mixed_lm failed: maxiter must be an integer")
        maxiter_value = int(maxiter)
        if maxiter_value < 1:
            return err("nonlinear_mixed_lm failed: maxiter must be >= 1")

        param_names, initial = _parse_initial(p0)
        func = _compile_model(model, param_names)

        frame = _load_table(data)
        for column in (groups, x, y):
            if column not in frame.columns:
                return err(
                    f"nonlinear_mixed_lm failed: column {column!r} not found in data"
                )

        try:
            x_values = pd.to_numeric(frame[x], errors="raise").to_numpy(dtype=float)
            y_values = pd.to_numeric(frame[y], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            return err(f"nonlinear_mixed_lm failed: x and y columns must be numeric: {exc}")
        if x_values.ndim != 1 or y_values.ndim != 1:
            return err("nonlinear_mixed_lm failed: x and y must be one-dimensional")
        if not np.all(np.isfinite(x_values)):
            return err("nonlinear_mixed_lm failed: x contains non-finite values")
        if not np.all(np.isfinite(y_values)):
            return err("nonlinear_mixed_lm failed: y contains non-finite values")
        if x_values.size == 0:
            return err("nonlinear_mixed_lm failed: data must contain at least one row")

        raw_groups = frame[groups].to_numpy()
        try:
            unique_raw = list(pd.unique(raw_groups))
        except TypeError as exc:
            return err(f"nonlinear_mixed_lm failed: groups column is not hashable: {exc}")
        group_labels = [str(item) for item in unique_raw]
        if len(set(group_labels)) != len(group_labels):
            return err(
                "nonlinear_mixed_lm failed: group labels are not unique after "
                "conversion to strings"
            )
        if len(group_labels) < 2:
            return err(
                "nonlinear_mixed_lm failed: at least 2 groups are required, "
                f"found {len(group_labels)}"
            )

        group_to_index = {label: index for index, label in enumerate(unique_raw)}
        try:
            group_index = np.asarray(
                [group_to_index[value] for value in raw_groups], dtype=int
            )
        except TypeError as exc:  # pragma: no cover - defensive only
            return err(f"nonlinear_mixed_lm failed: groups column is not usable: {exc}")

        counts = np.bincount(group_index, minlength=len(group_labels))
        too_small = [
            label
            for label, count in zip(group_labels, counts)
            if int(count) < 2
        ]
        if too_small:
            return err(
                "nonlinear_mixed_lm failed: every group needs at least 2 "
                "observations; too few in: " + ", ".join(too_small)
            )

        n_fixed = len(param_names)
        n_groups = len(group_labels)
        if random_effect:
            theta0 = np.concatenate([initial, np.zeros(n_groups, dtype=float)])
        else:
            theta0 = initial.copy()

        def residuals(theta: np.ndarray) -> np.ndarray:
            params = theta[:n_fixed]
            values = y_values - _predict(func, x_values, params)
            if random_effect:
                random_terms = theta[n_fixed:]
                values = values - random_terms[group_index]
                if penalty_value > 0:
                    values = np.concatenate(
                        [values, np.sqrt(penalty_value) * random_terms]
                    )
            return values

        try:
            fit = least_squares(
                residuals,
                theta0,
                method="trf",
                max_nfev=maxiter_value,
            )
        except Exception as exc:
            return err(
                f"nonlinear_mixed_lm failed: optimizer error: {exc}",
                approximation=True,
                note=_NOTE,
            )

        theta = np.asarray(fit.x, dtype=float)
        if not np.all(np.isfinite(theta)):
            fields = _result_fields(
                func,
                x_values,
                y_values,
                group_index,
                group_labels,
                np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0),
                n_fixed,
                param_names,
                penalty_value,
                random_effect,
            )
            return err(
                "nonlinear_mixed_lm failed: optimizer produced non-finite estimates",
                **fields,
            )

        fields = _result_fields(
            func,
            x_values,
            y_values,
            group_index,
            group_labels,
            theta,
            n_fixed,
            param_names,
            penalty_value,
            random_effect,
        )
        converged = bool(fit.success)
        fields["converged"] = converged

        if not converged:
            return err(
                "nonlinear_mixed_lm failed: optimizer did not converge "
                f"(status {int(fit.status)}: {fit.message})",
                **fields,
            )

        return ok(fields)
    except Exception as exc:
        return err(f"nonlinear_mixed_lm failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the nonlinear mixed-model tools on the MCP server."""
    mcp.tool()(nonlinear_mixed_lm)
