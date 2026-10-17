"""Descriptive statistics, probability distributions, curve fitting and plotting tools."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import optimize, stats

from ..core import err, load_array, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_PLOT_KINDS = {"line", "scatter", "hist", "bar", "errorbar", "contour", "surface"}
_DIST_OPERATIONS = {"pdf", "cdf", "ppf", "rvs", "fit"}

# Process-wide counter for auto-generated plot file names.
_PLOT_COUNTER = 0


def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convert a JSON-style list (or numpy array) of numbers to a float array."""
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=float)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list of numbers")
    for item in values:
        if not isinstance(item, numbers.Real):
            raise TypeError(f"{name} contains a non-numeric value: {item!r}")
    return np.asarray(values, dtype=float)


def _array_1d(data: dict[str, Any], key: str) -> np.ndarray:
    """Extract a required one-dimensional numeric array from a plot data mapping."""
    if key not in data or data[key] is None:
        raise ValueError(f"data['{key}'] is required")
    arr = _as_float_array(data[key], f"data['{key}']")
    if arr.ndim != 1:
        raise ValueError(f"data['{key}'] must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"data['{key}'] must not be empty")
    return arr


def _array_2d(data: dict[str, Any], key: str) -> np.ndarray:
    """Extract a required two-dimensional numeric array from a plot data mapping."""
    if key not in data or data[key] is None:
        raise ValueError(f"data['{key}'] is required")
    raw = data[key]
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"data['{key}'] must be a 2-D numeric array") from exc
    if arr.ndim != 2:
        raise ValueError(f"data['{key}'] must be two-dimensional")
    if arr.size == 0:
        raise ValueError(f"data['{key}'] must not be empty")
    return arr


def describe(data: list[float] | str) -> dict:
    """Return descriptive statistics (mean, sample std, quartiles, skew, kurtosis) for a sample."""
    try:
        arr = _as_float_array(load_array(data), "data")
        if arr.ndim != 1:
            return err("describe failed: data must be one-dimensional")
        if arr.size == 0:
            return err("describe failed: data must not be empty")

        q1, q3 = np.percentile(arr, [25.0, 75.0], method="linear")
        return ok(
            {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "q1": float(q1),
                "q3": float(q3),
                "skew": float(stats.skew(arr)),
                "kurtosis": float(stats.kurtosis(arr)),
            }
        )
    except Exception as exc:
        return err(f"describe failed: {exc}")


def distribution(
    operation: str,
    dist: str,
    params: dict[str, float] | None = None,
    x: float | list[float] | None = None,
    n: int = 1000,
    seed: int | None = None,
    data: list[float] | str | None = None,
) -> dict:
    """Evaluate pdf/cdf/ppf/rvs or fit a scipy.stats distribution by attribute name."""
    try:
        data_arr = _as_float_array(load_array(data), "data") if data is not None else None

        op = str(operation).strip().lower()
        if op not in _DIST_OPERATIONS:
            return err(f"distribution failed: unknown operation {operation!r}")

        name = str(dist).strip()
        if not name or name.startswith("_") or not hasattr(stats, name):
            return err(f"distribution failed: unknown distribution {dist!r}")

        dist_obj = getattr(stats, name)
        if not hasattr(dist_obj, "cdf"):
            return err(f"distribution failed: {dist!r} is not a scipy.stats distribution")

        # scipy 1.18 removed the ``pdf`` alias on discrete distributions; use ``pmf`` there.
        density = getattr(dist_obj, "pdf", None) or getattr(dist_obj, "pmf", None)
        if density is None:
            return err(f"distribution failed: {dist!r} has no pdf/pmf method")

        if params is None:
            params = {}
        if not isinstance(params, dict):
            return err("distribution failed: params must be a dict")
        kwargs = dict(params)

        if op in {"pdf", "cdf", "ppf"}:
            if x is None:
                return err(f"distribution failed: operation {op!r} requires x")
            xarr = np.asarray(x, dtype=float)
            if xarr.ndim > 1:
                return err(f"distribution failed: x must be a scalar or one-dimensional list")
            if op == "pdf":
                values = density(xarr, **kwargs)
            elif op == "cdf":
                values = dist_obj.cdf(xarr, **kwargs)
            else:  # ppf
                if not np.all(np.isfinite(xarr)) or np.any(xarr < 0.0) or np.any(xarr > 1.0):
                    return err("distribution failed: ppf probabilities must lie in [0, 1]")
                values = dist_obj.ppf(xarr, **kwargs)
            return ok({"values": values})

        if op == "rvs":
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                return err("distribution failed: n must be a positive integer")
            if n_int != n or n_int <= 0:
                return err("distribution failed: n must be a positive integer")
            if not hasattr(dist_obj, "rvs"):
                return err(f"distribution failed: {dist!r} has no rvs method")
            values = dist_obj.rvs(size=n_int, random_state=seed, **kwargs)
            return ok({"values": values})

        # op == "fit"
        if data_arr is None:
            return err("distribution failed: operation 'fit' requires data")
        if not hasattr(dist_obj, "fit"):
            return err(f"distribution failed: {dist!r} does not provide a fit method")
        arr = data_arr
        if arr.ndim != 1:
            return err("distribution failed: data must be one-dimensional")
        if arr.size == 0:
            return err("distribution failed: data must not be empty")

        fitted = np.asarray(dist_obj.fit(arr, **kwargs), dtype=float).ravel()
        shape_names: list[str] = []
        shapes = getattr(dist_obj, "shapes", None)
        if shapes:
            shape_names = [part.strip() for part in str(shapes).split(",") if part.strip()]
        names = shape_names + ["loc", "scale"]
        if len(names) != fitted.size:
            names = [f"p{i}" for i in range(fitted.size)]
        return ok({"params": {key: float(value) for key, value in zip(names, fitted)}})
    except Exception as exc:
        return err(f"distribution failed: {exc}")


def curve_fit(
    model: str,
    xdata: list[float] | str,
    ydata: list[float] | str,
    p0: list[float] | None = None,
    param_names: list[str] | None = None,
) -> dict:
    """Fit y = model(x) to data with scipy.optimize.curve_fit using a sympy-parsed expression."""
    try:
        if not isinstance(model, str) or not model.strip():
            return err("curve_fit failed: model must be a non-empty expression string")

        xarr = _as_float_array(load_array(xdata), "xdata")
        yarr = _as_float_array(load_array(ydata), "ydata")
        if xarr.ndim != 1 or yarr.ndim != 1:
            return err("curve_fit failed: xdata and ydata must be one-dimensional")
        if xarr.size != yarr.size:
            return err("curve_fit failed: xdata and ydata must have the same length")
        if xarr.size == 0:
            return err("curve_fit failed: xdata and ydata must not be empty")

        x_symbol = sympy.Symbol("x")

        if param_names is None:
            # ``sympify`` resolves names such as ``beta``/``gamma``/``N``/``S``/``E``/``I``
            # to sympy builtins; pin ``x`` and hint at ``param_names`` when that happens.
            try:
                expr = sympy.sympify(model, locals={"x": x_symbol})
            except Exception as exc:
                return err(
                    "curve_fit failed: could not parse model expression "
                    f"({exc}); 可用 param_names 显式指定参数名"
                )
            names = None
        else:
            if not isinstance(param_names, (list, tuple)) or not param_names:
                return err("curve_fit failed: param_names must be a non-empty list of names")
            names = [str(item) for item in param_names]
            for item in names:
                if not item.isidentifier() or item == "x":
                    return err(f"curve_fit failed: invalid parameter name {item!r}")
            if len(set(names)) != len(names):
                return err("curve_fit failed: param_names must be unique")

            # Force every declared parameter name to be a plain Symbol so that sympy
            # does not reinterpret names like ``beta``/``gamma``/``N``/``S``/``E``/``I``
            # as builtin functions or constants.
            local_dict = {"x": x_symbol}
            local_dict.update({item: sympy.Symbol(item) for item in names})
            try:
                expr = sympy.sympify(model, locals=local_dict)
            except Exception as exc:
                return err(f"curve_fit failed: could not parse model expression ({exc})")

        if not isinstance(expr, sympy.Expr):
            return err("curve_fit failed: model did not parse to a scalar expression")

        free_symbols = sorted(
            (symbol for symbol in expr.free_symbols if symbol.name != "x"),
            key=lambda symbol: symbol.name,
        )
        if not free_symbols:
            return err("curve_fit failed: model contains no free parameters to fit")

        if names is None:
            names = [symbol.name for symbol in free_symbols]
        elif len(names) != len(free_symbols):
            return err(
                "curve_fit failed: param_names must match the number of free parameters "
                f"in the model ({len(free_symbols)})"
            )

        if names != [symbol.name for symbol in free_symbols]:
            substitutions = {
                symbol: sympy.Symbol(new_name)
                for symbol, new_name in zip(free_symbols, names)
            }
            expr = expr.subs(substitutions)

        func = sympy.lambdify(
            [x_symbol] + [sympy.Symbol(item) for item in names], expr, modules="numpy"
        )

        p0_array = None
        if p0 is not None:
            p0_array = _as_float_array(p0, "p0")
            if p0_array.ndim != 1 or p0_array.size != len(names):
                return err(
                    "curve_fit failed: p0 must contain exactly "
                    f"{len(names)} initial values"
                )

        popt, pcov = optimize.curve_fit(func, xarr, yarr, p0=p0_array)
        popt = np.asarray(popt, dtype=float).ravel()

        with np.errstate(invalid="ignore", divide="ignore"):
            param_errors = np.sqrt(np.diag(np.asarray(pcov, dtype=float)))
        param_errors = np.asarray(param_errors, dtype=float).ravel()

        predicted = np.asarray(func(xarr, *popt), dtype=float)
        residuals = yarr - predicted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((yarr - np.mean(yarr)) ** 2))
        # R^2 is 0/0 when ydata is constant; report it as undefined rather than 1.0.
        scale = float(np.max(np.abs(yarr))) if yarr.size else 0.0
        ss_tot_tol = np.finfo(float).eps * max(1.0, scale**2) * yarr.size
        result: dict[str, Any] = {
            "params": {name: float(value) for name, value in zip(names, popt)},
            "param_errors": {
                name: float(value) for name, value in zip(names, param_errors)
            },
        }
        if ss_tot > ss_tot_tol:
            result["r_squared"] = float(1.0 - ss_res / ss_tot)
        else:
            result["r_squared"] = None
            result["r_squared_note"] = "undefined: ydata is constant"

        return ok(result)
    except Exception as exc:
        return err(f"curve_fit failed: {exc}")


def _resolve_plot_path(kind: str, filename: str | None) -> Path:
    """Return the output path, validating caller-supplied file names and avoiding collisions."""
    global _PLOT_COUNTER

    out_dir = output_dir()
    if filename is None:
        while True:
            _PLOT_COUNTER += 1
            candidate = out_dir / f"plot_{kind}_{_PLOT_COUNTER}.png"
            if not candidate.exists():
                return candidate

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    if filename != Path(filename).name or ".." in filename:
        raise ValueError("filename must be a bare file name without path separators or '..'")
    if "/" in filename or "\\" in filename or ":" in filename:
        raise ValueError("filename must be a bare file name without path separators or '..'")

    safe_name = filename if Path(filename).suffix else f"{filename}.png"
    return out_dir / safe_name


def plot(
    kind: str,
    data: dict,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    filename: str | None = None,
    dpi: int = 150,
) -> dict:
    """Render a line/scatter/hist/bar/errorbar/contour/surface plot to a PNG file."""
    try:
        plot_kind = str(kind).strip().lower()
        if plot_kind not in _PLOT_KINDS:
            return err(f"plot failed: unknown kind {kind!r}")
        if not isinstance(data, dict):
            return err("plot failed: data must be a dict")
        if not isinstance(dpi, numbers.Real) or isinstance(dpi, bool) or dpi <= 0:
            return err("plot failed: dpi must be a positive number")

        path = _resolve_plot_path(plot_kind, filename)
        fig = None
        try:
            if plot_kind in {"line", "scatter", "errorbar"}:
                x_values = _array_1d(data, "x")
                y_values = _array_1d(data, "y")
                if x_values.size != y_values.size:
                    raise ValueError("x and y must have the same length")

                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                if plot_kind == "line":
                    ax.plot(x_values, y_values)
                elif plot_kind == "scatter":
                    ax.scatter(x_values, y_values)
                else:
                    if "yerr" not in data or data["yerr"] is None:
                        raise ValueError("data['yerr'] is required for an errorbar plot")
                    raw_yerr = data["yerr"]
                    if isinstance(raw_yerr, numbers.Real) and not isinstance(raw_yerr, bool):
                        yerr: Any = float(raw_yerr)
                    else:
                        yerr = _as_float_array(raw_yerr, "data['yerr']")
                        if yerr.ndim != 1 or yerr.size != y_values.size:
                            raise ValueError(
                                "data['yerr'] must be a scalar or match the length of y"
                            )
                    ax.errorbar(x_values, y_values, yerr=yerr, fmt="o")

            elif plot_kind == "hist":
                values = _array_1d(data, "values")
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                ax.hist(values, bins="auto")

            elif plot_kind == "bar":
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                if "values" in data and data["values"] is not None:
                    values = _array_1d(data, "values")
                    ax.bar(np.arange(values.size), values)
                else:
                    x_values = _array_1d(data, "x")
                    y_values = _array_1d(data, "y")
                    if x_values.size != y_values.size:
                        raise ValueError("x and y must have the same length")
                    ax.bar(x_values, y_values)

            else:  # contour / surface
                x_values = _array_1d(data, "x")
                y_values = _array_1d(data, "y")
                z_values = _array_2d(data, "z")
                if z_values.shape == (y_values.size, x_values.size):
                    pass
                elif z_values.shape == (x_values.size, y_values.size):
                    z_values = z_values.T
                else:
                    raise ValueError(
                        "z must be a 2-D array shaped (len(y), len(x)) or (len(x), len(y))"
                    )
                if x_values.size < 2 or y_values.size < 2:
                    raise ValueError("contour/surface plots need at least 2 x and 2 y values")

                if plot_kind == "contour":
                    fig, ax = plt.subplots(figsize=(6.4, 4.8))
                    ax.contour(x_values, y_values, z_values, levels=20)
                else:
                    fig = plt.figure(figsize=(6.4, 5.2))
                    ax = fig.add_subplot(1, 1, 1, projection="3d")
                    grid_x, grid_y = np.meshgrid(x_values, y_values)
                    ax.plot_surface(grid_x, grid_y, z_values, cmap="viridis")

            if title:
                ax.set_title(str(title))
            if xlabel:
                ax.set_xlabel(str(xlabel))
            if ylabel:
                ax.set_ylabel(str(ylabel))

            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            if fig is not None:
                plt.close(fig)

        return ok({"path": str(path.resolve()), "kind": plot_kind})
    except Exception as exc:
        return err(f"plot failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the statistics and plotting tools on the MCP server."""
    mcp.tool()(describe)
    mcp.tool()(distribution)
    mcp.tool()(curve_fit)
    mcp.tool()(plot)
