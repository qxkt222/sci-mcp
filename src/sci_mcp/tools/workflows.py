"""High-level workflow tools: dataset profiling and time-series forecasting."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..core import err, load_array, ok, output_dir
from .stats_inference import linear_regression
from .stats_plot import plot as plot_tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_ROWS = 1_000_000
_MAX_POINTS = 2000
_MAX_HISTOGRAMS = 6
_MAX_PAIRS = 10

_FORECAST_METHODS = {"arima", "exp_smoothing"}

# Process-wide counter for auto-generated workflow plot names.
_PLOT_COUNTER = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_input_path(path: Any) -> Path:
    """Resolve an input path (relative paths are anchored at output_dir()) and check limits."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir() / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"not a regular file: {candidate}")
    size = candidate.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"file is too large ({size} bytes); the limit is {_MAX_FILE_BYTES} bytes"
        )
    return candidate


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a CSV file with pandas while enforcing the row limit."""
    frame = pd.read_csv(path, nrows=_MAX_ROWS + 1)
    if len(frame) > _MAX_ROWS:
        raise ValueError(f"file has more than {_MAX_ROWS} rows")
    return frame


def _numeric_column_names(frame: pd.DataFrame) -> list[Any]:
    """Return the columns with a numeric dtype, excluding booleans."""
    names: list[Any] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series.dtype):
            continue
        # Complex data cannot be reduced to the real-valued summaries this tool
        # reports, so it is treated like any other non-numeric column.
        if pd.api.types.is_complex_dtype(series.dtype):
            continue
        if pd.api.types.is_numeric_dtype(series.dtype):
            names.append(name)
    return names


def _non_numeric_column_names(frame: pd.DataFrame) -> list[Any]:
    """Return the columns without a numeric dtype (booleans count as non-numeric here)."""
    numeric = set(_numeric_column_names(frame))
    return [name for name in frame.columns if name not in numeric]


def _column_summary(series: pd.Series) -> dict[str, Any]:
    """Return n/mean/std/min/max/median/q1/q3 for the finite values of a column."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q1": None,
            "q3": None,
        }
    if n > 1:
        std = float(np.std(values, ddof=1))
    else:
        std = 0.0
    q1, q3 = np.percentile(values, [25.0, 75.0], method="linear")
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
    }


def _correlation_block(
    frame: pd.DataFrame, columns: list[Any]
) -> dict[str, Any]:
    """Build the pairwise correlation pairs and the full correlation matrix."""
    if not columns:
        return {"pairs": [], "matrix_columns": [], "matrix": []}

    matrix_frame = frame[columns].corr()
    matrix = np.asarray(matrix_frame.to_numpy(dtype=float), dtype=float)

    pairs: list[dict[str, Any]] = []
    for row in range(len(columns)):
        for col in range(row + 1, len(columns)):
            value = float(matrix[row, col])
            if math.isfinite(value):
                pairs.append(
                    {
                        "a": str(columns[row]),
                        "b": str(columns[col]),
                        "r": value,
                    }
                )
    # Strongest linear relationships first; ties keep the original column order.
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)

    return {
        "pairs": pairs[:_MAX_PAIRS],
        "matrix_columns": [str(name) for name in columns],
        "matrix": matrix.tolist(),
    }


def _resolve_target(frame: pd.DataFrame, target: Any) -> Any:
    """Resolve a column name or positional index to an existing column label."""
    if isinstance(target, bool):
        raise ValueError("target must be a column name or index")
    if isinstance(target, numbers.Integral):
        index = int(target)
        if index < 0 or index >= len(frame.columns):
            raise ValueError(
                f"target column index {index} not found "
                f"(the file has {len(frame.columns)} columns)"
            )
        return frame.columns[index]
    if isinstance(target, str):
        if target in frame.columns:
            return target
        for name in frame.columns:
            if str(name) == target:
                return name
        raise ValueError(f"target column {target!r} not found")
    raise ValueError("target must be a column name or index")


def _target_analysis(
    frame: pd.DataFrame, target_name: Any, numeric_columns: list[Any]
) -> dict[str, Any]:
    """Correlate numeric columns with the target and regress the target on them."""
    series = frame[target_name]
    is_numeric = (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
    )
    label = str(target_name)

    if not is_numeric:
        return {
            "target": label,
            "correlations": [],
            "regression": None,
            "note": "target is not numeric; correlations and regression are not defined",
        }

    correlations: list[dict[str, Any]] = []
    for name in numeric_columns:
        if name == target_name:
            continue
        value = float(series.corr(frame[name]))
        if math.isfinite(value):
            correlations.append({"column": str(name), "r": value})

    predictors = [name for name in numeric_columns if name != target_name]
    regression: dict[str, Any] | None = None
    regression_error: str | None = None

    if predictors:
        subset = frame[predictors + [target_name]].apply(
            pd.to_numeric, errors="coerce"
        )
        values = np.asarray(subset.to_numpy(dtype=float), dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if values.shape[0] >= 2:
            x_values = values[:, :-1]
            y_values = values[:, -1]
            if x_values.shape[1] == 1:
                x_arg: Any = x_values[:, 0].tolist()
            else:
                x_arg = x_values.tolist()
            fitted = linear_regression(
                x_arg, y_values.tolist(), feature_names=[str(name) for name in predictors]
            )
            if fitted.get("ok"):
                result = fitted["result"]
                regression = {
                    "coefficients": result["coefficients"],
                    "r_squared": result["r_squared"],
                }
            else:
                regression_error = str(fitted.get("error", "regression failed"))
        else:
            regression_error = "too few complete observations for regression"
    else:
        regression_error = "no predictor columns available"

    payload: dict[str, Any] = {
        "target": label,
        "correlations": correlations,
        "regression": regression,
    }
    if regression_error is not None and regression is None:
        payload["regression_error"] = regression_error
    return payload


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _next_plot_path(prefix: str) -> Path:
    """Return a fresh auto-numbered PNG path in the output directory."""
    global _PLOT_COUNTER
    out_dir = output_dir()
    while True:
        _PLOT_COUNTER += 1
        candidate = out_dir / f"plot_{prefix}_{_PLOT_COUNTER}.png"
        if not candidate.exists():
            return candidate


def _render_heatmap(columns: list[Any], matrix: np.ndarray) -> Path:
    """Render and save a correlation heatmap with matplotlib's Agg backend."""
    path = _next_plot_path("correlation_heatmap")
    labels = [str(name) for name in columns]
    size = max(4.8, 0.6 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(size, max(4.0, 0.5 * len(labels) + 1.6)))
    try:
        image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        positions = np.arange(len(labels))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_title("Correlation matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path.resolve()


def _make_plots(
    frame: pd.DataFrame, columns: list[Any], matrix: np.ndarray, enabled: bool
) -> list[dict[str, Any]]:
    """Generate per-column histograms (up to six) or one correlation heatmap."""
    if not enabled or not columns:
        return []

    plots: list[dict[str, Any]] = []

    if len(columns) > _MAX_HISTOGRAMS:
        try:
            path = _render_heatmap(columns, matrix)
            plots.append({"kind": "correlation_heatmap", "path": str(path)})
        except Exception:
            # Plotting must never invalidate an otherwise successful analysis.
            pass
        return plots

    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rendered = plot_tool(
            "hist",
            {"values": values.tolist()},
            title=f"Distribution of {name}",
            xlabel=str(name),
            ylabel="frequency",
        )
        if rendered.get("ok"):
            result = rendered["result"]
            plots.append({"kind": "hist", "path": result["path"]})
    return plots


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def analyze_dataset(
    path: str,
    target: str | int | None = None,
    max_columns: int = 20,
    plot: bool = True,
) -> dict:
    """Profile a CSV file: numeric summaries, correlations, optional target regression and plots."""
    try:
        if (
            isinstance(max_columns, bool)
            or not isinstance(max_columns, numbers.Integral)
            or int(max_columns) < 1
        ):
            return err("analyze_dataset failed: max_columns must be a positive integer")
        max_columns = int(max_columns)
        make_plots = bool(plot)

        resolved = _resolve_input_path(path)
        frame = _read_frame(resolved)
        if frame.shape[1] == 0:
            return err("analyze_dataset failed: the file has no columns")

        numeric_all = _numeric_column_names(frame)
        if not numeric_all:
            return err("analyze_dataset failed: the file has no numeric columns")

        # Analysis is capped at max_columns; the remaining columns are not profiled.
        analyzed = numeric_all[:max_columns]
        ignored = _non_numeric_column_names(frame)

        summary = {str(name): _column_summary(frame[name]) for name in analyzed}
        correlations = _correlation_block(frame, analyzed)

        target_analysis = None
        if target is not None:
            target_name = _resolve_target(frame, target)
            target_analysis = _target_analysis(frame, target_name, analyzed)

        matrix = np.asarray(correlations["matrix"], dtype=float)
        plots = _make_plots(frame, analyzed, matrix, make_plots)

        return ok(
            {
                "path": str(resolved),
                "n_rows": int(frame.shape[0]),
                "numeric_columns": [str(name) for name in analyzed],
                "ignored_columns": [str(name) for name in ignored],
                "summary": summary,
                "correlations": correlations,
                "target_analysis": target_analysis,
                "plots": plots,
            }
        )
    except Exception as exc:
        return err(f"analyze_dataset failed: {exc}")


def _validate_steps(steps: Any) -> int:
    """Validate the forecast horizon (1..10000) and return it as an int."""
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral):
        raise ValueError("steps must be an integer")
    value = int(steps)
    if value < 1 or value > 10000:
        raise ValueError("steps must be between 1 and 10000")
    return value


def _validate_order(order: Any) -> tuple[int, int, int]:
    """Validate an ARIMA (p, d, q) order and return it as a tuple."""
    if order is None:
        return (1, 0, 0)
    if not isinstance(order, (list, tuple)) or len(order) != 3:
        raise ValueError("order must be a list of three non-negative integers")
    values: list[int] = []
    for item in order:
        if (
            isinstance(item, bool)
            or not isinstance(item, numbers.Integral)
            or int(item) < 0
        ):
            raise ValueError("order must be a list of three non-negative integers")
        values.append(int(item))
    return (values[0], values[1], values[2])


def _validate_seasonal_period(seasonal_period: Any) -> int | None:
    """Validate an optional seasonal period of at least two."""
    if seasonal_period is None:
        return None
    if (
        isinstance(seasonal_period, bool)
        or not isinstance(seasonal_period, numbers.Integral)
        or int(seasonal_period) < 2
    ):
        raise ValueError("seasonal_period must be an integer greater than or equal to 2")
    return int(seasonal_period)


def _validate_alpha(conf_alpha: Any) -> float:
    """Validate a confidence level alpha in the open interval (0, 1)."""
    if isinstance(conf_alpha, bool) or not isinstance(conf_alpha, numbers.Real):
        raise ValueError("conf_alpha must be a number between 0 and 1")
    value = float(conf_alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("conf_alpha must be between 0 and 1 (exclusive)")
    return value


def _downsample_forecast(
    forecast: np.ndarray, intervals: np.ndarray
) -> tuple[list[float], list[list[float]], dict[str, Any]]:
    """Limit forecast output to 2000 points, reporting the original horizon."""
    n = int(forecast.size)
    if n <= _MAX_POINTS:
        return (
            [float(value) for value in forecast],
            [[float(lo), float(hi)] for lo, hi in intervals],
            {},
        )
    indices = np.linspace(0, n - 1, _MAX_POINTS).round().astype(np.int64)
    sampled = forecast[indices]
    sampled_intervals = intervals[indices]
    return (
        [float(value) for value in sampled],
        [[float(lo), float(hi)] for lo, hi in sampled_intervals],
        {"downsampled": True, "original_length": n},
    )


def forecast(
    series: list[float] | str,
    steps: int = 10,
    method: str = "arima",
    order: list[int] | None = None,
    seasonal_period: int | None = None,
    conf_alpha: float = 0.05,
) -> dict:
    """Forecast a univariate series with ARIMA or Holt-Winters exponential smoothing.

    ARIMA uses the given ``order`` (default ``[1, 0, 0]``); when
    ``seasonal_period`` is set the seasonal order is ``[1, 0, 0, period]``.
    Exponential smoothing uses an additive trend and, when a period is given,
    an additive seasonal component. ARIMA intervals are analytic; exponential
    smoothing intervals are a normal approximation from the residual standard
    deviation.
    """
    try:
        name = str(method).strip().lower() if isinstance(method, str) else ""
        if name not in _FORECAST_METHODS:
            return err(
                f"forecast failed: unknown method {method!r}; "
                "use 'arima' or 'exp_smoothing'"
            )

        horizon = _validate_steps(steps)
        alpha = _validate_alpha(conf_alpha)
        period = _validate_seasonal_period(seasonal_period)

        values = load_array(series)
        if values.ndim != 1:
            return err("forecast failed: series must be one-dimensional")
        if values.size < 4:
            return err("forecast failed: series must contain at least 4 values")
        if not np.all(np.isfinite(values)):
            return err("forecast failed: series contains non-finite values")

        model_info: dict[str, Any] = {"n_obs": int(values.size)}

        if name == "arima":
            arima_order = _validate_order(order)

            # Imported lazily so the server does not pay the statsmodels import
            # cost when this module is merely discovered.
            from statsmodels.tsa.arima.model import ARIMA

            kwargs: dict[str, Any] = {"order": arima_order}
            seasonal_order: tuple[int, int, int, int] | None = None
            if period is not None:
                seasonal_order = (1, 0, 0, period)
                kwargs["seasonal_order"] = seasonal_order

            fitted = ARIMA(values, **kwargs).fit()
            forecast_result = fitted.get_forecast(steps=horizon)
            mean = np.asarray(forecast_result.predicted_mean, dtype=float).ravel()
            intervals = np.asarray(
                forecast_result.conf_int(alpha=alpha), dtype=float
            )

            model_info.update(
                {
                    "order": list(arima_order),
                    "seasonal_order": (
                        list(seasonal_order) if seasonal_order is not None else None
                    ),
                    "conf_int_method": "analytic (statsmodels ARIMA)",
                }
            )
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            seasonal = "add" if period is not None else None
            hw_kwargs: dict[str, Any] = {"trend": "add", "seasonal": seasonal}
            if period is not None:
                hw_kwargs["seasonal_periods"] = period

            fitted = ExponentialSmoothing(values, **hw_kwargs).fit()
            mean = np.asarray(fitted.forecast(horizon), dtype=float).ravel()

            residuals = np.asarray(fitted.resid, dtype=float).ravel()
            residuals = residuals[np.isfinite(residuals)]
            sigma = (
                float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
            )
            z_value = float(stats.norm.ppf(1.0 - alpha / 2.0))
            half_width = z_value * sigma * np.sqrt(np.arange(1, horizon + 1, dtype=float))
            intervals = np.column_stack([mean - half_width, mean + half_width])

            model_info.update(
                {
                    "trend": "add",
                    "seasonal": seasonal,
                    "seasonal_periods": period,
                    "conf_int_method": (
                        "normal approximation from residual standard deviation"
                    ),
                }
            )

        if mean.size != horizon:
            return err(
                "forecast failed: the model returned an unexpected forecast length "
                f"({mean.size} instead of {horizon})"
            )
        if intervals.ndim != 2 or intervals.shape != (horizon, 2):
            return err("forecast failed: the model returned malformed confidence intervals")

        forecast_values, conf_int, meta = _downsample_forecast(mean, intervals)

        payload: dict[str, Any] = {
            "method": name,
            "steps": horizon,
            "forecast": forecast_values,
            "conf_int": conf_int,
            "model_info": model_info,
        }
        aic = float(fitted.aic)
        bic = float(fitted.bic)
        if math.isfinite(aic):
            payload["aic"] = aic
        if math.isfinite(bic):
            payload["bic"] = bic
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"forecast failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the high-level workflow tools on the MCP server."""
    mcp.tool()(analyze_dataset)
    mcp.tool()(forecast)
