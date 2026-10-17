"""Statistical models and time-series tools: GLM, ARIMA, decomposition, ADF.

The tools in this module wrap statsmodels/pandas. Every public function returns
an ``ok()``/``err()`` envelope and never raises.
"""

from __future__ import annotations

import itertools
import math
import numbers
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from ..core import err, load_array, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


# BRIEF section 11: never return more than 2000 numeric points in one array.
_MAX_POINTS = 2000
_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_TABLE_ROWS = 1_000_000
_MAX_FORECAST_STEPS = 10_000

_OUTPUT_COUNTER = itertools.count(1)

_FAMILIES: dict[str, Any] = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
    "gamma": sm.families.Gamma,
    "inverse_gaussian": sm.families.InverseGaussian,
    "negativebinomial": sm.families.NegativeBinomial,
}

_DECOMPOSE_MODELS = ("additive", "multiplicative")
_ADF_REGRESSIONS = ("c", "ct", "ctt", "n")
_ADF_AUTOLAGS = {"aic": "AIC", "bic": "BIC", "t-stat": "t-stat"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_series(value: Any, name: str, minimum: int = 2) -> np.ndarray:
    """Validate a list or ``@<path>`` reference into a finite 1-D float array."""
    if not isinstance(value, (list, tuple, str, np.ndarray)):
        raise ValueError(f"{name} must be a list of numbers or an '@<path>' reference")
    array = np.asarray(load_array(value, dtype=float), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series")
    if array.size < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _series_payload(values: np.ndarray) -> tuple[list[Any], dict[str, Any]]:
    """Convert an array to a list, evenly downsampling it above 2000 points."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size <= _MAX_POINTS:
        return flat.tolist(), {}
    step = (flat.size + _MAX_POINTS - 1) // _MAX_POINTS  # ceil(size / limit)
    reduced = flat[::step]
    meta: dict[str, Any] = {
        "downsampled": True,
        "original_length": int(flat.size),
        "downsample_step": int(step),
    }
    return reduced.tolist(), meta


def _merge_meta(target: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge downsampling metadata, recording the largest original length."""
    for key, value in extra.items():
        if key == "original_length":
            target[key] = max(int(target.get(key, 0)), int(value))
        elif key == "downsample_step":
            target[key] = max(int(target.get(key, 0)), int(value))
        else:
            target[key] = value


def _as_int(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    """Validate an integral scalar (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _int_sequence(
    values: Any, name: str, length: int, minimum: int = 0
) -> tuple[int, ...]:
    """Validate a fixed-length sequence of non-negative integers."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of {length} integers")
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    result: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise ValueError(f"{name}[{index}] must be an integer")
        number = int(item)
        if number < minimum:
            raise ValueError(f"{name}[{index}] must be >= {minimum}")
        result.append(number)
    return tuple(result)


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


def _load_glm_data(data: Any) -> pd.DataFrame:
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


def _new_output_path(prefix: str) -> Path:
    """Allocate a unique .npz path inside output_dir() without overwriting files."""
    directory = output_dir()
    for _ in range(1000):
        candidate = directory / f"{prefix}_{next(_OUTPUT_COUNTER):06d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


# ---------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------


def glm(
    family: str,
    formula: str,
    data: dict[str, list] | str,
) -> dict:
    """Fit a generalized linear model from a formula and a data table.

    ``family`` is one of gaussian, binomial, poisson, gamma, inverse_gaussian
    or negativebinomial. ``data`` is either a ``{column: list}`` mapping or an
    ``"@<path>.csv"`` reference (relative paths resolve against ``output_dir()``).
    """
    try:
        if not isinstance(family, str):
            return err("glm failed: family must be a string")
        family_name = family.strip().lower()
        family_class = _FAMILIES.get(family_name)
        if family_class is None:
            return err(
                "glm failed: unknown family "
                f"{family!r}; expected one of "
                + ", ".join(sorted(_FAMILIES))
            )

        if not isinstance(formula, str) or not formula.strip():
            return err("glm failed: formula must be a non-empty string")

        frame = _load_glm_data(data)

        with warnings.catch_warnings():
            # NegativeBinomial warns about its default alpha and some link
            # combinations emit DomainWarnings; the fit result is still valid.
            warnings.simplefilter("ignore")
            result = smf.glm(
                formula, data=frame, family=family_class()
            ).fit()

        params = np.asarray(result.params, dtype=float)
        names = [str(name) for name in getattr(result.params, "index", [])]
        if len(names) != params.size:
            names = [f"param{index}" for index in range(params.size)]

        coefficients = {
            name: float(value) for name, value in zip(names, params)
        }
        std_errors = {
            name: float(value)
            for name, value in zip(names, np.asarray(result.bse, dtype=float))
        }
        z_values = {
            name: float(value)
            for name, value in zip(names, np.asarray(result.tvalues, dtype=float))
        }
        p_values = {
            name: float(value)
            for name, value in zip(names, np.asarray(result.pvalues, dtype=float))
        }

        pseudo_r2: float | None = getattr(result, "prsquared", None)
        pseudo_r2_method: str | None = "prsquared" if pseudo_r2 is not None else None
        if pseudo_r2 is None:
            llf = getattr(result, "llf", None)
            llnull = getattr(result, "llnull", None)
            if (
                llf is not None
                and llnull is not None
                and math.isfinite(float(llf))
                and math.isfinite(float(llnull))
                and float(llnull) != 0.0
            ):
                pseudo_r2 = 1.0 - float(llf) / float(llnull)
                pseudo_r2_method = "McFadden"

        return ok(
            {
                "family": family_name,
                "n": int(result.nobs),
                "coefficients": coefficients,
                "std_errors": std_errors,
                "z_values": z_values,
                "p_values": p_values,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "pseudo_r2": pseudo_r2,
                "pseudo_r2_method": pseudo_r2_method,
            }
        )
    except Exception as exc:
        return err(f"glm failed: {exc}")


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def arima(
    series: list[float] | str,
    order: list[int] = [1, 0, 0],
    seasonal_order: list[int] | None = None,
    forecast_steps: int = 0,
    conf_alpha: float = 0.05,
) -> dict:
    """Fit an ARIMA model and optionally forecast.

    ``series`` may be an inline list or an ``"@<path>"`` file reference.
    ``forecast_steps`` is limited to 0..10000; ``conf_alpha`` must be in (0, 1).
    """
    try:
        values = _coerce_series(series, "series")
        ar_order = _int_sequence(order, "order", 3)
        if seasonal_order is None:
            seasonal = (0, 0, 0, 0)
        else:
            seasonal = _int_sequence(seasonal_order, "seasonal_order", 4)
            if seasonal[3] == 0 and any(seasonal[:3]):
                return err(
                    "arima failed: seasonal_order period must be > 0 when "
                    "seasonal AR/MA/differencing terms are requested"
                )

        steps = _as_int(
            forecast_steps, "forecast_steps", minimum=0, maximum=_MAX_FORECAST_STEPS
        )
        if isinstance(conf_alpha, bool) or not isinstance(conf_alpha, numbers.Real):
            return err("arima failed: conf_alpha must be a number in (0, 1)")
        alpha = float(conf_alpha)
        if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
            return err("arima failed: conf_alpha must be in (0, 1)")

        result = ARIMA(
            values, order=ar_order, seasonal_order=seasonal
        ).fit()

        params = np.asarray(result.params, dtype=float)
        param_names = getattr(result, "param_names", None)
        if param_names is None or len(param_names) != params.size:
            param_names = [f"param{index}" for index in range(params.size)]
        parameter_map = {
            str(name): float(value) for name, value in zip(param_names, params)
        }

        forecast: list[float] = []
        conf_int: list[list[float]] = []
        if steps > 0:
            prediction = result.get_forecast(steps=steps)
            forecast = [
                float(value)
                for value in np.asarray(prediction.predicted_mean, dtype=float).ravel()
            ]
            interval = np.asarray(
                prediction.conf_int(alpha=alpha), dtype=float
            ).reshape(-1, 2)
            conf_int = [
                [float(low), float(high)] for low, high in interval
            ]

        return ok(
            {
                "order": list(ar_order),
                "seasonal_order": list(seasonal),
                "params": parameter_map,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "forecast": forecast,
                "conf_int": conf_int,
            }
        )
    except Exception as exc:
        return err(f"arima failed: {exc}")


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------


def time_series_decompose(
    series: list[float] | str,
    period: int,
    model: str = "additive",
) -> dict:
    """Decompose a series into trend, seasonal and residual components.

    ``period`` must be at least 2 and smaller than the series length. The full
    components are saved as ``.npz`` under ``output_dir()``; returned arrays are
    evenly downsampled to at most 2000 points each.
    """
    try:
        values = _coerce_series(series, "series")
        if isinstance(period, bool) or not isinstance(period, numbers.Integral):
            return err("time_series_decompose failed: period must be an integer")
        cycles = int(period)
        if cycles < 2:
            return err("time_series_decompose failed: period must be >= 2")
        if cycles >= values.size:
            return err(
                "time_series_decompose failed: period must be smaller than the "
                f"series length (period={cycles}, n={values.size})"
            )

        if not isinstance(model, str):
            return err("time_series_decompose failed: model must be a string")
        model_name = model.strip().lower()
        if model_name not in _DECOMPOSE_MODELS:
            return err(
                "time_series_decompose failed: model must be 'additive' or "
                "'multiplicative'"
            )

        decomposition = seasonal_decompose(
            values, model=model_name, period=cycles
        )
        observed = np.asarray(decomposition.observed, dtype=float)
        trend = np.asarray(decomposition.trend, dtype=float)
        seasonal = np.asarray(decomposition.seasonal, dtype=float)
        resid = np.asarray(decomposition.resid, dtype=float)

        payload: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for key, component in (
            ("observed_preview", observed),
            ("trend", trend),
            ("seasonal", seasonal),
            ("resid", resid),
        ):
            data, component_meta = _series_payload(component)
            payload[key] = data
            _merge_meta(meta, component_meta)

        output_path = _new_output_path("time_series_decompose")
        np.savez(
            output_path,
            observed=observed,
            trend=trend,
            seasonal=seasonal,
            resid=resid,
        )

        result: dict[str, Any] = {
            "period": cycles,
            "model": model_name,
            **payload,
            "path": str(output_path),
        }
        result.update(meta)
        return ok(result)
    except Exception as exc:
        return err(f"time_series_decompose failed: {exc}")


# ---------------------------------------------------------------------------
# Stationarity test
# ---------------------------------------------------------------------------


def stationarity_test(
    series: list[float] | str,
    regression: str = "c",
    autolag: str | None = "AIC",
) -> dict:
    """Run the Augmented Dickey-Fuller test and report stationarity at 5%.

    ``regression`` is one of c, ct, ctt or n. ``autolag`` is AIC, BIC, t-stat
    or None (no automatic lag selection).
    """
    try:
        values = _coerce_series(series, "series", minimum=3)

        if not isinstance(regression, str):
            return err("stationarity_test failed: regression must be a string")
        regression_name = regression.strip().lower()
        if regression_name not in _ADF_REGRESSIONS:
            return err(
                "stationarity_test failed: regression must be one of "
                + ", ".join(_ADF_REGRESSIONS)
            )

        if autolag is None:
            autolag_name: str | None = None
        elif isinstance(autolag, str):
            autolag_name = _ADF_AUTOLAGS.get(autolag.strip().lower())
            if autolag_name is None:
                return err(
                    "stationarity_test failed: autolag must be 'AIC', 'BIC', "
                    "'t-stat' or None"
                )
        else:
            return err(
                "stationarity_test failed: autolag must be 'AIC', 'BIC', "
                "'t-stat' or None"
            )

        adf_result = adfuller(
            values,
            regression=regression_name,
            autolag=autolag_name,
            result_object=False,
        )
        # adfuller returns a 5-tuple when autolag is None (lag chosen by maxlag)
        # and a 6-tuple with the information criterion otherwise.
        statistic, p_value, lags, nobs, critical_values = adf_result[:5]
        p_float = float(p_value)
        return ok(
            {
                "statistic": float(statistic),
                "p_value": p_float,
                "lags": int(lags),
                "nobs": int(nobs),
                "critical_values": {
                    str(level): float(value)
                    for level, value in critical_values.items()
                },
                "stationary": bool(p_float < 0.05),
            }
        )
    except Exception as exc:
        return err(f"stationarity_test failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the statistical model and time-series tools on the MCP server."""
    mcp.tool()(glm)
    mcp.tool()(arima)
    mcp.tool()(time_series_decompose)
    mcp.tool()(stationarity_test)
