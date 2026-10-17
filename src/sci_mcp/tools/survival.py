"""Survival analysis tools: Cox proportional hazards and Kaplan-Meier.

``cox_regression`` wraps :class:`statsmodels.duration.hazard_regression.PHReg`;
``kaplan_meier`` is a small, dependency-free implementation of the product-limit
estimator. Every public function returns an ``ok()``/``err()`` envelope and never
raises.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ..core import err, load_array, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_TABLE_ROWS = 1_000_000


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


def _coerce_vector(value: Any, name: str, minimum: int = 1) -> np.ndarray:
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


def _coerce_event_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    """Validate an event indicator vector: integers restricted to {0, 1}."""
    events = _coerce_vector(value, name)
    if length is not None and events.size != length:
        raise ValueError(
            f"{name} must have the same length as durations "
            f"({events.size} != {length})"
        )
    if not np.all((events == 0.0) | (events == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 event indicators")
    return events.astype(int)


def _load_string_column(value: Any, expected: int, name: str) -> list[str]:
    """Load a group-label column from an inline sequence or an ``@<path>`` file."""
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [str(item) for item in np.asarray(value).reshape(-1).tolist()]
    elif isinstance(value, str):
        # Accept both "@<path>" (the shared convention) and a bare path.
        raw_path = value[1:] if value.startswith("@") else value
        if not raw_path.strip():
            raise ValueError(f"{name} path must be a non-empty string")
        path = _resolve_input_path(raw_path)
        items = _read_string_items(path, name)
        # A single-column file may carry a header row; drop it when its length
        # is exactly one longer than the duration vector.
        if len(items) == expected + 1:
            items = items[1:]
    else:
        raise ValueError(f"{name} must be a list of strings or an '@<path>' reference")

    if len(items) != expected:
        raise ValueError(
            f"{name} must have the same length as durations "
            f"({len(items)} != {expected})"
        )
    return items


def _read_string_items(path: Path, name: str) -> list[str]:
    """Read one string per row from a JSON list/object or a delimited text file."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if not payload:
                raise ValueError(f"{name} JSON object in {path} is empty")
            payload = next(iter(payload.values()))
        if not isinstance(payload, list):
            raise ValueError(f"{name} JSON in {path} must be a list or a column mapping")
        return [str(item) for item in payload]

    if suffix not in {".csv", ".tsv", ".txt"}:
        raise ValueError(f"{name} file must be .csv, .tsv, .txt or .json")

    items: list[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            # Use the first delimited field; fall back to the first token.
            fields = re.split(r"[,;\t]", stripped)
            token = fields[0].strip()
            if token == stripped and " " in stripped:
                token = stripped.split()[0]
            items.append(token)
    if not items:
        raise ValueError(f"{name} file {path} contains no values")
    return items


def _formula_name(name: str) -> str:
    """Quote a column name for patsy when it is not a plain identifier."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return name
    return f"Q({json.dumps(name)})"


def _load_table(data: Any) -> pd.DataFrame:
    """Build a DataFrame from a ``{column: list}`` mapping or ``@<path>.csv``."""
    if isinstance(data, dict):
        if not data:
            raise ValueError("data must not be empty")
        columns: dict[str, list[Any]] = {}
        lengths: set[int] = set()
        for column_name, values in data.items():
            if not isinstance(column_name, str) or not column_name:
                raise ValueError("data column names must be non-empty strings")
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(f"data[{column_name!r}] must be a list of values")
            lengths.add(len(values))
            columns[column_name] = list(values)
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


def _numeric_column(frame: pd.DataFrame, column: Any, role: str) -> np.ndarray:
    """Extract one column as a finite float array, raising on non-numeric data."""
    if not isinstance(column, str) or not column:
        raise ValueError(f"{role} must be a non-empty column name")
    if column not in frame.columns:
        raise ValueError(f"{role} column {column!r} not found in data")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = np.asarray(numeric, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{role} column {column!r} must contain finite numbers")
    return values


def _harrell_concordance(
    times: np.ndarray, events: np.ndarray, risk: np.ndarray
) -> float | None:
    """Compute Harrell's C-index from event times and risk scores.

    A pair is comparable when the earlier subject has an observed event. Tied
    risk scores count as half a concordant pair.
    """
    concordant = 0.0
    comparable = 0.0
    n = times.size
    for index in range(n):
        if events[index] != 1:
            continue
        earlier = times[index]
        risk_i = risk[index]
        for other in range(n):
            if other == index or times[other] <= earlier:
                continue
            comparable += 1.0
            if risk_i > risk[other]:
                concordant += 1.0
            elif risk_i == risk[other]:
                concordant += 0.5
    if comparable == 0.0:
        return None
    return concordant / comparable


def _km_estimate(durations: np.ndarray, events: np.ndarray) -> dict[str, Any]:
    """Compute the Kaplan-Meier product-limit estimate at every unique time."""
    order = np.argsort(durations, kind="mergesort")
    times = durations[order]
    indicators = events[order]

    unique_times = np.unique(times)
    survival: list[float] = []
    at_risk: list[int] = []
    event_counts: list[int] = []

    estimate = 1.0
    for time in unique_times:
        risky = int(np.count_nonzero(times >= time))
        observed = int(np.count_nonzero((times == time) & (indicators == 1)))
        at_risk.append(risky)
        event_counts.append(observed)
        if observed:
            estimate *= 1.0 - observed / risky
        survival.append(float(estimate))

    median: float | None = None
    for time, value in zip(unique_times, survival):
        if value <= 0.5:
            median = float(time)
            break

    return {
        "time": [float(time) for time in unique_times],
        "survival": survival,
        "at_risk": at_risk,
        "events": event_counts,
        "median_survival": median,
    }


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------


def _safe_exp(value: float) -> float:
    """Exponentiate without raising; overflow yields ``inf`` for detection."""
    try:
        return float(math.exp(value))
    except OverflowError:
        return math.inf


def _null_nonfinite(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for JSON safety.

    ``core.err()`` does not sanitize its metadata (unlike ``core.ok()``), so
    failure payloads are cleaned explicitly before being returned.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {key: _null_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_nonfinite(item) for item in value]
    return value


def cox_regression(
    formula: str,
    data: dict[str, list] | str,
    duration_col: str,
    event_col: str,
) -> dict:
    """Fit a Cox proportional hazards regression with statsmodels PHReg.

    ``formula`` contains only the covariate terms (for example ``"x1 + x2"``).
    ``data`` is a ``{column: list}`` mapping or an ``"@<path>.csv"`` reference.
    Event indicators must be 0/1 and durations must be positive.
    """
    try:
        if not isinstance(formula, str) or not formula.strip():
            return err("cox_regression failed: formula must be a non-empty string")
        if not isinstance(duration_col, str) or not duration_col.strip():
            return err("cox_regression failed: duration_col must be a non-empty string")
        if not isinstance(event_col, str) or not event_col.strip():
            return err("cox_regression failed: event_col must be a non-empty string")

        if duration_col == event_col:
            return err(
                "cox_regression failed: duration_col and event_col must be different"
            )

        frame = _load_table(data)

        durations = _numeric_column(frame, duration_col, "duration")
        if np.any(durations <= 0.0):
            return err(
                f"cox_regression failed: duration column {duration_col!r} "
                "must contain only positive values"
            )

        event_values = _numeric_column(frame, event_col, "event")
        if not np.all((event_values == 0.0) | (event_values == 1.0)):
            return err(
                f"cox_regression failed: event column {event_col!r} "
                "must contain only 0/1 indicators"
            )
        frame = frame.copy()
        frame[event_col] = event_values.astype(int)
        frame[duration_col] = durations

        covariate_formula = formula.strip()
        full_formula = f"{_formula_name(duration_col)} ~ {covariate_formula}"

        model = PHReg.from_formula(
            full_formula, data=frame, status=event_col
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit()

        params = np.asarray(result.params, dtype=float).reshape(-1)
        names = [str(name) for name in getattr(result.model, "exog_names", [])]
        if len(names) != params.size:
            names = [f"param{index}" for index in range(params.size)]

        coefficients = {
            name: float(value) for name, value in zip(names, params)
        }
        hazard_ratios = {
            name: _safe_exp(value) for name, value in zip(names, params)
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

        exog = np.asarray(result.model.exog, dtype=float)
        risk = exog @ params
        concordance = _harrell_concordance(
            np.asarray(result.model.endog, dtype=float),
            np.asarray(result.model.status, dtype=float),
            risk.reshape(-1),
        )

        aic: float | None = None
        loglike = getattr(result, "llf", None)
        if loglike is not None:
            value = float(loglike)
            if math.isfinite(value):
                aic = float(-2.0 * value + 2.0 * params.size)

        # A converged PHReg fit on overlapping data has small coefficients,
        # strictly positive finite hazard ratios and finite p-values. Complete
        # separation (or collinearity) violates these: statsmodels emits a
        # ConvergenceWarning, drives coefficients to extreme values, and can
        # underflow hazard ratios to zero.
        warning_triggered = any(
            issubclass(entry.category, ConvergenceWarning) for entry in caught
        )
        coefficient_triggered = bool(
            params.size and float(np.max(np.abs(params))) > 20.0
        )
        hazard_triggered = any(
            (not math.isfinite(ratio)) or ratio == 0.0
            for ratio in hazard_ratios.values()
        )
        p_value_triggered = any(
            not math.isfinite(value) for value in p_values.values()
        )

        if (
            warning_triggered
            or coefficient_triggered
            or hazard_triggered
            or p_value_triggered
        ):
            reasons: list[str] = []
            if warning_triggered:
                messages = "; ".join(
                    str(entry.message) for entry in caught
                    if issubclass(entry.category, ConvergenceWarning)
                )
                reasons.append(f"ConvergenceWarning: {messages}")
            if coefficient_triggered:
                reasons.append(
                    "max |coefficient| = "
                    f"{float(np.max(np.abs(params))):.6g} exceeds 20"
                )
            if hazard_triggered:
                reasons.append(
                    "at least one hazard ratio is zero or non-finite"
                )
            if p_value_triggered:
                reasons.append("at least one p-value is non-finite")
            convergence_note = (
                "PHReg did not converge (possible complete separation or "
                "collinearity); " + "; ".join(reasons)
            )
            return err(
                "cox_regression did not converge "
                "(possible complete separation or collinearity)",
                **_null_nonfinite(
                    {
                        "coefficients": coefficients,
                        "hazard_ratios": hazard_ratios,
                        "std_errors": std_errors,
                        "p_values": p_values,
                        "concordance": concordance,
                        "aic": aic,
                        "convergence_note": convergence_note,
                    }
                ),
            )

        return ok(
            {
                "formula": covariate_formula,
                "full_formula": full_formula,
                "n": int(result.model.nobs),
                "coefficients": coefficients,
                "hazard_ratios": hazard_ratios,
                "std_errors": std_errors,
                "z_values": z_values,
                "p_values": p_values,
                "concordance": concordance,
                "aic": aic,
            }
        )
    except Exception as exc:
        return err(f"cox_regression failed: {exc}")


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------


def kaplan_meier(
    durations: list[float] | str,
    events: list[int] | str,
    groups: list[str] | str | None = None,
) -> dict:
    """Estimate survival curves with the Kaplan-Meier product-limit method.

    ``durations`` and ``events`` may be inline lists or ``"@<path>"`` references.
    ``groups`` optionally supplies one group label per observation (inline list
    or ``"@<path>"`` file); when given, a separate curve is returned per group.
    """
    try:
        durations_array = _coerce_vector(durations, "durations")
        events_array = _coerce_event_vector(
            events, "events", length=durations_array.size
        )

        if np.any(durations_array <= 0.0):
            return err(
                "kaplan_meier failed: durations must contain only positive values"
            )

        if groups is None:
            return ok(_km_estimate(durations_array, events_array))

        labels = _load_string_column(groups, durations_array.size, "groups")
        curves: dict[str, Any] = {}
        for label in dict.fromkeys(labels):
            mask = np.asarray([item == label for item in labels], dtype=bool)
            curves[label] = _km_estimate(durations_array[mask], events_array[mask])
        return ok({"groups": curves})
    except Exception as exc:
        return err(f"kaplan_meier failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the survival analysis tools on the MCP server."""
    mcp.tool()(cox_regression)
    mcp.tool()(kaplan_meier)
