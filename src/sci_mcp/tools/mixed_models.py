"""Mixed models: linear mixed models (MixedLM) and generalized estimating equations.

The tools in this module wrap statsmodels/pandas. Every public function returns
an ``ok()``/``err()`` envelope and never raises.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ..core import err, ok, output_dir

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_TABLE_ROWS = 1_000_000

_FAMILIES: dict[str, Any] = {
    "gaussian": sm.families.Gaussian,
    "binomial": sm.families.Binomial,
    "poisson": sm.families.Poisson,
}

_COV_STRUCTURES: dict[str, Any] = {
    "independence": sm.cov_struct.Independence,
    "exchangeable": sm.cov_struct.Exchangeable,
    "autoregressive": sm.cov_struct.Autoregressive,
}


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


def _series_pairs(
    names: Any, values: Any
) -> tuple[dict[str, float], list[str]]:
    """Zip parameter names with numeric values, coercing names to strings."""
    clean_names = [str(name) for name in names]
    numbers = np.asarray(values, dtype=float)
    if numbers.size != len(clean_names):
        clean_names = [f"param{index}" for index in range(numbers.size)]
    return {
        name: float(value) for name, value in zip(clean_names, numbers)
    }, clean_names


def _finite_or_none(value: Any) -> float | None:
    """Convert a scalar to float, returning None when it is missing/non-finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _group_variance(result: Any) -> float | None:
    """Extract the random-effects variance (first diagonal of ``cov_re``)."""
    cov_re = getattr(result, "cov_re", None)
    if cov_re is None:
        return None
    array = np.asarray(cov_re, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 2:
        return _finite_or_none(array[0, 0])
    return _finite_or_none(array.reshape(-1)[0])


def _cov_re_is_psd(result: Any) -> bool:
    """Return False when ``cov_re`` is clearly not positive semidefinite."""
    cov_re = getattr(result, "cov_re", None)
    if cov_re is None:
        return True
    array = np.asarray(cov_re, dtype=float)
    if array.size == 0 or array.ndim != 2:
        return True
    if not np.all(np.isfinite(array)):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh((array + array.T) / 2.0)
    except np.linalg.LinAlgError:
        return False
    scale = max(1.0, float(np.max(np.abs(array))))
    return bool(float(np.min(eigenvalues)) >= -1e-8 * scale)


def _convergence_message(captured: list[warnings.WarningMessage]) -> str:
    """Summarize recorded convergence-related warnings for an error envelope."""
    keywords = ("converge", "failed", "boundary", "singular", "gradient")
    seen: list[str] = []
    for item in captured:
        text = str(item.message)
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords) and text not in seen:
            seen.append(text)
    if not seen:
        return "MixedLM optimization failed to converge"
    return " | ".join(seen[:5])


def _mixed_diagnostics(result: Any) -> dict[str, Any]:
    """Best-effort result fields for an err() envelope (never raises)."""
    diagnostics: dict[str, Any] = {}
    try:
        fe_params = getattr(result, "fe_params", None)
        if fe_params is not None:
            diagnostics["fixed_effects"] = {
                str(name): _finite_or_none(value)
                for name, value in fe_params.items()
            }
    except Exception:  # pragma: no cover - defensive only
        pass
    try:
        diagnostics["group_variance"] = _group_variance(result)
    except Exception:  # pragma: no cover - defensive only
        diagnostics["group_variance"] = None
    return diagnostics


def _n_groups(result: Any, frame: pd.DataFrame, groups: str) -> int:
    """Number of distinct groups, preferring the fitted model's count."""
    model = getattr(result, "model", None)
    count = getattr(model, "n_groups", None)
    if count is not None:
        try:
            return int(count)
        except (TypeError, ValueError):  # pragma: no cover - defensive only
            pass
    return int(frame[groups].nunique())


# ---------------------------------------------------------------------------
# Linear mixed model
# ---------------------------------------------------------------------------


def mixed_lm(
    formula: str,
    data: dict[str, list] | str,
    groups: str,
    re_formula: str | None = None,
    reml: bool = True,
) -> dict:
    """Fit a linear mixed-effects model with a random-effects structure.

    ``data`` is either a ``{column: list}`` mapping or an ``"@<path>.csv"``
    reference (relative paths resolve against ``output_dir()``). ``groups``
    names the grouping column. ``re_formula`` optionally describes the random
    effects (e.g. ``"~1"`` for random intercepts only). ``reml`` selects REML
    (default) or maximum likelihood.

    Returns fixed-effect estimates with standard errors, z values and p values,
    the random-intercept variance, AIC/BIC and the convergence flag.
    """
    try:
        if not isinstance(formula, str) or not formula.strip():
            return err("mixed_lm failed: formula must be a non-empty string")
        if not isinstance(groups, str) or not groups.strip():
            return err("mixed_lm failed: groups must be a non-empty column name")
        if re_formula is not None and (
            not isinstance(re_formula, str) or not re_formula.strip()
        ):
            return err("mixed_lm failed: re_formula must be a non-empty string")

        frame = _load_table(data)
        if groups not in frame.columns:
            return err(f"mixed_lm failed: groups column {groups!r} not found in data")

        captured: list[warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.mixedlm(
                    formula,
                    data=frame,
                    groups=frame[groups],
                    re_formula=re_formula,
                ).fit(reml=bool(reml))
            finally:
                captured = list(caught)

        if not bool(getattr(result, "converged", False)):
            return err(
                "mixed_lm failed: model did not converge",
                solver_message=_convergence_message(captured),
                **_mixed_diagnostics(result),
            )
        if not _cov_re_is_psd(result):
            return err(
                "mixed_lm failed: random-effects covariance is not "
                "positive semidefinite",
                solver_message="cov_re is not positive semidefinite",
                **_mixed_diagnostics(result),
            )

        fe_params = result.fe_params
        fixed_effects = {
            str(name): float(value) for name, value in fe_params.items()
        }
        std_errors = {
            str(name): float(result.bse_fe[name]) for name in fe_params.index
        }
        z_values = {
            str(name): float(result.tvalues[name]) for name in fe_params.index
        }
        p_values = {
            str(name): float(result.pvalues[name]) for name in fe_params.index
        }

        return ok(
            {
                "formula": formula,
                "groups": groups,
                "n": int(round(float(result.nobs))),
                "n_groups": _n_groups(result, frame, groups),
                "fixed_effects": fixed_effects,
                "std_errors": std_errors,
                "z_values": z_values,
                "p_values": p_values,
                "group_variance": _group_variance(result),
                "aic": float(result.aic),
                "bic": float(result.bic),
                "converged": bool(result.converged),
            }
        )
    except Exception as exc:
        return err(f"mixed_lm failed: {exc}")


# ---------------------------------------------------------------------------
# Generalized estimating equations
# ---------------------------------------------------------------------------


def gee(
    formula: str,
    data: dict[str, list] | str,
    groups: str,
    family: str = "gaussian",
    cov_struct: str = "exchangeable",
) -> dict:
    """Fit a generalized estimating equation model.

    ``data`` is either a ``{column: list}`` mapping or an ``"@<path>.csv"``
    reference. ``groups`` names the clustering column. ``family`` is one of
    gaussian, binomial or poisson; ``cov_struct`` is one of independence,
    exchangeable or autoregressive.

    Returns population-averaged coefficients with robust standard errors,
    z values and p values.
    """
    try:
        if not isinstance(formula, str) or not formula.strip():
            return err("gee failed: formula must be a non-empty string")
        if not isinstance(groups, str) or not groups.strip():
            return err("gee failed: groups must be a non-empty column name")

        if not isinstance(family, str):
            return err("gee failed: family must be a string")
        family_name = family.strip().lower()
        family_class = _FAMILIES.get(family_name)
        if family_class is None:
            return err(
                "gee failed: unknown family "
                f"{family!r}; expected one of " + ", ".join(sorted(_FAMILIES))
            )

        if not isinstance(cov_struct, str):
            return err("gee failed: cov_struct must be a string")
        cov_name = cov_struct.strip().lower()
        cov_class = _COV_STRUCTURES.get(cov_name)
        if cov_class is None:
            return err(
                "gee failed: unknown cov_struct "
                f"{cov_struct!r}; expected one of "
                + ", ".join(sorted(_COV_STRUCTURES))
            )

        frame = _load_table(data)
        if groups not in frame.columns:
            return err(f"gee failed: groups column {groups!r} not found in data")

        with warnings.catch_warnings():
            # statsmodels emits a grid= FutureWarning for some covariance
            # structures; the fit itself is unaffected.
            warnings.simplefilter("ignore", FutureWarning)
            result = smf.gee(
                formula,
                groups=groups,
                data=frame,
                family=family_class(),
                cov_struct=cov_class(),
            ).fit()

        coefficient_map, names = _series_pairs(result.params.index, result.params)
        std_errors, _ = _series_pairs(result.params.index, result.bse)
        z_values, _ = _series_pairs(result.params.index, result.tvalues)
        p_values, _ = _series_pairs(result.params.index, result.pvalues)

        model_groups = getattr(result.model, "groups", None)
        if model_groups is None:
            group_count = int(frame[groups].nunique())
        else:
            group_count = int(len(pd.unique(np.asarray(model_groups).ravel())))

        return ok(
            {
                "formula": formula,
                "groups": groups,
                "n": int(round(float(result.nobs))),
                "n_groups": group_count,
                "family": family_name,
                "cov_struct": cov_name,
                "coefficients": coefficient_map,
                "std_errors": std_errors,
                "z_values": z_values,
                "p_values": p_values,
            }
        )
    except Exception as exc:
        return err(f"gee failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the mixed-model tools on the MCP server."""
    mcp.tool()(mixed_lm)
    mcp.tool()(gee)
