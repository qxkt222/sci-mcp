"""Statistical inference tools: t-tests, chi-square tests, ANOVA, correlation and regression."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from ..core import err, load_array, ok

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_ALTERNATIVES = {"two-sided", "less", "greater"}
_CORRELATION_METHODS = {"pearson", "spearman", "kendall"}


def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convert a JSON-style list (or numpy array) of numbers to a float array."""
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=float)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list of numbers")
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            raise TypeError(f"{name} must be one-dimensional")
        if not isinstance(item, numbers.Real):
            raise TypeError(f"{name} contains a non-numeric value: {item!r}")
    return np.asarray(values, dtype=float)


def _check_sample(arr: np.ndarray, name: str, minimum: int = 2) -> None:
    """Validate a one-dimensional finite sample of sufficient length."""
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def _t_and_p(estimate: float, standard_error: float, df: int) -> tuple[float, float]:
    """Return the t statistic and two-sided p value for an OLS coefficient."""
    if standard_error > 0.0:
        t_value = estimate / standard_error
        p_value = 2.0 * float(stats.t.sf(abs(t_value), df))
        return float(t_value), float(p_value)
    # A zero standard error only occurs for an exact fit; the t statistic is
    # infinite for a non-zero estimate and formally undefined (reported as 0)
    # for a zero estimate.
    if estimate == 0.0:
        return 0.0, 1.0
    return float("inf"), 0.0


def t_test(
    sample1: list[float] | str,
    sample2: list[float] | str | None = None,
    popmean: float | None = None,
    paired: bool = False,
    alternative: str = "two-sided",
) -> dict:
    """Run a one-sample, two-sample or paired t-test with an optional one-sided alternative.

    Samples may be passed inline or as an ``"@<path>"`` file reference.
    """
    try:
        alt = str(alternative).strip().lower()
        if alt not in _ALTERNATIVES:
            return err(
                "t_test failed: alternative must be 'two-sided', 'less' or 'greater'"
            )

        first = load_array(sample1)
        _check_sample(first, "sample1")

        if sample2 is None:
            if popmean is None:
                return err("t_test failed: either sample2 or popmean must be provided")
            if not isinstance(popmean, numbers.Real) or isinstance(popmean, bool):
                return err("t_test failed: popmean must be a number")
            result = stats.ttest_1samp(first, float(popmean), alternative=alt)
            return ok(
                {
                    "test": "one-sample",
                    "statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "df": int(first.size - 1),
                }
            )

        second = load_array(sample2)
        _check_sample(second, "sample2")

        if paired:
            if first.size != second.size:
                return err("t_test failed: paired samples must have the same length")
            result = stats.ttest_rel(first, second, alternative=alt)
            return ok(
                {
                    "test": "paired",
                    "statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "df": int(first.size - 1),
                }
            )

        result = stats.ttest_ind(first, second, alternative=alt)
        return ok(
            {
                "test": "two-sample",
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "df": float(result.df),
            }
        )
    except Exception as exc:
        return err(f"t_test failed: {exc}")


def chi2_test(
    observed: list[list[float]] | list[float],
    expected: list[float] | None = None,
) -> dict:
    """Run a chi-square goodness-of-fit test (1-D) or independence test (2-D table)."""
    try:
        raw = observed
        if isinstance(raw, np.ndarray):
            if raw.ndim not in (1, 2):
                return err("chi2_test failed: observed must be one- or two-dimensional")
            is_2d = raw.ndim == 2
        elif isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                return err("chi2_test failed: observed must not be empty")
            is_2d = isinstance(raw[0], (list, tuple, np.ndarray))
        else:
            return err("chi2_test failed: observed must be a list or numpy array")

        if is_2d:
            if expected is not None:
                return err(
                    "chi2_test failed: expected is not accepted for a 2-D contingency "
                    "table (independence test)"
                )
            table = np.asarray(raw, dtype=float)
            if table.ndim != 2:
                return err("chi2_test failed: observed must be a rectangular 2-D table")
            if table.shape[0] < 2 or table.shape[1] < 2:
                return err("chi2_test failed: a contingency table needs at least 2x2 cells")
            if not np.all(np.isfinite(table)):
                return err("chi2_test failed: observed contains non-finite values")
            if np.any(table < 0):
                return err("chi2_test failed: observed counts must be non-negative")
            statistic, p_value, dof, expected_freq = stats.chi2_contingency(table)
            return ok(
                {
                    "test": "independence",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "dof": int(dof),
                    "expected": expected_freq,
                }
            )

        arr = _as_float_array(raw, "observed")
        if arr.ndim != 1 or arr.size == 0:
            return err("chi2_test failed: observed must be a non-empty 1-D list")
        if not np.all(np.isfinite(arr)):
            return err("chi2_test failed: observed contains non-finite values")
        if np.any(arr < 0):
            return err("chi2_test failed: observed counts must be non-negative")

        if expected is not None:
            expected_arr = _as_float_array(expected, "expected")
            if expected_arr.ndim != 1 or expected_arr.size != arr.size:
                return err(
                    "chi2_test failed: expected must be one-dimensional and match "
                    "the length of observed"
                )
            if not np.all(np.isfinite(expected_arr)) or np.any(expected_arr <= 0):
                return err("chi2_test failed: expected frequencies must be positive")
            statistic, p_value = stats.chisquare(arr, expected_arr)
            return ok(
                {
                    "test": "goodness-of-fit",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "dof": int(arr.size - 1),
                    "expected": [float(value) for value in expected_arr],
                }
            )

        statistic, p_value = stats.chisquare(arr)
        return ok(
            {
                "test": "goodness-of-fit",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "dof": int(arr.size - 1),
            }
        )
    except Exception as exc:
        return err(f"chi2_test failed: {exc}")


def anova(groups: list[list[float]]) -> dict:
    """Run a one-way ANOVA and return the F statistic, p value and degrees of freedom.

    ``groups`` is a nested list of samples; ``"@<path>"`` file references are
    not supported for this parameter.
    """
    try:
        if not isinstance(groups, (list, tuple)) or len(groups) < 2:
            return err("anova failed: at least two groups are required")

        arrays: list[np.ndarray] = []
        for index, group in enumerate(groups):
            arr = _as_float_array(group, f"groups[{index}]")
            _check_sample(arr, f"groups[{index}]")
            arrays.append(arr)

        combined = np.concatenate(arrays)
        if float(np.ptp(combined)) == 0.0:
            return err(
                "anova failed: all observations are identical; the F statistic is undefined"
            )

        result = stats.f_oneway(*arrays)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            return err("anova failed: the F statistic is undefined for these groups")

        n_groups = len(arrays)
        n_total = int(combined.size)
        return ok(
            {
                "statistic": statistic,
                "p_value": p_value,
                "df_between": int(n_groups - 1),
                "df_within": int(n_total - n_groups),
            }
        )
    except Exception as exc:
        return err(f"anova failed: {exc}")


def correlation(
    x: list[float] | str, y: list[float] | str, method: str = "pearson"
) -> dict:
    """Compute the Pearson, Spearman or Kendall correlation coefficient and its p value.

    ``x`` and ``y`` may be passed inline or as ``"@<path>"`` file references.
    """
    try:
        name = str(method).strip().lower()
        if name not in _CORRELATION_METHODS:
            return err(
                "correlation failed: method must be 'pearson', 'spearman' or 'kendall'"
            )

        xarr = load_array(x)
        yarr = load_array(y)
        _check_sample(xarr, "x")
        _check_sample(yarr, "y")
        if xarr.size != yarr.size:
            return err("correlation failed: x and y must have the same length")
        if float(np.ptp(xarr)) == 0.0 or float(np.ptp(yarr)) == 0.0:
            return err("correlation failed: correlation is undefined for constant input")

        func = {
            "pearson": stats.pearsonr,
            "spearman": stats.spearmanr,
            "kendall": stats.kendalltau,
        }[name]
        result = func(xarr, yarr)
        r_value = float(result.statistic)
        p_value = float(result.pvalue)
        if not np.isfinite(r_value) or not np.isfinite(p_value):
            return err("correlation failed: the correlation coefficient is undefined")

        return ok({"method": name, "r": r_value, "p_value": p_value})
    except Exception as exc:
        return err(f"correlation failed: {exc}")


def linear_regression(
    x: list[float] | list[list[float]] | str,
    y: list[float] | str,
    feature_names: list[str] | None = None,
) -> dict:
    """Fit an ordinary least-squares regression, simple or multiple, and report coefficient inference.

    ``x`` and ``y`` may be passed inline or as ``"@<path>"`` file references;
    a file for ``x`` may contain a one- or two-dimensional array.
    """
    try:
        x = load_array(x)
        yarr = load_array(y)
        if yarr.ndim != 1:
            return err("linear_regression failed: y must be one-dimensional")
        if yarr.size < 2:
            return err("linear_regression failed: y must contain at least two values")
        if not np.all(np.isfinite(yarr)):
            return err("linear_regression failed: y contains non-finite values")
        if float(np.ptp(yarr)) == 0.0:
            return err(
                "linear_regression failed: y is constant; R-squared is undefined"
            )

        if isinstance(x, np.ndarray):
            if x.ndim not in (1, 2):
                return err("linear_regression failed: x must be one- or two-dimensional")
            multivariate = x.ndim == 2
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return err("linear_regression failed: x must not be empty")
            multivariate = isinstance(x[0], (list, tuple, np.ndarray))
        else:
            return err("linear_regression failed: x must be a list or numpy array")

        if not multivariate:
            xarr = _as_float_array(x, "x")
            _check_sample(xarr, "x", minimum=2)
            if xarr.size != yarr.size:
                return err(
                    "linear_regression failed: x and y must have the same length"
                )
            if float(np.ptp(xarr)) == 0.0:
                return err("linear_regression failed: x is constant")

            result = stats.linregress(xarr, yarr)
            n = int(xarr.size)
            df = n - 2
            slope = float(result.slope)
            intercept = float(result.intercept)
            slope_se = float(result.stderr)
            intercept_se = float(result.intercept_stderr)
            intercept_t, intercept_p = _t_and_p(intercept, intercept_se, df)
            slope_t, slope_p = _t_and_p(slope, slope_se, df)

            return ok(
                {
                    "coefficients": {"intercept": intercept, "x0": slope},
                    "std_errors": {"intercept": intercept_se, "x0": slope_se},
                    "t_values": {"intercept": intercept_t, "x0": slope_t},
                    "p_values": {"intercept": intercept_p, "x0": slope_p},
                    "r_squared": float(result.rvalue**2),
                    "n": n,
                }
            )

        try:
            design = np.asarray(list(x), dtype=float)
        except (TypeError, ValueError) as exc:
            return err(
                "linear_regression failed: x must be a rectangular 2-D numeric array"
            )
        if design.ndim != 2:
            return err("linear_regression failed: x must be a rectangular 2-D array")
        n_obs, n_features = design.shape
        if n_features == 0:
            return err("linear_regression failed: x must contain at least one feature")
        if not np.all(np.isfinite(design)):
            return err("linear_regression failed: x contains non-finite values")
        if n_obs != yarr.size:
            return err(
                "linear_regression failed: x and y must have the same number of observations"
            )

        n_params = n_features + 1  # intercept plus one slope per feature
        if n_obs <= n_params:
            return err(
                "linear_regression failed: too few observations "
                f"(n={n_obs}, parameters={n_params}); need n > parameters"
            )

        if feature_names is not None:
            if not isinstance(feature_names, (list, tuple)):
                return err("linear_regression failed: feature_names must be a list")
            if len(feature_names) != n_features:
                return err(
                    "linear_regression failed: feature_names must contain exactly "
                    f"{n_features} names"
                )
            names = [str(item) for item in feature_names]
            if any(not item for item in names) or len(set(names)) != len(names):
                return err(
                    "linear_regression failed: feature_names must be non-empty and unique"
                )
        else:
            names = [f"x{index}" for index in range(n_features)]

        matrix = np.column_stack([np.ones(n_obs), design])
        beta, _residuals, _rank, _singular = np.linalg.lstsq(matrix, yarr, rcond=None)
        fitted = matrix @ beta
        residuals = yarr - fitted
        sse = float(residuals @ residuals)
        df = n_obs - n_params
        sigma2 = sse / df

        try:
            covariance = sigma2 * np.linalg.inv(matrix.T @ matrix)
        except np.linalg.LinAlgError:
            return err(
                "linear_regression failed: design matrix is singular "
                "(features are collinear)"
            )
        std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

        t_values: list[float] = []
        p_values: list[float] = []
        for estimate, standard_error in zip(beta, std_errors):
            t_value, p_value = _t_and_p(float(estimate), float(standard_error), int(df))
            t_values.append(t_value)
            p_values.append(p_value)

        ss_tot = float(np.sum((yarr - np.mean(yarr)) ** 2))
        r_squared = 1.0 - sse / ss_tot if ss_tot > 0.0 else 1.0

        coefficient_keys = ["intercept", *names]
        return ok(
            {
                "coefficients": {
                    key: float(value) for key, value in zip(coefficient_keys, beta)
                },
                "std_errors": {
                    key: float(value) for key, value in zip(coefficient_keys, std_errors)
                },
                "t_values": dict(zip(coefficient_keys, t_values)),
                "p_values": dict(zip(coefficient_keys, p_values)),
                "r_squared": float(r_squared),
                "n": int(n_obs),
            }
        )
    except Exception as exc:
        return err(f"linear_regression failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the statistical inference tools on the MCP server."""
    mcp.tool()(t_test)
    mcp.tool()(chi2_test)
    mcp.tool()(anova)
    mcp.tool()(correlation)
    mcp.tool()(linear_regression)
