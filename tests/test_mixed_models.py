"""Tests for the mixed-model tools (MM-): linear mixed models and GEE."""

import json
from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.mixed_models import gee, mixed_lm, register


def _assert_json_safe(payload):
    """Every envelope must survive strict JSON encoding (no NaN/Infinity)."""
    json.dumps(payload, allow_nan=False)


# Fixed group offsets; their population variance is 2.0.
_GROUP_OFFSETS = np.array([0.0, 1.0, -1.0, 2.0, -2.0])


def _simulate_mixed(seed=20261017, n_per_group=10):
    """5 groups x 10 observations from ``y = 2*x + offset + noise``."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.array(list("ABCDE")), n_per_group)
    x = rng.normal(0.0, 1.0, groups.size)
    offsets = np.repeat(_GROUP_OFFSETS, n_per_group)
    y = 2.0 * x + offsets + rng.normal(0.0, 0.2, groups.size)
    return {"x": x.tolist(), "y": y.tolist(), "g": groups.tolist()}


def _simulate_binary(seed=11, n_per_group=20):
    """Binary outcomes with a logistic signal in x."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.array(list("ABCDE")), n_per_group)
    x = rng.normal(0.0, 1.0, groups.size)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + 1.0 * x)))
    y = (rng.random(groups.size) < probability).astype(int)
    return {"x": x.tolist(), "y": y.tolist(), "g": groups.tolist()}


def _simulate_counts(seed=12):
    """Poisson counts with a log-linear signal in x."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.array(list("ABCDE")), 16)
    x = np.linspace(-1.5, 1.5, groups.size)
    y = rng.poisson(np.exp(0.5 + 0.4 * x))
    return {"x": x.tolist(), "y": y.tolist(), "g": groups.tolist()}


def _write_csv(path: Path, data: dict) -> None:
    lines = ["g,x,y"]
    for group, x_value, y_value in zip(data["g"], data["x"], data["y"]):
        lines.append(f"{group},{float(x_value)!r},{float(y_value)!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Linear mixed model
# ---------------------------------------------------------------------------


def test_mixed_lm_recovers_fixed_slope():
    result = mixed_lm("y ~ x", _simulate_mixed(), groups="g")
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["formula"] == "y ~ x"
    assert out["groups"] == "g"
    assert out["n"] == 50
    assert out["n_groups"] == 5
    assert out["converged"] is True
    assert abs(out["fixed_effects"]["x"] - 2.0) < 0.2
    assert out["fixed_effects"]["Intercept"] == pytest.approx(0.0, abs=0.5)
    assert out["std_errors"]["x"] > 0.0
    assert out["p_values"]["x"] < 0.05
    # Only fixed effects are reported; the variance parameter is not.
    assert set(out["fixed_effects"]) == {"Intercept", "x"}
    assert set(out["std_errors"]) == {"Intercept", "x"}
    assert set(out["z_values"]) == {"Intercept", "x"}
    assert set(out["p_values"]) == {"Intercept", "x"}
    assert "aic" in out and "bic" in out


def test_mixed_lm_random_effect_variance_is_positive():
    result = mixed_lm("y ~ x", _simulate_mixed(seed=7), groups="g")
    assert result["ok"] is True
    variance = result["result"]["group_variance"]
    assert variance is not None
    assert variance > 0.5


def test_mixed_lm_accepts_re_formula_intercept():
    result = mixed_lm(
        "y ~ x", _simulate_mixed(seed=3), groups="g", re_formula="~1"
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert abs(out["fixed_effects"]["x"] - 2.0) < 0.2
    assert out["group_variance"] is not None
    assert out["group_variance"] > 0.0


def test_mixed_lm_ml_fit_reports_aic_bic():
    result = mixed_lm("y ~ x", _simulate_mixed(seed=4), groups="g", reml=False)
    assert result["ok"] is True
    out = result["result"]
    assert np.isfinite(out["aic"])
    assert np.isfinite(out["bic"])


def test_mixed_lm_reads_at_csv(sci_output_dir):
    data = _simulate_mixed(seed=9)
    _write_csv(Path(sci_output_dir) / "mixed_lm_input.csv", data)

    result = mixed_lm("y ~ x", "@mixed_lm_input.csv", groups="g")
    _assert_json_safe(result)

    assert result["ok"] is True
    assert result["result"]["n"] == 50
    assert abs(result["result"]["fixed_effects"]["x"] - 2.0) < 0.2


# ---------------------------------------------------------------------------
# Generalized estimating equations
# ---------------------------------------------------------------------------


def test_gee_gaussian_exchangeable_matches_ols():
    data = _simulate_mixed(seed=21)
    result = gee(
        "y ~ x", data, groups="g", family="gaussian", cov_struct="exchangeable"
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    slope, intercept = np.polyfit(np.asarray(data["x"]), np.asarray(data["y"]), 1)
    # Exchangeable GEE weights clusters, so its point estimates are close to
    # (but not exactly equal to) ordinary least squares.
    assert abs(out["coefficients"]["x"] - slope) < 0.15
    assert abs(out["coefficients"]["Intercept"] - intercept) < 0.15
    assert out["n"] == 50
    assert out["n_groups"] == 5
    assert out["family"] == "gaussian"
    assert out["cov_struct"] == "exchangeable"
    assert out["std_errors"]["x"] > 0.0
    assert out["p_values"]["x"] < 0.05


def test_gee_binomial_positive_effect():
    result = gee(
        "y ~ x",
        _simulate_binary(),
        groups="g",
        family="binomial",
        cov_struct="independence",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["family"] == "binomial"
    assert out["cov_struct"] == "independence"
    assert out["coefficients"]["x"] > 0.5
    assert out["p_values"]["x"] < 0.05


def test_gee_poisson_positive_effect():
    result = gee(
        "y ~ x",
        _simulate_counts(),
        groups="g",
        family="poisson",
        cov_struct="exchangeable",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["family"] == "poisson"
    assert out["n"] == 80
    assert out["n_groups"] == 5
    assert out["coefficients"]["x"] > 0.0
    assert out["p_values"]["x"] < 0.05


def test_gee_autoregressive_cov_struct_runs():
    result = gee(
        "y ~ x",
        _simulate_counts(seed=31),
        groups="g",
        family="poisson",
        cov_struct="autoregressive",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    assert result["result"]["cov_struct"] == "autoregressive"
    assert result["result"]["coefficients"]["x"] > 0.0


def test_gee_reads_at_csv(sci_output_dir):
    data = _simulate_mixed(seed=13)
    _write_csv(Path(sci_output_dir) / "gee_input.csv", data)

    result = gee("y ~ x", "@gee_input.csv", groups="g")
    _assert_json_safe(result)

    assert result["ok"] is True
    assert result["result"]["n"] == 50
    assert result["result"]["n_groups"] == 5


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_mixed_lm_rejects_missing_groups_column():
    result = mixed_lm("y ~ x", _simulate_mixed(), groups="cluster")
    assert result["ok"] is False
    assert "groups column" in result["error"]


def test_gee_rejects_missing_groups_column():
    result = gee("y ~ x", _simulate_mixed(), groups="cluster")
    assert result["ok"] is False
    assert "groups column" in result["error"]


def test_mixed_lm_rejects_missing_formula_variable():
    result = mixed_lm("y ~ z", _simulate_mixed(), groups="g")
    assert result["ok"] is False
    assert "mixed_lm failed" in result["error"]


def test_gee_rejects_missing_formula_variable():
    result = gee("y ~ z", _simulate_mixed(), groups="g")
    assert result["ok"] is False
    assert "gee failed" in result["error"]


def test_gee_rejects_unknown_family():
    result = gee("y ~ x", _simulate_mixed(), groups="g", family="tweedie")
    assert result["ok"] is False
    assert "unknown family" in result["error"]


def test_gee_rejects_unknown_cov_struct():
    result = gee(
        "y ~ x", _simulate_mixed(), groups="g", cov_struct="toeplitz"
    )
    assert result["ok"] is False
    assert "unknown cov_struct" in result["error"]


def test_mixed_lm_rejects_invalid_formula():
    result = mixed_lm("y ~", _simulate_mixed(), groups="g")
    assert result["ok"] is False
    assert "mixed_lm failed" in result["error"]


def test_gee_rejects_invalid_formula():
    result = gee("y ~", _simulate_mixed(), groups="g")
    assert result["ok"] is False
    assert "gee failed" in result["error"]


def test_register_registers_both_tools():
    class _FakeServer:
        def __init__(self):
            self.registered = []

        def tool(self):
            def decorator(function):
                self.registered.append(function)
                return function

            return decorator

    server = _FakeServer()
    register(server)
    assert server.registered == [mixed_lm, gee]
