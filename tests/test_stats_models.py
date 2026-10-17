"""Tests for the statistical model and time-series tools (SM-)."""

import json
import math
from pathlib import Path

import numpy as np

from sci_mcp.tools.stats_models import (
    arima,
    glm,
    register,
    stationarity_test,
    time_series_decompose,
)


def _assert_json_safe(payload):
    """Every envelope must survive strict JSON encoding (no NaN/Infinity)."""
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------


def test_glm_logistic_positive_slope_and_significant():
    rng = np.random.default_rng(7)
    x = np.linspace(-2.0, 2.0, 100)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + 1.3 * x)))
    y = (rng.random(100) < probability).astype(int)

    result = glm("binomial", "y ~ x", {"x": x.tolist(), "y": y.tolist()})
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["family"] == "binomial"
    assert out["n"] == 100
    assert out["coefficients"]["x"] > 0.0
    assert out["p_values"]["x"] < 0.05
    assert out["std_errors"]["x"] > 0.0
    assert math.isfinite(out["aic"])
    assert math.isfinite(out["bic"])
    assert out["pseudo_r2"] is not None
    assert math.isfinite(out["pseudo_r2"])


def test_glm_poisson_counts():
    rng = np.random.default_rng(11)
    x = np.linspace(-1.5, 1.5, 80)
    y = rng.poisson(np.exp(0.5 + 0.4 * x))

    result = glm("poisson", "y ~ x", {"x": x.tolist(), "y": y.tolist()})
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["family"] == "poisson"
    assert out["n"] == 80
    assert out["coefficients"]["x"] > 0.0
    assert math.isfinite(out["aic"])
    assert math.isfinite(out["bic"])


def test_glm_gaussian_matches_least_squares():
    rng = np.random.default_rng(3)
    x = np.linspace(-3.0, 3.0, 60)
    y = 1.5 - 2.0 * x + rng.normal(0.0, 0.3, 60)

    result = glm(
        "gaussian", "y ~ x", {"x": x.tolist(), "y": y.tolist()}
    )
    assert result["ok"] is True
    out = result["result"]

    slope, intercept = np.polyfit(x, y, 1)
    assert abs(out["coefficients"]["x"] - slope) < 1e-8
    assert abs(out["coefficients"]["Intercept"] - intercept) < 1e-8


def test_glm_reads_csv_file(sci_output_dir):
    rng = np.random.default_rng(5)
    x = np.linspace(-2.0, 2.0, 40)
    y = 0.5 + 1.1 * x + rng.normal(0.0, 0.2, 40)
    lines = ["x,y"] + [f"{float(xi)},{float(yi)}" for xi, yi in zip(x, y)]
    (Path(sci_output_dir) / "glm_input.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    result = glm("gaussian", "y ~ x", "@glm_input.csv")
    _assert_json_safe(result)

    assert result["ok"] is True
    assert result["result"]["n"] == 40
    assert abs(result["result"]["coefficients"]["x"] - 1.1) < 0.05


def test_glm_rejects_unknown_family():
    result = glm("tweedie", "y ~ x", {"x": [1, 2, 3], "y": [1, 2, 3]})
    assert result["ok"] is False
    assert "unknown family" in result["error"]


def test_glm_rejects_invalid_formula():
    result = glm("gaussian", "y ~", {"x": [1, 2, 3], "y": [1, 2, 3]})
    assert result["ok"] is False
    assert "glm failed" in result["error"]


def test_glm_rejects_missing_variable():
    result = glm("gaussian", "y ~ z", {"x": [1, 2, 3], "y": [1, 2, 3]})
    assert result["ok"] is False
    assert "glm failed" in result["error"]


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def _ar1_series(n=150, phi=0.7, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.5, n)
    series = np.zeros(n)
    for index in range(1, n):
        series[index] = phi * series[index - 1] + noise[index]
    return series


def test_arima_ar1_fit_and_forecast():
    series = _ar1_series()
    result = arima(series.tolist(), order=[1, 0, 0], forecast_steps=8)
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["order"] == [1, 0, 0]
    assert 0.4 < out["params"]["ar.L1"] < 0.95
    assert math.isfinite(out["aic"])
    assert math.isfinite(out["bic"])
    assert len(out["forecast"]) == 8
    assert len(out["conf_int"]) == 8
    for value in out["forecast"]:
        assert math.isfinite(value)
        assert abs(value) < 5.0
    for low, high in out["conf_int"]:
        assert math.isfinite(low) and math.isfinite(high)
        assert low <= high


def test_arima_zero_forecast_returns_empty_lists():
    series = _ar1_series(seed=1)
    result = arima(series.tolist(), order=[1, 0, 0], forecast_steps=0)
    assert result["ok"] is True
    assert result["result"]["forecast"] == []
    assert result["result"]["conf_int"] == []


def test_arima_rejects_wrong_order_length():
    series = _ar1_series(seed=2)
    result = arima(series.tolist(), order=[1, 0])
    assert result["ok"] is False
    assert "order" in result["error"]


def test_arima_rejects_negative_order_term():
    series = _ar1_series(seed=3)
    result = arima(series.tolist(), order=[1, -1, 0])
    assert result["ok"] is False
    assert "order" in result["error"]


def test_arima_reads_at_file_series(sci_output_dir):
    series = _ar1_series(n=120, seed=47)
    np.savetxt(
        Path(sci_output_dir) / "arima_series.csv",
        series,
        header="value",
        comments="",
    )

    result = arima("@arima_series.csv", order=[1, 0, 0], forecast_steps=4)
    _assert_json_safe(result)

    assert result["ok"] is True
    assert len(result["result"]["forecast"]) == 4


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------


def test_time_series_decompose_periodic_pattern(tmp_path, sci_output_dir):
    rng = np.random.default_rng(13)
    n = 160
    pattern = np.array([1.0, -1.0, 0.5, -0.5])
    seasonal = np.tile(pattern, n // 4)
    values = 10.0 + 0.01 * np.arange(n) + seasonal + 0.001 * rng.normal(size=n)

    result = time_series_decompose(values.tolist(), period=4)
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["period"] == 4
    assert out["model"] == "additive"
    assert len(out["seasonal"]) == n
    assert len(out["trend"]) == n
    assert len(out["resid"]) == n
    assert len(out["observed_preview"]) == n
    assert out["seasonal"][:4] == out["seasonal"][4:8]
    assert np.allclose(out["seasonal"][:4], pattern, atol=0.02)
    # The first/last trend values are NaN and ok() converts them to null.
    assert out["trend"][0] is None

    path = Path(out["path"])
    assert path.is_file()
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {"observed", "trend", "seasonal", "resid"}
        assert saved["seasonal"].shape == (n,)


def test_time_series_decompose_downsamples_long_output(sci_output_dir):
    rng = np.random.default_rng(17)
    n = 3000
    values = 5.0 + np.sin(2.0 * np.pi * np.arange(n) / 4.0) + 0.01 * rng.normal(size=n)

    result = time_series_decompose(values.tolist(), period=4)
    assert result["ok"] is True
    out = result["result"]
    assert len(out["seasonal"]) <= 2000
    assert len(out["trend"]) <= 2000
    assert len(out["resid"]) <= 2000
    assert out["downsampled"] is True
    assert out["original_length"] == n
    assert Path(out["path"]).is_file()


def test_time_series_decompose_rejects_too_large_period():
    values = [_ for _ in range(50)]
    result = time_series_decompose(values, period=50)
    assert result["ok"] is False
    assert "period" in result["error"]

    result = time_series_decompose(values, period=1)
    assert result["ok"] is False
    assert "period" in result["error"]


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------


def test_stationarity_white_noise_is_stationary():
    rng = np.random.default_rng(23)
    series = rng.normal(size=300)
    result = stationarity_test(series.tolist())
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["p_value"] < 0.05
    assert out["stationary"] is True
    assert set(out["critical_values"]) == {"1%", "5%", "10%"}
    assert out["nobs"] > 0


def test_stationarity_strong_trend_is_not_stationary():
    rng = np.random.default_rng(29)
    series = 0.1 * np.arange(250) + rng.normal(0.0, 1.0, 250)
    result = stationarity_test(series.tolist(), regression="c", autolag="AIC")
    assert result["ok"] is True
    out = result["result"]
    assert out["p_value"] > 0.05
    assert out["stationary"] is False


def test_stationarity_accepts_autolag_none():
    rng = np.random.default_rng(31)
    series = rng.normal(size=200)
    result = stationarity_test(series.tolist(), regression="n", autolag=None)
    assert result["ok"] is True
    assert result["result"]["stationary"] is True


# ---------------------------------------------------------------------------
# Envelope and registration
# ---------------------------------------------------------------------------


def test_all_outputs_are_strict_json_serializable():
    rng = np.random.default_rng(37)
    x = np.linspace(-2.0, 2.0, 80)
    probability = 1.0 / (1.0 + np.exp(-(0.2 + x)))
    y = (rng.random(80) < probability).astype(int)
    series = _ar1_series(n=120, seed=41)

    results = [
        glm("binomial", "y ~ x", {"x": x.tolist(), "y": y.tolist()}),
        arima(series.tolist(), order=[1, 0, 0], forecast_steps=3),
        time_series_decompose(
            (10.0 + np.sin(2.0 * np.pi * np.arange(80) / 4.0)).tolist(), period=4
        ),
        stationarity_test(rng.normal(size=200).tolist()),
    ]
    for result in results:
        assert result["ok"] is True
        json.dumps(result, allow_nan=False)


class _DummyMCP:
    """Minimal stand-in for MCPServer that records registered tool functions."""

    def __init__(self) -> None:
        self.tools = []

    def tool(self):
        def decorator(func):
            self.tools.append(func)
            return func

        return decorator


def test_register_registers_all_tools():
    dummy = _DummyMCP()
    assert register(dummy) is None
    assert [tool.__name__ for tool in dummy.tools] == [
        "glm",
        "arima",
        "time_series_decompose",
        "stationarity_test",
    ]
