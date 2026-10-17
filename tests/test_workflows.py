"""Tests for the high-level workflow tools: dataset analysis and forecasting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sci_mcp.tools.workflows import analyze_dataset, forecast


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SCI_MCP_OUTPUT_DIR at a temporary directory and return it."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def _write_csv(directory: Path, name: str, text: str) -> Path:
    """Write a CSV fixture and return its absolute path."""
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# x, y, z are numeric and y = 2 * x exactly; label is non-numeric.
BASE_CSV = (
    "x,y,z,label\n"
    "1,2,5,a\n"
    "2,4,3,b\n"
    "3,6,4,c\n"
    "4,8,1,d\n"
    "5,10,2,e\n"
)


# ---------------------------------------------------------------------------
# analyze_dataset: summaries and column handling
# ---------------------------------------------------------------------------


def test_analyze_dataset_summary(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    response = analyze_dataset(str(path), plot=False)
    assert response["ok"] is True
    payload = response["result"]

    assert payload["path"] == str(path.resolve())
    assert payload["n_rows"] == 5
    assert payload["numeric_columns"] == ["x", "y", "z"]

    summary = payload["summary"]["x"]
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(3.0, abs=1e-12)
    assert summary["std"] == pytest.approx(math.sqrt(2.5), abs=1e-12)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(5.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["q1"] == pytest.approx(2.0)
    assert summary["q3"] == pytest.approx(4.0)


def test_analyze_dataset_reports_non_numeric_columns_as_ignored(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    payload = analyze_dataset(str(path), plot=False)["result"]

    assert payload["ignored_columns"] == ["label"]
    assert "label" not in payload["summary"]
    assert "label" not in payload["numeric_columns"]


def test_analyze_dataset_target_correlations_and_regression(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    payload = analyze_dataset(str(path), target="y", plot=False)["result"]
    target_analysis = payload["target_analysis"]

    assert target_analysis["target"] == "y"
    correlations = {item["column"]: item["r"] for item in target_analysis["correlations"]}
    assert correlations["x"] == pytest.approx(1.0, abs=1e-12)

    regression = target_analysis["regression"]
    assert regression["coefficients"]["intercept"] == pytest.approx(0.0, abs=1e-8)
    assert regression["coefficients"]["x"] == pytest.approx(2.0, abs=1e-8)
    assert regression["coefficients"]["z"] == pytest.approx(0.0, abs=1e-8)
    assert regression["r_squared"] == pytest.approx(1.0, abs=1e-10)


def test_analyze_dataset_target_accepts_column_index(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    response = analyze_dataset(str(path), target=1, plot=False)

    assert response["ok"] is True
    assert response["result"]["target_analysis"]["target"] == "y"


def test_analyze_dataset_non_numeric_target_is_reported_not_fatal(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    response = analyze_dataset(str(path), target="label", plot=False)

    assert response["ok"] is True
    target_analysis = response["result"]["target_analysis"]
    assert target_analysis["target"] == "label"
    assert target_analysis["correlations"] == []
    assert target_analysis["regression"] is None
    assert "note" in target_analysis


def test_analyze_dataset_missing_target_fails(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    response = analyze_dataset(str(path), target="nope", plot=False)

    assert response["ok"] is False
    assert "target" in response["error"]


def test_analyze_dataset_correlation_matrix_diagonal_is_one(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    correlations = analyze_dataset(str(path), plot=False)["result"]["correlations"]

    assert correlations["matrix_columns"] == ["x", "y", "z"]
    matrix = correlations["matrix"]
    assert len(matrix) == 3
    for index in range(3):
        assert matrix[index][index] == pytest.approx(1.0, abs=1e-12)

    pairs = correlations["pairs"]
    assert pairs[0]["a"] == "x" and pairs[0]["b"] == "y"
    assert pairs[0]["r"] == pytest.approx(1.0, abs=1e-12)
    assert len(pairs) <= 10


def test_analyze_dataset_max_columns_caps_analysis(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    payload = analyze_dataset(str(path), max_columns=2, plot=False)["result"]

    assert payload["numeric_columns"] == ["x", "y"]
    assert set(payload["summary"]) == {"x", "y"}
    assert payload["correlations"]["matrix_columns"] == ["x", "y"]
    assert len(payload["correlations"]["matrix"]) == 2


# ---------------------------------------------------------------------------
# analyze_dataset: plotting
# ---------------------------------------------------------------------------


def test_analyze_dataset_plot_true_writes_files(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    payload = analyze_dataset(str(path), plot=True)["result"]

    assert len(payload["plots"]) >= 1
    for item in payload["plots"]:
        assert item["kind"] in {"hist", "correlation_heatmap"}
        assert Path(item["path"]).is_file()


def test_analyze_dataset_plot_false_writes_nothing(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    payload = analyze_dataset(str(path), plot=False)["result"]

    assert payload["plots"] == []
    assert list(out_dir.glob("*.png")) == []


def test_analyze_dataset_many_columns_uses_one_heatmap(out_dir: Path) -> None:
    header = ",".join(f"c{index}" for index in range(7))
    rows = [
        ",".join(str((row + 1) * (column + 1)) for column in range(7))
        for row in range(10)
    ]
    path = _write_csv(out_dir, "wide.csv", "\n".join([header, *rows]) + "\n")

    payload = analyze_dataset(str(path), plot=True)["result"]

    assert len(payload["numeric_columns"]) == 7
    assert len(payload["plots"]) == 1
    assert payload["plots"][0]["kind"] == "correlation_heatmap"
    assert Path(payload["plots"][0]["path"]).is_file()


# ---------------------------------------------------------------------------
# forecast: normal paths
# ---------------------------------------------------------------------------


def test_forecast_arima_on_trend(out_dir: Path) -> None:
    series = [float(value) for value in range(1, 41)]

    response = forecast(series, steps=5, method="arima", order=[1, 1, 0])
    assert response["ok"] is True
    payload = response["result"]

    assert payload["method"] == "arima"
    assert payload["steps"] == 5
    assert len(payload["forecast"]) == 5
    assert len(payload["conf_int"]) == 5

    # The series rises by one per step; a random-walk-with-drift forecast
    # continues above the last observation and keeps increasing.
    assert payload["forecast"][0] > 40.0
    for earlier, later in zip(payload["forecast"], payload["forecast"][1:]):
        assert later > earlier
    for value, (low, high) in zip(payload["forecast"], payload["conf_int"]):
        assert low <= value <= high
    assert "aic" in payload and "bic" in payload


def test_forecast_arima_default_order(out_dir: Path) -> None:
    series = [float(value) for value in range(1, 41)]

    response = forecast(series, steps=4)

    assert response["ok"] is True
    payload = response["result"]
    assert payload["model_info"]["order"] == [1, 0, 0]
    assert payload["model_info"]["seasonal_order"] is None
    assert len(payload["forecast"]) == 4


def test_forecast_exp_smoothing_on_trend(out_dir: Path) -> None:
    series = [float(value) for value in range(1, 41)]

    response = forecast(series, steps=6, method="exp_smoothing")
    assert response["ok"] is True
    payload = response["result"]

    assert payload["method"] == "exp_smoothing"
    assert len(payload["forecast"]) == 6
    assert len(payload["conf_int"]) == 6
    assert payload["forecast"][0] > 40.0
    assert payload["model_info"]["seasonal"] is None


def test_forecast_arima_seasonal_order(out_dir: Path) -> None:
    series = [
        10.0 + 0.2 * index + math.sin(2.0 * math.pi * index / 12.0)
        for index in range(48)
    ]

    response = forecast(
        series, steps=6, method="arima", order=[1, 0, 0], seasonal_period=12
    )
    assert response["ok"] is True
    payload = response["result"]

    assert len(payload["forecast"]) == 6
    assert payload["model_info"]["seasonal_order"] == [1, 0, 0, 12]


def test_forecast_exp_smoothing_seasonal(out_dir: Path) -> None:
    series = [
        10.0 + 0.2 * index + math.sin(2.0 * math.pi * index / 12.0)
        for index in range(48)
    ]

    response = forecast(series, steps=6, method="exp_smoothing", seasonal_period=12)
    assert response["ok"] is True
    payload = response["result"]

    assert len(payload["forecast"]) == 6
    assert payload["model_info"]["seasonal"] == "add"
    assert payload["model_info"]["seasonal_periods"] == 12


def test_forecast_accepts_file_reference(out_dir: Path) -> None:
    series = [float(value) for value in range(1, 41)]
    (out_dir / "series.json").write_text(json.dumps(series), encoding="utf-8")

    response = forecast("@series.json", steps=5, method="exp_smoothing")

    assert response["ok"] is True
    assert len(response["result"]["forecast"]) == 5


def test_forecast_down_samples_long_horizon(out_dir: Path) -> None:
    series = [float(value) for value in range(1, 31)]

    payload = forecast(series, steps=2500, method="exp_smoothing")["result"]

    assert payload["steps"] == 2500
    assert len(payload["forecast"]) == 2000
    assert len(payload["conf_int"]) == 2000
    assert payload["downsampled"] is True
    assert payload["original_length"] == 2500


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_analyze_dataset_missing_file_fails(out_dir: Path) -> None:
    response = analyze_dataset(str(out_dir / "missing.csv"), plot=False)

    assert response["ok"] is False
    assert "error" in response


def test_analyze_dataset_without_numeric_columns_fails(out_dir: Path) -> None:
    path = _write_csv(out_dir, "text.csv", "name,city\na,x\nb,y\n")

    response = analyze_dataset(str(path), plot=False)

    assert response["ok"] is False
    assert "numeric" in response["error"]


def test_forecast_rejects_zero_steps(out_dir: Path) -> None:
    response = forecast([1.0, 2.0, 3.0, 4.0], steps=0)

    assert response["ok"] is False
    assert "steps" in response["error"]


@pytest.mark.parametrize("steps", [-1, 0, 10001, 1.5, True])
def test_forecast_rejects_invalid_steps(out_dir: Path, steps: object) -> None:
    response = forecast([1.0, 2.0, 3.0, 4.0], steps=steps)  # type: ignore[arg-type]

    assert response["ok"] is False
    assert "error" in response


def test_forecast_rejects_unknown_method(out_dir: Path) -> None:
    response = forecast([1.0, 2.0, 3.0, 4.0], method="prophet")

    assert response["ok"] is False
    assert "method" in response["error"]


def test_forecast_rejects_short_series(out_dir: Path) -> None:
    response = forecast([1.0, 2.0, 3.0], steps=2)

    assert response["ok"] is False
    assert "4" in response["error"]


def test_forecast_rejects_invalid_order(out_dir: Path) -> None:
    response = forecast([1.0, 2.0, 3.0, 4.0], method="arima", order=[1, 0])

    assert response["ok"] is False
    assert "order" in response["error"]


def test_forecast_rejects_invalid_alpha(out_dir: Path) -> None:
    response = forecast([1.0, 2.0, 3.0, 4.0], conf_alpha=1.5)

    assert response["ok"] is False
    assert "conf_alpha" in response["error"]


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_outputs_are_strict_json_serializable(out_dir: Path) -> None:
    path = _write_csv(out_dir, "data.csv", BASE_CSV)

    analysis = analyze_dataset(str(path), target="y", plot=True)
    prediction = forecast([float(value) for value in range(1, 41)], steps=5)

    json.dumps(analysis, allow_nan=False)
    json.dumps(prediction, allow_nan=False)


def test_constant_column_serializes_as_null(out_dir: Path) -> None:
    path = _write_csv(out_dir, "constant.csv", "x,y\n2,1\n2,2\n2,3\n2,4\n2,5\n")

    response = analyze_dataset(str(path), plot=False)

    assert response["ok"] is True
    payload = response["result"]
    assert payload["summary"]["x"]["std"] == 0.0
    # Undefined correlations become null instead of NaN/Infinity.
    assert payload["correlations"]["matrix"][0][0] is None
    assert "warnings" in response
    json.dumps(response, allow_nan=False)
