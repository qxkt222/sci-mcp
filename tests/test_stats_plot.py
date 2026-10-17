"""Tests for the statistics, distribution, curve fitting and plotting tools."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sci_mcp.tools.stats_plot import curve_fit, describe, distribution, plot


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_basic_stats() -> None:
    result = describe([1, 2, 3, 4])
    assert result["ok"] is True
    payload = result["result"]

    assert payload["n"] == 4
    assert payload["mean"] == pytest.approx(2.5, abs=1e-12)
    assert payload["median"] == pytest.approx(2.5, abs=1e-12)
    assert payload["std"] == pytest.approx(1.2909944487358056, abs=1e-9)
    assert payload["min"] == pytest.approx(1.0)
    assert payload["max"] == pytest.approx(4.0)
    assert payload["q1"] == pytest.approx(1.75, abs=1e-12)
    assert payload["q3"] == pytest.approx(3.25, abs=1e-12)
    assert "skew" in payload and "kurtosis" in payload


def test_describe_rejects_empty_data() -> None:
    result = describe([])
    assert result["ok"] is False
    assert "error" in result


def test_describe_rejects_non_numeric_data() -> None:
    result = describe([1.0, "two", 3.0])  # type: ignore[list-item]
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------


def test_distribution_ppf_median_of_norm_is_zero() -> None:
    result = distribution("ppf", "norm", x=0.5)
    assert result["ok"] is True
    assert result["result"]["values"] == pytest.approx(0.0, abs=1e-9)


def test_distribution_cdf_of_norm_at_zero_is_half() -> None:
    result = distribution("cdf", "norm", x=0.0)
    assert result["ok"] is True
    assert result["result"]["values"] == pytest.approx(0.5, abs=1e-12)


def test_distribution_pdf_of_norm_at_zero() -> None:
    result = distribution("pdf", "norm", x=0.0)
    assert result["ok"] is True
    assert result["result"]["values"] == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-12)


def test_distribution_pdf_accepts_list_and_params() -> None:
    result = distribution("pdf", "norm", params={"loc": 0.0, "scale": 2.0}, x=[0.0, 2.0])
    assert result["ok"] is True
    values = result["result"]["values"]
    assert len(values) == 2
    assert values[0] == pytest.approx(1.0 / (2.0 * math.sqrt(2.0 * math.pi)), abs=1e-12)


def test_distribution_rvs_is_reproducible_with_seed() -> None:
    first = distribution("rvs", "norm", n=10, seed=42)
    second = distribution("rvs", "norm", n=10, seed=42)

    assert first["ok"] is True and second["ok"] is True
    values = first["result"]["values"]
    assert len(values) == 10
    assert values == second["result"]["values"]


def test_distribution_fit_norm_recovers_mean_and_scale() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = distribution("fit", "norm", data=data)
    assert result["ok"] is True

    params = result["result"]["params"]
    # scipy's MLE scale is the population std: sqrt(2) for this sample.
    assert params["loc"] == pytest.approx(3.0, abs=1e-9)
    assert params["scale"] == pytest.approx(math.sqrt(2.0), abs=1e-9)


def test_distribution_rejects_unknown_distribution() -> None:
    result = distribution("pdf", "definitely_not_a_distribution", x=0.0)
    assert result["ok"] is False
    assert "error" in result


def test_distribution_ppf_rejects_probability_outside_unit_interval() -> None:
    result = distribution("ppf", "norm", x=1.5)
    assert result["ok"] is False
    assert "error" in result


def test_distribution_pdf_requires_x() -> None:
    result = distribution("pdf", "norm")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# curve_fit
# ---------------------------------------------------------------------------


def test_curve_fit_recovers_linear_coefficients() -> None:
    xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
    ydata = [2.0 * x + 1.0 for x in xdata]

    result = curve_fit("a*x + b", xdata, ydata)
    assert result["ok"] is True
    payload = result["result"]

    assert payload["params"]["a"] == pytest.approx(2.0, abs=1e-8)
    assert payload["params"]["b"] == pytest.approx(1.0, abs=1e-8)
    assert payload["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert set(payload["param_errors"]) == {"a", "b"}


def test_curve_fit_honours_param_names() -> None:
    xdata = [0.0, 1.0, 2.0, 3.0]
    ydata = [3.0 * x + 2.0 for x in xdata]

    result = curve_fit("a*x + b", xdata, ydata, param_names=["slope", "intercept"])
    assert result["ok"] is True
    payload = result["result"]
    assert set(payload["params"]) == {"slope", "intercept"}
    assert payload["params"]["slope"] == pytest.approx(3.0, abs=1e-8)
    assert payload["params"]["intercept"] == pytest.approx(2.0, abs=1e-8)


def test_curve_fit_rejects_mismatched_data_lengths() -> None:
    result = curve_fit("a*x + b", [0.0, 1.0, 2.0], [1.0, 2.0])
    assert result["ok"] is False
    assert "error" in result


def test_curve_fit_rejects_wrong_number_of_param_names() -> None:
    result = curve_fit("a*x + b", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], param_names=["a"])
    assert result["ok"] is False
    assert "error" in result


def test_curve_fit_rejects_invalid_expression() -> None:
    result = curve_fit("a*x +", [0.0, 1.0], [1.0, 2.0])
    assert result["ok"] is False
    assert "error" in result


def test_curve_fit_accepts_sympy_builtin_like_param_names() -> None:
    result = curve_fit(
        "beta*x + gamma",
        xdata=[0, 1, 2],
        ydata=[1, 3, 5],
        p0=[1, 1],
        param_names=["beta", "gamma"],
    )
    assert result["ok"] is True, result
    payload = result["result"]
    assert payload["params"]["beta"] == pytest.approx(2.0, abs=1e-8)
    assert payload["params"]["gamma"] == pytest.approx(1.0, abs=1e-8)
    assert payload["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_curve_fit_without_param_names_hints_param_names_on_parse_failure() -> None:
    result = curve_fit("beta*x + gamma", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
    assert result["ok"] is False
    assert "param_names" in result["error"]


@pytest.mark.parametrize(
    "param_names",
    [
        ["a", "a"],  # duplicate
        ["x", "b"],  # collides with the independent variable
        ["1a", "b"],  # not a valid identifier
        ["a b", "b"],  # not a valid identifier
        ["", "b"],  # empty name
    ],
)
def test_curve_fit_rejects_invalid_param_names(param_names: list[str]) -> None:
    result = curve_fit(
        "a*x + b", [0.0, 1.0, 2.0], [1.0, 3.0, 5.0], param_names=param_names
    )
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def _assert_plot_file(result: dict, output_dir: Path) -> Path:
    assert result["ok"] is True, result
    path = Path(result["result"]["path"])
    assert path.is_absolute()
    assert path.parent == output_dir.resolve()
    assert path.exists()
    assert path.stat().st_size > 0
    return path


def test_plot_line_writes_file(sci_output_dir: Path) -> None:
    result = plot("line", {"x": [0, 1, 2], "y": [0, 1, 4]}, filename="test_line.png")
    path = _assert_plot_file(result, sci_output_dir)
    assert path.name == "test_line.png"


def test_plot_hist_writes_file(sci_output_dir: Path) -> None:
    result = plot("hist", {"values": [1, 2, 2, 3, 3, 3]}, filename="test_hist.png")
    _assert_plot_file(result, sci_output_dir)


def test_plot_bar_accepts_values(sci_output_dir: Path) -> None:
    result = plot(
        "bar", {"values": [3, 1, 4]}, title="bars", xlabel="i", ylabel="v",
        filename="test_bar.png",
    )
    _assert_plot_file(result, sci_output_dir)


def test_plot_errorbar_writes_file(sci_output_dir: Path) -> None:
    result = plot(
        "errorbar",
        {"x": [0, 1, 2], "y": [1.0, 2.0, 3.0], "yerr": [0.1, 0.2, 0.1]},
        filename="test_errorbar.png",
    )
    _assert_plot_file(result, sci_output_dir)


def test_plot_contour_writes_file(sci_output_dir: Path) -> None:
    result = plot(
        "contour",
        {"x": [0, 1, 2], "y": [0, 1], "z": [[0, 1, 2], [1, 2, 3]]},
        filename="test_contour.png",
    )
    _assert_plot_file(result, sci_output_dir)


def test_plot_surface_writes_file(sci_output_dir: Path) -> None:
    result = plot(
        "surface",
        {"x": [0, 1, 2], "y": [0, 1, 2], "z": [[0, 0, 0], [0, 1, 2], [0, 2, 4]]},
        filename="test_surface.png",
    )
    _assert_plot_file(result, sci_output_dir)


def test_plot_auto_filename_avoids_overwriting(sci_output_dir: Path) -> None:
    first = plot("line", {"x": [0, 1], "y": [0, 1]})
    second = plot("line", {"x": [0, 1], "y": [1, 0]})

    first_path = _assert_plot_file(first, sci_output_dir)
    second_path = _assert_plot_file(second, sci_output_dir)
    assert first_path != second_path
    assert first_path.name.startswith("plot_line_")


def test_plot_rejects_unknown_kind() -> None:
    result = plot("pie", {"x": [1], "y": [1]})
    assert result["ok"] is False
    assert "error" in result


def test_plot_rejects_path_traversal_filename(sci_output_dir: Path) -> None:
    result = plot("line", {"x": [0, 1], "y": [0, 1]}, filename="../escape.png")
    assert result["ok"] is False
    assert "error" in result


def test_plot_rejects_missing_required_field() -> None:
    result = plot("scatter", {"x": [0, 1]})
    assert result["ok"] is False
    assert "error" in result


def test_plot_rejects_bad_z_shape(sci_output_dir: Path) -> None:
    result = plot(
        "contour",
        {"x": [0, 1, 2], "y": [0, 1], "z": [[0, 1], [1, 2]]},
        filename="bad_shape.png",
    )
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# curve_fit constant ydata: R^2 is undefined (quality-fix phase)
# ---------------------------------------------------------------------------


def test_curve_fit_constant_ydata_has_undefined_r_squared() -> None:
    result = curve_fit("a", [1, 2, 3], [5, 5, 5], p0=[1])
    assert result["ok"] is True, result
    payload = result["result"]

    # 0/0 is undefined, not a perfect fit.
    assert payload["r_squared"] is None
    assert payload["r_squared_note"] == "undefined: ydata is constant"
    assert payload["params"]["a"] == pytest.approx(5.0, abs=1e-8)


def test_curve_fit_linear_data_still_reports_unit_r_squared() -> None:
    xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
    ydata = [2.0 * x + 1.0 for x in xdata]

    result = curve_fit("a*x + b", xdata, ydata)
    assert result["ok"] is True, result
    payload = result["result"]
    assert payload["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert "r_squared_note" not in payload


# ---------------------------------------------------------------------------
# "@file" array inputs
# ---------------------------------------------------------------------------


def test_describe_loads_csv_file_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "data.csv").write_text("1\n2\n3\n4\n", encoding="utf-8")

    result = describe("@data.csv")
    assert result["ok"] is True, result
    payload = result["result"]
    assert payload["n"] == 4
    assert payload["mean"] == pytest.approx(2.5, abs=1e-12)
    assert payload["median"] == pytest.approx(2.5, abs=1e-12)


def test_curve_fit_loads_csv_file_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "x.csv").write_text("0\n1\n2\n3\n4\n", encoding="utf-8")
    (tmp_path / "y.csv").write_text("1\n3\n5\n7\n9\n", encoding="utf-8")

    result = curve_fit("a*x + b", "@x.csv", "@y.csv")
    assert result["ok"] is True, result
    payload = result["result"]
    assert payload["params"]["a"] == pytest.approx(2.0, abs=1e-8)
    assert payload["params"]["b"] == pytest.approx(1.0, abs=1e-8)
    assert payload["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_distribution_fit_loads_csv_file_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "data.csv").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")

    result = distribution("fit", "norm", data="@data.csv")
    assert result["ok"] is True, result
    params = result["result"]["params"]
    assert params["loc"] == pytest.approx(3.0, abs=1e-9)
    assert params["scale"] == pytest.approx(math.sqrt(2.0), abs=1e-9)


def test_describe_missing_file_reference_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))

    result = describe("@does_not_exist.csv")
    assert result["ok"] is False
    assert "error" in result
