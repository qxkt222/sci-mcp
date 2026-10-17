"""Tests for the statistical inference tools (STI-)."""

import json
import math

import numpy as np
import pytest

from sci_mcp.tools.stats_inference import (
    anova,
    chi2_test,
    correlation,
    linear_regression,
    register,
    t_test,
)


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------


def test_t_test_one_sample_zero_statistic():
    result = t_test([1, 2, 3, 4, 5], popmean=3)
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "one-sample"
    assert abs(out["statistic"]) < 1e-12
    assert abs(out["p_value"] - 1.0) < 1e-12
    assert out["df"] == 4


def test_t_test_two_sample_significant_difference():
    result = t_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "two-sample"
    assert out["p_value"] < 0.01
    assert out["statistic"] < 0  # second sample is larger
    assert out["df"] == 8  # pooled Student t-test: n1 + n2 - 2


def test_t_test_paired_returns_result():
    result = t_test([1, 2, 3, 4, 5], [2, 4, 5, 7, 9], paired=True)
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "paired"
    assert out["df"] == 4
    assert 0.0 <= out["p_value"] <= 1.0
    assert math.isfinite(out["statistic"])


def test_t_test_rejects_unknown_alternative():
    result = t_test([1, 2, 3], [4, 5, 6], alternative="both")
    assert result["ok"] is False
    assert "error" in result


def test_t_test_requires_reference():
    result = t_test([1, 2, 3, 4, 5])
    assert result["ok"] is False
    assert "error" in result


def test_t_test_paired_length_mismatch():
    result = t_test([1, 2, 3], [1, 2, 3, 4], paired=True)
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# chi-square
# ---------------------------------------------------------------------------


def test_chi2_independence_2x2():
    # Expected counts: [[12, 18], [28, 42]]. With Yates' continuity correction
    # the statistic is N*(|ad-bc| - N/2)^2 / (r1*r2*c1*c2)
    # = 100*(200-50)^2 / (30*70*40*60) = 0.4464285714285714.
    result = chi2_test([[10, 20], [30, 40]])
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "independence"
    assert out["dof"] == 1
    expected_stat = 0.4464285714285714
    assert abs(out["statistic"] - expected_stat) < 1e-9
    # For one degree of freedom the chi-square survival function is
    # erfc(sqrt(x / 2)); compute it independently with the math module.
    expected_p = math.erfc(math.sqrt(expected_stat / 2.0))
    assert abs(out["p_value"] - expected_p) < 1e-12
    flat_expected = [value for row in out["expected"] for value in row]
    assert flat_expected == pytest.approx([12.0, 18.0, 28.0, 42.0])


def test_chi2_goodness_of_fit_perfect_match():
    result = chi2_test([10, 10, 10], [10, 10, 10])
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "goodness-of-fit"
    assert abs(out["statistic"]) < 1e-12
    assert abs(out["p_value"] - 1.0) < 1e-12
    assert out["dof"] == 2


def test_chi2_goodness_of_fit_without_expected():
    result = chi2_test([10, 20, 30])
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "goodness-of-fit"
    assert out["dof"] == 2


def test_chi2_2d_rejects_expected():
    result = chi2_test([[10, 20], [30, 40]], [50, 50])
    assert result["ok"] is False
    assert "error" in result


def test_chi2_rejects_negative_counts():
    result = chi2_test([10, -1, 5])
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


def test_anova_separated_means():
    result = anova([[1, 2, 3], [10, 11, 12], [20, 21, 22]])
    assert result["ok"] is True
    out = result["result"]
    assert out["p_value"] < 0.01
    assert out["df_between"] == 2
    assert out["df_within"] == 6


def test_anova_identical_groups_p_value_one():
    result = anova([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert result["ok"] is True
    out = result["result"]
    assert abs(out["statistic"]) < 1e-12
    assert abs(out["p_value"] - 1.0) < 1e-12


def test_anova_errors():
    assert anova([[1, 2, 3]])["ok"] is False
    assert anova([[1, 2, 3], [4]])["ok"] is False
    # All observations identical -> zero within-group variance -> undefined F.
    assert anova([[5, 5, 5], [5, 5, 5]])["ok"] is False


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def test_pearson_perfect_linear_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2 * value + 1 for value in x]
    result = correlation(x, y, method="pearson")
    assert result["ok"] is True
    out = result["result"]
    assert out["method"] == "pearson"
    assert abs(out["r"] - 1.0) < 1e-12
    assert out["p_value"] < 0.01


def test_spearman_and_kendall_monotonic_relationship():
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]

    spearman = correlation(x, y, method="spearman")
    assert spearman["ok"] is True
    assert abs(spearman["result"]["r"] - 1.0) < 1e-12
    assert spearman["result"]["p_value"] < 0.05

    kendall = correlation(x, y, method="kendall")
    assert kendall["ok"] is True
    assert abs(kendall["result"]["r"] - 1.0) < 1e-12
    assert kendall["result"]["p_value"] < 0.05


def test_correlation_length_mismatch():
    result = correlation([1, 2, 3], [1, 2])
    assert result["ok"] is False
    assert "error" in result


def test_correlation_rejects_unknown_method():
    result = correlation([1, 2, 3], [4, 5, 6], method="cosine")
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# linear regression
# ---------------------------------------------------------------------------


def test_linear_regression_univariate():
    x = [1, 2, 3, 4, 5]
    y = [2 * value + 1 for value in x]
    result = linear_regression(x, y)
    assert result["ok"] is True
    out = result["result"]
    assert out["n"] == 5
    assert set(out["coefficients"]) == {"intercept", "x0"}
    assert out["coefficients"]["intercept"] == pytest.approx(1.0, abs=1e-9)
    assert out["coefficients"]["x0"] == pytest.approx(2.0, abs=1e-9)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert out["std_errors"]["x0"] == pytest.approx(0.0, abs=1e-12)


def test_linear_regression_multivariate_known_coefficients():
    # y = 3*x0 - 2*x1 + 5, evaluated independently by hand.
    x = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
    y = [4, 9, 6, 11, 10]
    result = linear_regression(x, y)
    assert result["ok"] is True
    out = result["result"]
    assert out["n"] == 5
    assert set(out["coefficients"]) == {"intercept", "x0", "x1"}
    assert out["coefficients"]["intercept"] == pytest.approx(5.0, abs=1e-8)
    assert out["coefficients"]["x0"] == pytest.approx(3.0, abs=1e-8)
    assert out["coefficients"]["x1"] == pytest.approx(-2.0, abs=1e-8)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-8)
    for key in ("std_errors", "t_values", "p_values"):
        assert set(out[key]) == {"intercept", "x0", "x1"}


def test_linear_regression_custom_feature_names():
    x = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
    y = [4, 9, 6, 11, 10]
    result = linear_regression(x, y, feature_names=["height", "weight"])
    assert result["ok"] is True
    out = result["result"]
    assert set(out["coefficients"]) == {"intercept", "height", "weight"}
    assert out["coefficients"]["height"] == pytest.approx(3.0, abs=1e-8)
    assert out["coefficients"]["weight"] == pytest.approx(-2.0, abs=1e-8)


def test_linear_regression_errors():
    # Length mismatch.
    assert linear_regression([1, 2, 3], [1, 2])["ok"] is False
    # Too few observations for the number of parameters (n <= p).
    assert linear_regression([[1, 2], [3, 4]], [1, 2])["ok"] is False
    # Constant response.
    assert linear_regression([1, 2, 3], [5, 5, 5])["ok"] is False
    # Wrong number of feature names.
    assert (
        linear_regression(
            [[1, 2], [2, 3], [3, 4], [4, 5]], [1, 2, 3, 4], feature_names=["only-one"]
        )["ok"]
        is False
    )


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_results_are_json_serializable():
    responses = [
        t_test([1, 2, 3, 4, 5], popmean=3),
        t_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14]),
        chi2_test([[10, 20], [30, 40]]),
        chi2_test([10, 10, 10], [10, 10, 10]),
        anova([[1, 2, 3], [10, 11, 12], [20, 21, 22]]),
        correlation([1, 2, 3, 4, 5], [2.1, 4.2, 5.8, 8.1, 9.9]),
        linear_regression([1, 2, 3, 4, 5], [2.1, 4.2, 5.8, 8.1, 9.9]),
        linear_regression(
            [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]], [4.1, 8.9, 6.2, 10.8, 10.1]
        ),
    ]
    for response in responses:
        assert response["ok"] is True
        encoded = json.dumps(response)
        assert json.loads(encoded)["ok"] is True


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
        "t_test",
        "chi2_test",
        "anova",
        "correlation",
        "linear_regression",
    ]


# ---------------------------------------------------------------------------
# "@<path>" array file references
# ---------------------------------------------------------------------------


def test_t_test_one_sample_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "sample.csv").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")

    result = t_test("@sample.csv", popmean=3)
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "one-sample"
    assert abs(out["statistic"]) < 1e-12
    assert abs(out["p_value"] - 1.0) < 1e-12
    assert out["df"] == 4


def test_t_test_two_sample_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "first.txt").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("10\n11\n12\n13\n14\n", encoding="utf-8")

    result = t_test("@first.txt", "@second.txt")
    assert result["ok"] is True
    out = result["result"]
    assert out["test"] == "two-sample"
    assert out["statistic"] < 0
    assert out["p_value"] < 0.01
    assert out["df"] == 8


def test_correlation_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    x = [1, 2, 3, 4, 5]
    y = [2 * value + 1 for value in x]
    (tmp_path / "x.json").write_text(json.dumps(x), encoding="utf-8")
    (tmp_path / "y.json").write_text(json.dumps(y), encoding="utf-8")

    result = correlation("@x.json", "@y.json", method="pearson")
    assert result["ok"] is True
    out = result["result"]
    assert out["method"] == "pearson"
    assert abs(out["r"] - 1.0) < 1e-12
    assert out["p_value"] < 0.01


def test_linear_regression_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    x = [1, 2, 3, 4, 5]
    y = [2 * value + 1 for value in x]
    (tmp_path / "reg_x.csv").write_text("\n".join(map(str, x)), encoding="utf-8")
    (tmp_path / "reg_y.csv").write_text("\n".join(map(str, y)), encoding="utf-8")

    result = linear_regression("@reg_x.csv", "@reg_y.csv")
    assert result["ok"] is True
    out = result["result"]
    assert out["n"] == 5
    assert out["coefficients"]["intercept"] == pytest.approx(1.0, abs=1e-9)
    assert out["coefficients"]["x0"] == pytest.approx(2.0, abs=1e-9)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-12)


def test_linear_regression_multivariate_from_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    # y = 3*x0 - 2*x1 + 5, evaluated independently.
    rows = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
    y = [4, 9, 6, 11, 10]
    (tmp_path / "design.csv").write_text(
        "\n".join(",".join(str(value) for value in row) for row in rows),
        encoding="utf-8",
    )
    np.save(tmp_path / "target.npy", np.asarray(y, dtype=float))

    result = linear_regression("@design.csv", "@target.npy")
    assert result["ok"] is True
    out = result["result"]
    assert out["coefficients"]["intercept"] == pytest.approx(5.0, abs=1e-8)
    assert out["coefficients"]["x0"] == pytest.approx(3.0, abs=1e-8)
    assert out["coefficients"]["x1"] == pytest.approx(-2.0, abs=1e-8)


def test_array_file_missing_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))

    t_result = t_test("@missing.csv", popmean=0.0)
    assert t_result["ok"] is False
    assert "error" in t_result

    corr_result = correlation("@missing.csv", "@missing.csv")
    assert corr_result["ok"] is False
    assert "error" in corr_result

