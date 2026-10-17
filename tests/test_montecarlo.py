"""Tests for sci_mcp.tools.montecarlo against independently known distribution facts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sci_mcp.tools.montecarlo import monte_carlo


def test_standard_normal_mean_and_std() -> None:
    """x ~ N(0, 1) has theoretical mean 0 and standard deviation 1."""
    result = monte_carlo("x", {"x": {"dist": "norm", "loc": 0, "scale": 1}}, n=50000, seed=42)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["n"] == 50000
    assert abs(payload["mean"]) < 0.03
    assert 0.95 < payload["std"] < 1.05


def test_sum_of_two_standard_normals_has_std_sqrt2() -> None:
    """The sum of two independent N(0, 1) variables has standard deviation sqrt(2)."""
    result = monte_carlo(
        "x + y",
        {
            "x": {"dist": "norm", "loc": 0, "scale": 1},
            "y": {"dist": "norm", "loc": 0, "scale": 1},
        },
        n=50000,
        seed=7,
    )

    assert result["ok"] is True
    assert abs(result["result"]["mean"]) < 0.05
    assert 1.35 < result["result"]["std"] < 1.48


def test_uniform_low_high_mapping() -> None:
    """uniform(low=0, high=1) samples lie in [0, 1] with mean 0.5."""
    result = monte_carlo(
        "x",
        {"x": {"dist": "uniform", "low": 0, "high": 1}},
        n=20000,
        seed=11,
    )

    assert result["ok"] is True
    payload = result["result"]
    assert payload["min"] >= 0.0
    assert payload["max"] <= 1.0
    assert abs(payload["mean"] - 0.5) < 0.03
    assert len(payload["preview"]) <= 20


def test_same_seed_reproduces_identical_output() -> None:
    """Two calls with the same seed return bit-for-bit identical results."""
    variables = {"x": {"dist": "norm", "loc": 2.0, "scale": 0.5}}
    first = monte_carlo("x**2", variables, n=5000, seed=123, percentiles=[10, 50, 90])
    second = monte_carlo("x**2", variables, n=5000, seed=123, percentiles=[10, 50, 90])

    assert first["ok"] is True and second["ok"] is True
    assert first["result"] == second["result"]


def test_percentiles_are_monotone_and_median_matches_theory() -> None:
    """Percentiles of N(0, 1) are non-decreasing and the median is near 0."""
    result = monte_carlo(
        "x",
        {"x": {"dist": "norm", "loc": 0, "scale": 1}},
        n=50000,
        seed=3,
        percentiles=[1, 5, 50, 95, 99],
    )

    assert result["ok"] is True
    percentiles = result["result"]["percentiles"]
    assert set(percentiles) == {"1", "5", "50", "95", "99"}
    values = [percentiles[key] for key in ["1", "5", "50", "95", "99"]]
    assert values == sorted(values)
    assert abs(percentiles["50"]) < 0.03
    assert percentiles["5"] < percentiles["50"] < percentiles["95"]


def test_save_histogram_writes_png(sci_output_dir: Path) -> None:
    """save_histogram=True writes a PNG under the configured output directory."""
    result = monte_carlo(
        "x",
        {"x": {"dist": "uniform", "low": -1, "high": 1}},
        n=2000,
        seed=5,
        save_histogram=True,
        filename="mc_hist.png",
    )

    assert result["ok"] is True
    path = Path(result["result"]["path"])
    assert path.is_absolute()
    assert path.name == "mc_hist.png"
    assert path.parent == sci_output_dir
    assert path.exists()
    assert path.stat().st_size > 0


def test_unknown_distribution_is_rejected() -> None:
    """An unknown scipy.stats distribution name returns an error envelope."""
    result = monte_carlo("x", {"x": {"dist": "definitely_not_a_distribution"}}, n=100)

    assert result["ok"] is False
    assert "error" in result
    assert "monte_carlo" in result["error"]


def test_invalid_expression_is_rejected() -> None:
    """A malformed sympy expression returns an error envelope without raising."""
    result = monte_carlo("x +* y", {"x": {"dist": "norm"}, "y": {"dist": "norm"}}, n=100)

    assert result["ok"] is False
    assert "error" in result


def test_invalid_n_is_rejected() -> None:
    """n outside 1..10,000,000 returns an error envelope."""
    result = monte_carlo("x", {"x": {"dist": "norm"}}, n=0)

    assert result["ok"] is False
    assert "error" in result


def test_output_is_json_serializable() -> None:
    """The full result envelope can be passed straight to json.dumps."""
    result = monte_carlo(
        "x + 2*y",
        {
            "x": {"dist": "norm", "loc": 0, "scale": 1},
            "y": {"dist": "uniform", "low": 0, "high": 10},
        },
        n=1000,
        seed=9,
        percentiles=[25, 50, 75],
    )

    assert result["ok"] is True
    encoded = json.dumps(result)
    assert isinstance(encoded, str)
    decoded = json.loads(encoded)
    assert decoded["result"]["n"] == 1000
    assert math.isclose(decoded["result"]["mean"], result["result"]["mean"])
    assert np.isfinite(decoded["result"]["std"])
