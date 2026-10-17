"""Tests for the nonlinear mixed-model approximation (NM-)."""

import json

import numpy as np
import pandas as pd
import pytest

from sci_mcp.tools.nonlinear_mixed import nonlinear_mixed_lm


def _assert_json_safe(payload):
    """Every envelope must survive strict JSON encoding (no NaN/Infinity)."""
    json.dumps(payload, allow_nan=False)


def _simulated(seed=2026, noise=0.02):
    """5 groups x 8 observations of y = 2*exp(0.5*x) + group offset + noise."""
    rng = np.random.default_rng(seed)
    offsets = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    groups, xs, ys = [], [], []
    for index, offset in enumerate(offsets):
        x = np.linspace(0.0, 1.0, 8)
        y = 2.0 * np.exp(0.5 * x) + offset + rng.normal(0.0, noise, x.size)
        groups.extend([f"g{index}"] * x.size)
        xs.extend(x.tolist())
        ys.extend(y.tolist())
    data = {"group": groups, "x": xs, "y": ys}
    return data, offsets


def _fit(data, **kwargs):
    return nonlinear_mixed_lm(
        model="a*exp(b*x)",
        data=data,
        groups="group",
        x="x",
        y="y",
        p0={"a": 1.0, "b": 0.0},
        **kwargs,
    )


def test_recovers_fixed_parameters():
    data, _ = _simulated()
    result = _fit(data)
    _assert_json_safe(result)

    assert result["ok"] is True
    fixed = result["result"]["fixed_params"]
    assert abs(fixed["a"] - 2.0) < 0.3
    assert abs(fixed["b"] - 0.5) < 0.1


def test_random_effects_track_true_offsets():
    data, offsets = _simulated()
    result = _fit(data)
    assert result["ok"] is True
    random_effects = result["result"]["random_effects"]
    assert set(random_effects) == {f"g{i}" for i in range(5)}

    estimated = np.array(
        [random_effects[f"g{i}"] for i in range(5)], dtype=float
    )
    correlation = float(np.corrcoef(estimated, offsets)[0, 1])
    max_error = float(np.max(np.abs(estimated - offsets)))
    assert correlation > 0.8 or max_error < 0.3


def test_no_random_effect_drops_field_and_recovers_fixed_parameters():
    data, _ = _simulated(noise=0.01)
    result = _fit(data, random_effect=False)
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert "random_effects" not in out
    assert abs(out["fixed_params"]["a"] - 2.0) < 0.3
    assert abs(out["fixed_params"]["b"] - 0.5) < 0.1
    assert out["n_groups"] == 5


def test_reads_at_file_csv(tmp_path):
    data, _ = _simulated()
    path = tmp_path / "nm_data.csv"
    pd.DataFrame(data).to_csv(path, index=False)

    result = _fit(f"@{path}")
    _assert_json_safe(result)

    assert result["ok"] is True
    assert result["result"]["n"] == 40
    assert abs(result["result"]["fixed_params"]["a"] - 2.0) < 0.3


def test_unknown_column_returns_error():
    data, _ = _simulated()
    result = nonlinear_mixed_lm(
        model="a*exp(b*x)",
        data=data,
        groups="group",
        x="missing_x",
        y="y",
        p0={"a": 1.0, "b": 0.0},
    )
    _assert_json_safe(result)

    assert result["ok"] is False
    assert "error" in result
    assert "missing_x" in result["error"]


def test_invalid_model_returns_error():
    data, _ = _simulated()
    result = nonlinear_mixed_lm(
        model="a*exp(b*x",
        data=data,
        groups="group",
        x="x",
        y="y",
        p0={"a": 1.0, "b": 0.0},
    )
    assert result["ok"] is False
    assert "error" in result


def test_unknown_model_symbol_returns_error():
    data, _ = _simulated()
    result = nonlinear_mixed_lm(
        model="a*exp(b*x) + c",
        data=data,
        groups="group",
        x="x",
        y="y",
        p0={"a": 1.0, "b": 0.0},
    )
    assert result["ok"] is False
    assert "error" in result


def test_single_group_returns_error():
    data, _ = _simulated()
    data = dict(data)
    data["group"] = ["only"] * len(data["group"])
    result = _fit(data)
    assert result["ok"] is False
    assert "2 groups" in result["error"]


def test_group_with_fewer_than_two_observations_returns_error():
    result = _fit(
        {
            "group": ["a", "b", "b", "b"],
            "x": [0.0, 0.0, 0.5, 1.0],
            "y": [1.0, 1.0, 1.5, 2.0],
        }
    )
    assert result["ok"] is False
    assert "at least 2 observations" in result["error"]


def test_empty_p0_returns_error():
    data, _ = _simulated()
    result = nonlinear_mixed_lm(
        model="a*exp(b*x)",
        data=data,
        groups="group",
        x="x",
        y="y",
        p0={},
    )
    assert result["ok"] is False
    assert "p0" in result["error"]


def test_larger_penalty_shrinks_random_effects():
    data, _ = _simulated()
    weak = _fit(data, penalty=1.0)
    strong = _fit(data, penalty=100.0)

    assert weak["ok"] is True
    assert strong["ok"] is True

    weak_magnitude = sum(value**2 for value in weak["result"]["random_effects"].values())
    strong_magnitude = sum(
        value**2 for value in strong["result"]["random_effects"].values()
    )
    assert strong_magnitude < weak_magnitude


def test_approximation_is_explicitly_flagged():
    data, _ = _simulated()
    result = _fit(data)
    assert result["ok"] is True
    out = result["result"]
    assert out["approximation"] is True
    assert isinstance(out["note"], str) and out["note"]
    assert "not a full NLME" in out["note"]


def test_output_is_strict_json_serializable():
    data, _ = _simulated()
    result = _fit(data)
    assert result["ok"] is True
    json.dumps(result, allow_nan=False)


def test_deterministic_for_fixed_data():
    data, _ = _simulated()
    first = _fit(data)
    second = _fit(data)
    assert first["ok"] is True and second["ok"] is True
    assert first["result"]["fixed_params"] == pytest.approx(
        second["result"]["fixed_params"], rel=0.0, abs=0.0
    )
    assert first["result"]["random_effects"] == pytest.approx(
        second["result"]["random_effects"], rel=0.0, abs=0.0
    )
