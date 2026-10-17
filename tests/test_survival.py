"""Tests for the survival analysis tools (SV-)."""

import json

import numpy as np
import pytest

from sci_mcp.tools.survival import cox_regression, kaplan_meier, register


def _assert_json_safe(payload):
    """Every envelope must survive strict JSON encoding (no NaN/Infinity)."""
    json.dumps(payload, allow_nan=False)


def _cox_data(seed: int = 42, n: int = 200, direction: int = -1):
    """Simulate Cox data; ``direction=-1`` makes larger x fail earlier."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    survival_time = rng.exponential(scale=np.exp(direction * x))
    censor_time = rng.exponential(scale=5.0, size=n)
    duration = np.minimum(survival_time, censor_time)
    event = (survival_time <= censor_time).astype(int)
    return x, duration, event


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------


def test_kaplan_meier_hand_computed():
    result = kaplan_meier([1.0, 2.0, 3.0], [1, 1, 0])
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["time"] == [1.0, 2.0, 3.0]
    assert out["survival"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert out["at_risk"] == [3, 2, 1]
    assert out["events"] == [1, 1, 0]
    assert out["median_survival"] == pytest.approx(2.0)


def test_kaplan_meier_all_censored():
    result = kaplan_meier([1.0, 2.0, 3.0], [0, 0, 0])
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["time"] == [1.0, 2.0, 3.0]
    assert out["survival"] == [1.0, 1.0, 1.0]
    assert out["events"] == [0, 0, 0]
    assert out["median_survival"] is None


def test_kaplan_meier_groups_are_separate_and_named():
    result = kaplan_meier(
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        [1, 1, 0, 0, 1, 0],
        groups=["a", "a", "a", "b", "b", "b"],
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    groups = result["result"]["groups"]
    assert list(groups) == ["a", "b"]

    assert groups["a"]["survival"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert groups["a"]["median_survival"] == pytest.approx(2.0)

    # Group b: censored at 1, event at 2, censored at 3.
    assert groups["b"]["survival"] == pytest.approx([1.0, 0.5, 0.5])
    assert groups["b"]["at_risk"] == [3, 2, 1]
    assert groups["b"]["events"] == [0, 1, 0]
    assert groups["b"]["median_survival"] == pytest.approx(2.0)


def test_kaplan_meier_rejects_empty_and_mismatched_and_negative():
    empty = kaplan_meier([], [])
    assert empty["ok"] is False
    assert "error" in empty

    mismatched = kaplan_meier([1.0, 2.0], [1])
    assert mismatched["ok"] is False
    assert "error" in mismatched

    negative = kaplan_meier([-1.0, 2.0], [1, 0])
    assert negative["ok"] is False
    assert "error" in negative


def test_kaplan_meier_rejects_non_binary_events():
    result = kaplan_meier([1.0, 2.0], [1, 2])
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------


def test_cox_positive_coefficient_for_higher_risk():
    x, duration, event = _cox_data(seed=42, direction=-1)

    result = cox_regression(
        "x", {"time": duration.tolist(), "event": event.tolist(), "x": x.tolist()},
        "time", "event",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["n"] == 200
    assert out["coefficients"]["x"] > 0.0
    assert out["p_values"]["x"] < 0.05
    assert out["hazard_ratios"]["x"] > 1.0
    assert out["concordance"] is not None
    assert out["concordance"] > 0.5
    assert out["aic"] is not None


def test_cox_negative_coefficient_for_lower_risk():
    x, duration, event = _cox_data(seed=42, direction=1)

    result = cox_regression(
        "x", {"time": duration.tolist(), "event": event.tolist(), "x": x.tolist()},
        "time", "event",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["coefficients"]["x"] < 0.0
    assert out["hazard_ratios"]["x"] < 1.0
    assert out["p_values"]["x"] < 0.05


def test_cox_rejects_non_binary_event_column():
    x, duration, _ = _cox_data()
    result = cox_regression(
        "x",
        {"time": duration.tolist(), "event": [2] * duration.size, "x": x.tolist()},
        "time", "event",
    )
    assert result["ok"] is False
    assert "error" in result


def test_cox_rejects_zero_and_negative_durations():
    x, duration, event = _cox_data()
    for bad_value in (0.0, -3.0):
        broken = duration.copy()
        broken[0] = bad_value
        result = cox_regression(
            "x",
            {"time": broken.tolist(), "event": event.tolist(), "x": x.tolist()},
            "time", "event",
        )
        assert result["ok"] is False, f"duration {bad_value} should be rejected"
        assert "error" in result


def test_cox_two_covariates_have_named_coefficients():
    rng = np.random.default_rng(7)
    n = 150
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    survival_time = rng.exponential(scale=np.exp(-0.8 * x1 + 0.6 * x2))
    censor_time = rng.exponential(scale=5.0, size=n)
    duration = np.minimum(survival_time, censor_time)
    event = (survival_time <= censor_time).astype(int)

    result = cox_regression(
        "x1 + x2",
        {
            "time": duration.tolist(),
            "event": event.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist(),
        },
        "time", "event",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert set(out["coefficients"]) == {"x1", "x2"}
    assert out["coefficients"]["x1"] > 0.0
    assert out["coefficients"]["x2"] < 0.0


# ---------------------------------------------------------------------------
# File input
# ---------------------------------------------------------------------------


def test_cox_from_csv_file(tmp_path):
    x, duration, event = _cox_data(seed=11, direction=-1)
    path = tmp_path / "cox.csv"
    lines = ["time,event,x"]
    lines += [
        f"{time_value},{event_value},{x_value}"
        for time_value, event_value, x_value in zip(duration, event, x)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = cox_regression("x", f"@{path}", "time", "event")
    _assert_json_safe(result)

    assert result["ok"] is True
    out = result["result"]
    assert out["n"] == 200
    assert out["coefficients"]["x"] > 0.0


def test_kaplan_meier_from_csv_files(tmp_path):
    durations_path = tmp_path / "durations.csv"
    events_path = tmp_path / "events.csv"
    groups_path = tmp_path / "groups.csv"

    durations_path.write_text("1\n2\n3\n1\n2\n3\n", encoding="utf-8")
    events_path.write_text("1\n1\n0\n0\n1\n0\n", encoding="utf-8")
    groups_path.write_text("group\na\na\na\nb\nb\nb\n", encoding="utf-8")

    result = kaplan_meier(
        f"@{durations_path}", f"@{events_path}", groups=f"@{groups_path}"
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    groups = result["result"]["groups"]
    assert list(groups) == ["a", "b"]
    assert groups["a"]["survival"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert groups["b"]["survival"] == pytest.approx([1.0, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Envelope, fields, registration
# ---------------------------------------------------------------------------


def test_outputs_are_strict_json_and_have_required_fields():
    km = kaplan_meier([1.0, 2.0, 3.0], [1, 1, 0])
    cox = cox_regression(*_cox_frame())
    _assert_json_safe(km)
    _assert_json_safe(cox)

    assert set(km["result"]) >= {
        "time", "survival", "at_risk", "events", "median_survival"
    }
    assert set(cox["result"]) >= {
        "formula",
        "n",
        "coefficients",
        "hazard_ratios",
        "std_errors",
        "z_values",
        "p_values",
        "concordance",
        "aic",
    }


def _cox_frame():
    x, duration, event = _cox_data(seed=3)
    data = {
        "time": duration.tolist(),
        "event": event.tolist(),
        "x": x.tolist(),
    }
    return "x", data, "time", "event"


def test_register_registers_both_tools():
    class FakeMCP:
        def __init__(self):
            self.tools = []

        def tool(self):
            def decorator(function):
                self.tools.append(function)
                return function

            return decorator

    fake = FakeMCP()
    register(fake)
    names = {function.__name__ for function in fake.tools}
    assert {"cox_regression", "kaplan_meier"} <= names


# ---------------------------------------------------------------------------
# Convergence / complete separation (SVF-)
# ---------------------------------------------------------------------------

_SEPARATED_TIME = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_SEPARATED_EVENT = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
_SEPARATED_X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _separation_result(x_values):
    return cox_regression(
        "x",
        {
            "time": list(_SEPARATED_TIME),
            "event": list(_SEPARATED_EVENT),
            "x": list(x_values),
        },
        "time",
        "event",
    )


def test_cox_complete_separation_is_rejected():
    result = _separation_result(_SEPARATED_X)
    _assert_json_safe(result)

    assert result["ok"] is False
    message = result["error"].lower()
    assert "converge" in message or "separation" in message


def test_cox_complete_separation_reverse_direction_is_rejected():
    result = _separation_result(list(reversed(_SEPARATED_X)))
    _assert_json_safe(result)

    assert result["ok"] is False
    message = result["error"].lower()
    assert "converge" in message or "separation" in message


def test_cox_failure_payload_has_convergence_note_and_fields():
    result = _separation_result(_SEPARATED_X)

    assert result["ok"] is False
    assert isinstance(result.get("convergence_note"), str)
    assert result["convergence_note"].strip()
    assert set(result) >= {
        "coefficients",
        "hazard_ratios",
        "std_errors",
        "p_values",
        "concordance",
        "aic",
        "convergence_note",
    }


def test_cox_overlapping_data_still_converges():
    x, duration, event = _cox_data(seed=42, direction=-1)
    result = cox_regression(
        "x",
        {"time": duration.tolist(), "event": event.tolist(), "x": x.tolist()},
        "time",
        "event",
    )
    _assert_json_safe(result)

    assert result["ok"] is True
    assert "convergence_note" not in result
    assert result["result"]["coefficients"]["x"] > 0.0
