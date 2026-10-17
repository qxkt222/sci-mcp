"""Integration tests: MCP registration, dispatch and JSON serialization.

These tests go through ``mcp.call_tool`` (not the underlying functions) so that a
tool module that fails to import or register cannot pass silently: the exact-set
check below fails when ``server.py`` skips a broken module.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

import numpy as np

from sci_mcp.server import mcp

EXPECTED_TOOLS = {
    "analyze_dataset",
    "anova",
    "arima",
    "chi2_test",
    "convolve",
    "correlation",
    "cox_regression",
    "curl",
    "curve_fit",
    "describe",
    "describe_file",
    "distribution",
    "divergence",
    "dsolve_ode",
    "fft_analysis",
    "filter_signal",
    "find_peaks",
    "find_roots",
    "forecast",
    "gee",
    "glm",
    "global_optimize",
    "gradient",
    "health_check",
    "integer_program",
    "interpolate_1d",
    "kaplan_meier",
    "laplacian",
    "linear_program",
    "linear_regression",
    "load_data",
    "matrix_compute",
    "mixed_lm",
    "monte_carlo",
    "nonlinear_mixed_lm",
    "optimize_minimize",
    "physical_constant",
    "plot",
    "plot_file",
    "power_spectrum",
    "run_python",
    "save_data",
    "solve_burgers_1d",
    "solve_heat_1d",
    "solve_heat_1d_variable",
    "solve_heat_3d",
    "solve_laplace_2d",
    "solve_laplace_3d",
    "solve_navier_stokes_2d",
    "solve_ode",
    "solve_pde_1d",
    "solve_wave_1d",
    "solve_wave_3d",
    "sparse_compute",
    "stationarity_test",
    "stft",
    "symbolic_calculus",
    "symbolic_evaluate",
    "symbolic_solve",
    "t_test",
    "time_series_decompose",
    "unit_calculate",
    "unit_convert",
    "wavelet_denoise",
    "wavelet_transform",
}


def _content_text(result) -> str:
    """Join every text block of a CallToolResult (used for error diagnostics)."""
    chunks = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _call_ok(name: str, args: dict) -> dict:
    """Call a tool and return its unwrapped result, asserting a success envelope."""
    result = asyncio.run(mcp.call_tool(name, args))
    payload = _decode_payload(result)
    assert payload.get("ok") is True, f"{name} returned an error payload: {payload}"
    return payload["result"]


def _decode_payload(result) -> dict:
    """Decode a CallToolResult into the tool's ``ok``/``err`` envelope.

    Measured structure for mcp 2.x: ``result.is_error`` is a bool,
    ``result.structured_content`` is None for dict-returning tools, and the
    JSON-serialized envelope arrives as the single TextContent's ``text``.
    """
    assert getattr(result, "is_error", False) is False, (
        f"call_tool reported an error: {_content_text(result)}"
    )

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "ok" in structured:
        return structured

    texts = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)
    assert len(texts) == 1, f"expected exactly one text block, got {len(texts)}"
    payload = json.loads(texts[0])
    assert isinstance(payload, dict), f"tool envelope is not a JSON object: {payload!r}"
    return payload


# --- per-tool result checks -------------------------------------------------


def _check_symbolic_calculus(value: dict, output_dir: Path) -> None:
    assert value["operation"] == "diff"
    assert value["expression"] == "2*x"


def _check_symbolic_solve(value: dict, output_dir: Path) -> None:
    roots = sorted(int(item) for item in value["solutions"])
    assert roots == [-1, 1]


def _check_symbolic_evaluate(value: dict, output_dir: Path) -> None:
    assert value["numeric"] == pytest.approx(4.0)


def _check_physical_constant(value: dict, output_dir: Path) -> None:
    assert value["value"] == pytest.approx(299792458.0)
    assert value["unit"].replace(" ", "") == "m/s"


def _check_unit_convert(value: dict, output_dir: Path) -> None:
    assert value["value"] == pytest.approx(2.54)


def _check_unit_calculate(value: dict, output_dir: Path) -> None:
    assert value["value"] == pytest.approx(1.5)
    assert "meter" in value["unit"] and "second" in value["unit"]


def _check_solve_ode(value: dict, output_dir: Path) -> None:
    assert len(value["t"]) == len(value["y"])
    assert value["t"][0] == pytest.approx(0.0)
    assert value["t"][-1] == pytest.approx(1.0)
    assert value["y"][-1] == pytest.approx(math.exp(-1.0), rel=1e-6)


def _check_optimize_minimize(value: dict, output_dir: Path) -> None:
    assert value["success"] is True
    assert value["x"][0] == pytest.approx(2.0, abs=1e-4)


def _check_matrix_compute(value: float, output_dir: Path) -> None:
    assert value == pytest.approx(-2.0)


def _check_find_roots(value: dict, output_dir: Path) -> None:
    assert value["roots"][0] == pytest.approx(2.0)


def _check_fft_analysis(value: dict, output_dir: Path) -> None:
    assert value["dominant_frequency"] == pytest.approx(0.25)
    assert value["amplitudes"][1] == pytest.approx(1.0)


def _check_health_check(value: dict, output_dir: Path) -> None:
    assert value["healthy"] is True
    assert value["failed_modules"] == []
    assert value["module_count"] >= 14
    assert value["tool_function_count"] == 65


def _check_interpolate_1d(value: dict, output_dir: Path) -> None:
    assert value["yi"] == pytest.approx([1.0])


def _check_describe(value: dict, output_dir: Path) -> None:
    assert value["n"] == 3
    assert value["mean"] == pytest.approx(2.0)
    assert value["median"] == pytest.approx(2.0)


def _check_distribution(value: dict, output_dir: Path) -> None:
    assert value["values"] == pytest.approx(0.0)


def _check_curve_fit(value: dict, output_dir: Path) -> None:
    assert value["params"]["a"] == pytest.approx(2.0)
    assert value["params"]["b"] == pytest.approx(1.0)


def _check_plot(value: dict, output_dir: Path) -> None:
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()
    assert path.name == "integration_smoke.png"


def _check_run_python(value: dict, output_dir: Path) -> None:
    assert value["returncode"] == 0
    assert "2" in value["stdout"]


def _check_t_test(value: dict, output_dir: Path) -> None:
    assert value["test"] == "one-sample"
    assert value["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert value["p_value"] == pytest.approx(1.0)
    assert value["df"] == 4


def _check_chi2_test(value: dict, output_dir: Path) -> None:
    assert value["test"] == "independence"
    assert value["statistic"] == pytest.approx(0.4464285714285714, rel=1e-12)
    assert value["dof"] == 1
    assert value["p_value"] == pytest.approx(0.5040358664525046, rel=1e-9)


def _check_anova(value: dict, output_dir: Path) -> None:
    assert value["statistic"] == pytest.approx(13.5)
    assert value["df_between"] == 1
    assert value["df_within"] == 4
    assert value["p_value"] < 0.05


def _check_correlation(value: dict, output_dir: Path) -> None:
    assert value["method"] == "pearson"
    assert value["r"] == pytest.approx(1.0)


def _check_linear_regression(value: dict, output_dir: Path) -> None:
    assert value["coefficients"]["intercept"] == pytest.approx(1.0, abs=1e-12)
    assert value["coefficients"]["x0"] == pytest.approx(2.0, abs=1e-12)
    assert value["r_squared"] == pytest.approx(1.0)
    assert value["n"] == 3


def _check_dsolve_ode(value: dict, output_dir: Path) -> None:
    assert value["order"] == 1
    assert value["particular"] is False
    assert len(value["solutions"]) == 1
    assert "exp(t)" in value["solutions"][0]


def _check_gradient(value: dict, output_dir: Path) -> None:
    assert value["components"] == ["2*x", "2*y"]
    assert value["variables"] == ["x", "y"]


def _check_divergence(value: dict, output_dir: Path) -> None:
    assert value["divergence"] == "3"


def _check_curl(value: dict, output_dir: Path) -> None:
    assert value["components"] == ["0", "0", "2"]


def _check_laplacian(value: dict, output_dir: Path) -> None:
    assert value["laplacian"] == "4"


def _check_solve_heat_1d(value: dict, output_dir: Path) -> None:
    # ``shape`` is the returned (downsampled) array shape; ``original_shape``
    # keeps the full grid dimensions.
    assert value["shape"] == [50, 21]
    assert value["original_shape"] == [101, 21]
    assert len(value["x"]) == 21
    assert len(value["t"]) == 50  # downsampled from 101 time points
    assert len(value["u"]) == len(value["t"])
    assert len(value["u"][0]) == len(value["x"])
    assert value["downsampled"] is True
    assert value["max"] == pytest.approx(1.0)
    assert value["u"][0][10] == pytest.approx(1.0)
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_wave_1d(value: dict, output_dir: Path) -> None:
    assert value["shape"] == [21, 21]
    assert len(value["x"]) == 21
    assert len(value["t"]) == 21
    assert value["max"] == pytest.approx(1.0, abs=1e-9)
    # The standing wave stays non-negative over this short time window.
    assert value["min"] == pytest.approx(0.0, abs=1e-9)
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_laplace_2d(value: dict, output_dir: Path) -> None:
    assert value["shape"] == [11, 11]
    assert value["converged"] is True
    assert value["iterations"] < value["max_iter"]
    center = value["u"][5][5]
    assert 20.0 <= center <= 30.0
    assert value["max"] == pytest.approx(100.0)
    assert value["min"] == pytest.approx(0.0)
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_filter_signal(value: dict, output_dir: Path) -> None:
    assert value["kind"] == "lowpass"
    assert value["cutoff"] == pytest.approx(1.0)
    assert value["sample_rate"] == pytest.approx(10.0)
    assert len(value["filtered"]) == 40
    assert all(math.isfinite(item) for item in value["filtered"])


def _check_convolve(value: dict, output_dir: Path) -> None:
    assert value["mode"] == "full"
    assert value["result"] == pytest.approx([1.0, 3.0, 5.0, 3.0])


def _check_power_spectrum(value: dict, output_dir: Path) -> None:
    assert value["nperseg"] == 32
    assert len(value["frequencies"]) == 17
    assert len(value["psd"]) == len(value["frequencies"])
    assert value["peak_frequency"] == pytest.approx(2.5)


def _check_find_peaks(value: dict, output_dir: Path) -> None:
    assert value["indices"] == [1, 3, 5]
    assert value["values"] == pytest.approx([1.0, 2.0, 1.5])
    assert value["count"] == 3


def _check_monte_carlo(value: dict, output_dir: Path) -> None:
    assert value["n"] == 1000
    assert abs(value["mean"]) < 0.3
    assert value["std"] == pytest.approx(math.sqrt(2.0), abs=0.2)
    assert len(value["preview"]) > 0


def _check_linear_program(value: dict, output_dir: Path) -> None:
    assert value["success"] is True
    assert value["x"] == pytest.approx([2.0, 2.0])
    assert value["fun"] == pytest.approx(-10.0)


def _check_integer_program(value: dict, output_dir: Path) -> None:
    assert value["success"] is True
    assert value["x"] == pytest.approx([1.0, 1.0])
    assert value["fun"] == pytest.approx(-9.0)


def _check_save_data(value: dict, output_dir: Path) -> None:
    assert value["format"] == "csv"
    assert value["shape"] == [3]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()
    assert path.name == "smoke.csv"


def _check_load_data(value: dict, output_dir: Path) -> None:
    assert value["format"] == "csv"
    assert value["shape"] == [3]
    assert value["preview"] == pytest.approx([1.0, 2.0, 3.0])
    assert value["data"] == pytest.approx([1.0, 2.0, 3.0])


def _check_describe_file(value: dict, output_dir: Path) -> None:
    assert value["n"] == 3
    assert value["mean"] == pytest.approx(2.0)
    assert value["median"] == pytest.approx(2.0)


def _check_plot_file(value: dict, output_dir: Path) -> None:
    assert value["kind"] == "line"
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()
    assert path.name == "smoke_plot.png"


def _check_global_optimize(value: dict, output_dir: Path) -> None:
    assert value["success"] is True
    assert value["fun"] < 0.1


def _check_glm(value: dict, output_dir: Path) -> None:
    assert value["family"] == "gaussian"
    assert value["coefficients"]["x"] == pytest.approx(2.0, abs=0.05)


def _check_arima(value: dict, output_dir: Path) -> None:
    assert value["order"] == [1, 0, 0]
    assert len(value["forecast"]) == 3


def _check_time_series_decompose(value: dict, output_dir: Path) -> None:
    assert value["period"] == 2
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_stationarity_test(value: dict, output_dir: Path) -> None:
    assert 0.0 <= value["p_value"] <= 1.0


def _check_stft(value: dict, output_dir: Path) -> None:
    assert value["peak_frequency"] == pytest.approx(1.0, abs=0.2)


def _check_wavelet_transform(value: dict, output_dir: Path) -> None:
    assert value["coefficients"]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_wavelet_denoise(value: dict, output_dir: Path) -> None:
    assert len(value["denoised"]) == 64


def _check_analyze_dataset(value: dict, output_dir: Path) -> None:
    assert value["n_rows"] == 5
    assert value["target_analysis"]


def _check_forecast(value: dict, output_dir: Path) -> None:
    assert value["method"] == "arima"
    assert len(value["forecast"]) == 3


# --- mixed models / sparse / advanced PDE checks ----------------------------

_MIXED_MODEL_DATA = {
    "x": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    "g": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
    "y": [2, 4, 6, 8, 10, 3, 5, 7, 9, 11],
}


def _check_mixed_lm(value: dict, output_dir: Path) -> None:
    assert value["fixed_effects"]["x"] == pytest.approx(2.0, abs=0.3)
    assert value["n_groups"] == 2


def _check_gee(value: dict, output_dir: Path) -> None:
    assert value["family"] == "gaussian"
    assert "x" in value["coefficients"]


def _check_sparse_compute(value: dict, output_dir: Path) -> None:
    # Rebuild the same 3x3 matrix and solve independently with numpy.
    rows = [0, 1, 1, 2]
    cols = [0, 0, 1, 2]
    values = [4, -1, 4, -1]
    matrix = np.zeros((3, 3), dtype=float)
    for row, col, val in zip(rows, cols, values):
        matrix[row, col] = val
    expected = np.linalg.solve(matrix, np.array([15.0, 6.0, 1.0]))
    assert np.allclose(value["x"], expected, atol=1e-10)


def _check_solve_heat_3d(value: dict, output_dir: Path) -> None:
    assert list(np.asarray(value["u"]).shape) == value["shape"]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_laplace_3d(value: dict, output_dir: Path) -> None:
    assert value["converged"] is True
    assert 10.0 <= value["center"] <= 40.0
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_burgers_1d(value: dict, output_dir: Path) -> None:
    assert list(np.asarray(value["u"]).shape) == value["shape"]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


# --- variable-coefficient PDE / 3D wave / Navier-Stokes checks --------------

_NONLINEAR_MIXED_X = [0.0, 0.5, 1.0, 1.5, 2.0]
_NONLINEAR_MIXED_DATA = {
    "x": _NONLINEAR_MIXED_X * 3,
    "y": [
        2.0 * math.exp(0.5 * value) + offset
        for offset in (-0.3, 0.0, 0.3)
        for value in _NONLINEAR_MIXED_X
    ],
    "g": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
}

_SPARSE_CG_MATRIX = {
    "rows": [0, 0, 1, 1, 1, 2, 2],
    "cols": [0, 1, 0, 1, 2, 1, 2],
    "values": [4, -1, -1, 4, -1, -1, 4],
    "shape": [3, 3],
}


def _check_solve_heat_1d_variable(value: dict, output_dir: Path) -> None:
    assert len(value["u"]) == value["shape"][0]
    assert len(value["u"][0]) == value["shape"][1]
    assert len(value["u"]) == len(value["t"])
    assert len(value["u"][0]) == len(value["x"])
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_pde_1d(value: dict, output_dir: Path) -> None:
    assert value["stability_note"]
    assert len(value["u"]) == value["shape"][0]
    assert len(value["u"][0]) == value["shape"][1]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_wave_3d(value: dict, output_dir: Path) -> None:
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_solve_navier_stokes_2d(value: dict, output_dir: Path) -> None:
    assert value["model"]
    assert value["limitations"]
    path = Path(value["path"])
    assert path.is_file()
    assert path.parent == output_dir.resolve()


def _check_nonlinear_mixed_lm(value: dict, output_dir: Path) -> None:
    assert value["approximation"] is True
    parameters = value["fixed_params"]
    assert 1.5 <= parameters["a"] <= 2.5
    assert 0.3 <= parameters["b"] <= 0.7


def _check_cox_regression(value: dict, output_dir: Path) -> None:
    assert "x" in value["coefficients"]
    assert math.isfinite(value["coefficients"]["x"])
    assert "convergence_note" not in value


def _check_kaplan_meier(value: dict, output_dir: Path) -> None:
    assert value["survival"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert value["median_survival"] == 2


def _check_sparse_compute_cg(value: dict, output_dir: Path) -> None:
    assert value["solver"] == "cg"
    assert value["residual"] is not None
    assert value["residual"] < 1e-6
    matrix = np.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ]
    )
    expected = np.linalg.solve(matrix, np.array([15.0, 6.0, 1.0]))
    assert np.allclose(value["x"], expected, atol=1e-8)


SMOKE_CASES = [
    ("symbolic_calculus", {"expr": "x**2", "operation": "diff"}, _check_symbolic_calculus),
    ("symbolic_solve", {"equations": "x**2 - 1", "variables": "x"}, _check_symbolic_solve),
    (
        "symbolic_evaluate",
        {"expr": "x**2", "substitutions": {"x": 2}},
        _check_symbolic_evaluate,
    ),
    ("physical_constant", {"query": "c"}, _check_physical_constant),
    (
        "unit_convert",
        {"value": 1, "from_unit": "inch", "to_unit": "cm"},
        _check_unit_convert,
    ),
    ("unit_calculate", {"expression": "3 m / 2 s"}, _check_unit_calculate),
    ("solve_ode", {"rhs": "-y", "y0": [1], "t_span": [0, 1]}, _check_solve_ode),
    (
        "optimize_minimize",
        {"objective": "(x0 - 2)**2", "x0": [0.0]},
        _check_optimize_minimize,
    ),
    ("matrix_compute", {"operation": "det", "a": [[1, 2], [3, 4]]}, _check_matrix_compute),
    ("find_roots", {"expr": "x**2 - 4", "bracket": [0, 3]}, _check_find_roots),
    (
        "fft_analysis",
        {"signal": [0, 1, 0, -1], "sample_rate": 1.0},
        _check_fft_analysis,
    ),
    (
        "interpolate_1d",
        {"x": [0, 1], "y": [0, 2], "xi": [0.5]},
        _check_interpolate_1d,
    ),
    ("describe", {"data": [1, 2, 3]}, _check_describe),
    (
        "distribution",
        {"operation": "ppf", "dist": "norm", "x": 0.5},
        _check_distribution,
    ),
    (
        "curve_fit",
        {"model": "a*x + b", "xdata": [0, 1, 2], "ydata": [1, 3, 5], "p0": [1, 1]},
        _check_curve_fit,
    ),
    (
        "plot",
        {
            "kind": "line",
            "data": {"x": [0, 1], "y": [0, 1]},
            "filename": "integration_smoke.png",
        },
        _check_plot,
    ),
    ("run_python", {"code": "print(1+1)"}, _check_run_python),
    ("t_test", {"sample1": [1, 2, 3, 4, 5], "popmean": 3}, _check_t_test),
    ("chi2_test", {"observed": [[10, 20], [30, 40]]}, _check_chi2_test),
    ("anova", {"groups": [[1, 2, 3], [4, 5, 6]]}, _check_anova),
    ("correlation", {"x": [1, 2, 3], "y": [2, 4, 6]}, _check_correlation),
    (
        "linear_regression",
        {"x": [1, 2, 3], "y": [3, 5, 7]},
        _check_linear_regression,
    ),
    ("dsolve_ode", {"equation": "Derivative(y(t), t) - y(t)"}, _check_dsolve_ode),
    (
        "gradient",
        {"expr": "x**2 + y**2", "variables": ["x", "y"]},
        _check_gradient,
    ),
    ("health_check", {}, _check_health_check),
    (
        "divergence",
        {"field": ["x", "y", "z"], "variables": ["x", "y", "z"]},
        _check_divergence,
    ),
    (
        "curl",
        {"field": ["-y", "x", "0"], "variables": ["x", "y", "z"]},
        _check_curl,
    ),
    (
        "laplacian",
        {"expr": "x**2 + y**2", "variables": ["x", "y"]},
        _check_laplacian,
    ),
    (
        "solve_heat_1d",
        {
            "alpha": 1.0,
            "length": 1.0,
            "nx": 20,
            "t_end": 0.01,
            "nt": 100,
            "initial": "sin(pi*x)",
        },
        _check_solve_heat_1d,
    ),
    (
        "solve_wave_1d",
        {
            "c": 1.0,
            "length": 1.0,
            "nx": 20,
            "t_end": 0.01,
            "nt": 20,
            "initial_displacement": "sin(pi*x)",
        },
        _check_solve_wave_1d,
    ),
    (
        "solve_laplace_2d",
        {
            "nx": 10,
            "ny": 10,
            "left": 0.0,
            "right": 0.0,
            "top": 100.0,
            "bottom": 0.0,
            "tol": 1e-6,
            "max_iter": 5000,
        },
        _check_solve_laplace_2d,
    ),
    (
        "solve_heat_3d",
        {
            "alpha": 1.0,
            "nx": 8,
            "ny": 8,
            "nz": 8,
            "t_end": 0.005,
            "nt": 20,
            "initial": "sin(pi*x)*sin(pi*y)*sin(pi*z)",
        },
        _check_solve_heat_3d,
    ),
    (
        "solve_laplace_3d",
        {
            "nx": 10,
            "ny": 10,
            "nz": 10,
            "z1": 100.0,
            "tol": 1e-5,
            "max_iter": 5000,
        },
        _check_solve_laplace_3d,
    ),
    (
        "solve_burgers_1d",
        {
            "nu": 0.05,
            "nx": 50,
            "t_end": 0.01,
            "nt": 200,
            "initial": "sin(pi*x)",
        },
        _check_solve_burgers_1d,
    ),
    (
        "solve_heat_1d_variable",
        {
            "alpha_expr": "1.0",
            "length": 1.0,
            "nx": 20,
            "t_end": 0.01,
            "nt": 100,
            "initial": "sin(pi*x)",
        },
        _check_solve_heat_1d_variable,
    ),
    (
        "solve_pde_1d",
        {
            "equation": "1.0*u_xx",
            "length": 1.0,
            "nx": 20,
            "t_end": 0.01,
            "nt": 100,
            "initial": "sin(pi*x)",
        },
        _check_solve_pde_1d,
    ),
    (
        "solve_wave_3d",
        {
            "c": 1.0,
            "nx": 8,
            "ny": 8,
            "nz": 8,
            "t_end": 0.005,
            "nt": 20,
            "initial_displacement": "sin(pi*x)*sin(pi*y)*sin(pi*z)",
        },
        _check_solve_wave_3d,
    ),
    (
        "solve_navier_stokes_2d",
        {
            "nu": 0.05,
            "nx": 16,
            "ny": 16,
            "t_end": 0.05,
            "nt": 50,
            "initial_vorticity": "-2*sin(x)*sin(y)",
        },
        _check_solve_navier_stokes_2d,
    ),
    (
        "filter_signal",
        {
            "signal": [0, 1, 0, -1] * 10,
            "kind": "lowpass",
            "cutoff": 1.0,
            "sample_rate": 10.0,
        },
        _check_filter_signal,
    ),
    ("convolve", {"signal1": [1, 2, 3], "signal2": [1, 1]}, _check_convolve),
    (
        "sparse_compute",
        {
            "operation": "solve",
            "a": {
                "rows": [0, 1, 1, 2],
                "cols": [0, 0, 1, 2],
                "values": [4, -1, 4, -1],
                "shape": [3, 3],
            },
            "b": [15, 6, 1],
        },
        _check_sparse_compute,
    ),
    (
        "power_spectrum",
        {"signal": [0, 1, 0, -1] * 20, "sample_rate": 10.0, "nperseg": 32},
        _check_power_spectrum,
    ),
    ("find_peaks", {"signal": [0, 1, 0, 2, 0, 1.5, 0]}, _check_find_peaks),
    (
        "monte_carlo",
        {
            "expression": "x + y",
            "variables": {
                "x": {"dist": "norm", "loc": 0, "scale": 1},
                "y": {"dist": "norm", "loc": 0, "scale": 1},
            },
            "n": 1000,
            "seed": 1,
        },
        _check_monte_carlo,
    ),
    (
        "linear_program",
        {"c": [-3, -2], "a_ub": [[1, 1], [1, 0], [0, 1]], "b_ub": [4, 2, 3]},
        _check_linear_program,
    ),
    (
        "integer_program",
        {
            "c": [-5, -4],
            "a_ub": [[2, 3]],
            "b_ub": [6],
            "bounds": [[0, 1], [0, 1]],
        },
        _check_integer_program,
    ),
    ("save_data", {"data": [1, 2, 3], "filename": "smoke.csv"}, _check_save_data),
    ("load_data", {"path": "smoke_load.csv"}, _check_load_data),
    ("describe_file", {"path": "smoke_desc.csv"}, _check_describe_file),
    (
        "plot_file",
        {
            "kind": "line",
            "path": "smoke_plot.csv",
            "x": "x",
            "y": "y",
            "filename": "smoke_plot.png",
        },
        _check_plot_file,
    ),
    (
        "global_optimize",
        {
            "objective": "(x0**2 - 1)**2 + x1**2",
            "bounds": [[-3, 3], [-3, 3]],
            "method": "differential_evolution",
            "maxiter": 200,
            "seed": 42,
        },
        _check_global_optimize,
    ),
    (
        "glm",
        {
            "family": "gaussian",
            "formula": "y ~ x",
            "data": {
                "x": [1, 2, 3, 4, 5, 6, 7, 8],
                "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1],
            },
        },
        _check_glm,
    ),
    (
        "mixed_lm",
        {"formula": "y ~ x", "groups": "g", "data": _MIXED_MODEL_DATA},
        _check_mixed_lm,
    ),
    (
        "nonlinear_mixed_lm",
        {
            "model": "a*exp(b*x)",
            "data": _NONLINEAR_MIXED_DATA,
            "groups": "g",
            "x": "x",
            "y": "y",
            "p0": {"a": 2.0, "b": 0.5},
        },
        _check_nonlinear_mixed_lm,
    ),
    (
        "cox_regression",
        {
            "formula": "x",
            "data": {
                "time": [2, 5, 3, 8, 4, 9, 6, 12, 7, 15, 10, 18, 11, 20, 14],
                "event": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            },
            "duration_col": "time",
            "event_col": "event",
        },
        _check_cox_regression,
    ),
    (
        "kaplan_meier",
        {"durations": [1, 2, 3], "events": [1, 1, 0]},
        _check_kaplan_meier,
    ),
    (
        "gee",
        {
            "formula": "y ~ x",
            "groups": "g",
            "data": _MIXED_MODEL_DATA,
            "family": "gaussian",
            "cov_struct": "exchangeable",
        },
        _check_gee,
    ),
    (
        "arima",
        {"series": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "order": [1, 0, 0], "forecast_steps": 3},
        _check_arima,
    ),
    (
        "time_series_decompose",
        {"series": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], "period": 2},
        _check_time_series_decompose,
    ),
    (
        "stationarity_test",
        {"series": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        _check_stationarity_test,
    ),
    (
        # nperseg=32 gives a 3.125 Hz bin spacing at fs=100, so a 1 Hz tone
        # cannot be resolved; nperseg=100 (1 Hz spacing) makes the 1 Hz peak
        # assertion meaningful.
        "stft",
        {
            "signal": [math.sin(2.0 * math.pi * n / 100.0) for n in range(100)],
            "sample_rate": 100.0,
            "nperseg": 100,
        },
        _check_stft,
    ),
    (
        "wavelet_transform",
        {
            "signal": [math.sin(2.0 * math.pi * n / 64.0) for n in range(64)],
            "wavelet": "db4",
        },
        _check_wavelet_transform,
    ),
    (
        "wavelet_denoise",
        {
            "signal": [
                math.sin(2.0 * math.pi * n / 64.0) + 0.01 * ((-1) ** n)
                for n in range(64)
            ],
            "wavelet": "db4",
        },
        _check_wavelet_denoise,
    ),
    (
        "analyze_dataset",
        {"path": "smoke_analysis.csv", "target": "y", "plot": False},
        _check_analyze_dataset,
    ),
    (
        "forecast",
        {
            "series": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "steps": 3,
            "method": "arima",
            "order": [1, 1, 0],
        },
        _check_forecast,
    ),
]


def _prepare_smoke_load() -> None:
    _call_ok("save_data", {"data": [1, 2, 3], "filename": "smoke_load.csv"})


def _prepare_smoke_desc() -> None:
    _call_ok("save_data", {"data": [1, 2, 3], "filename": "smoke_desc.csv"})


def _prepare_smoke_plot() -> None:
    _call_ok(
        "save_data",
        {"data": {"x": [0, 1, 2], "y": [0, 1, 4]}, "filename": "smoke_plot.csv"},
    )


def _prepare_smoke_analysis() -> None:
    _call_ok(
        "save_data",
        {
            "data": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]},
            "filename": "smoke_analysis.csv",
        },
    )


PREPARE_SMOKE_CASE = {
    "load_data": _prepare_smoke_load,
    "describe_file": _prepare_smoke_desc,
    "plot_file": _prepare_smoke_plot,
    "analyze_dataset": _prepare_smoke_analysis,
}


def test_expected_tools_constant_is_exactly_sixty_five_unique_names() -> None:
    assert len(EXPECTED_TOOLS) == 65
    assert all(isinstance(name, str) and name for name in EXPECTED_TOOLS)


def test_list_tools_matches_expected_exactly() -> None:
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS
    assert not missing and not extra, (
        f"tool registration mismatch: missing={sorted(missing)} extra={sorted(extra)}"
    )
    assert names == EXPECTED_TOOLS


def test_smoke_case_names_cover_every_expected_tool() -> None:
    assert {name for name, _, _ in SMOKE_CASES} == EXPECTED_TOOLS


@pytest.mark.parametrize(
    "name,args,checker",
    SMOKE_CASES,
    ids=[case[0] for case in SMOKE_CASES],
)
def test_tool_call_smoke(name: str, args: dict, checker, sci_output_dir: Path) -> None:
    prepare = PREPARE_SMOKE_CASE.get(name)
    if prepare is not None:
        prepare()
    result = asyncio.run(mcp.call_tool(name, args))
    payload = _decode_payload(result)
    assert payload.get("ok") is True, f"{name} returned an error payload: {payload}"
    checker(payload["result"], sci_output_dir)


def test_optimize_minimize_with_equality_constraint() -> None:
    result = asyncio.run(
        mcp.call_tool(
            "optimize_minimize",
            {
                "objective": "x0**2 + x1**2",
                "x0": [0.0, 0.0],
                "constraints": [{"kind": "eq", "expr": "x0 + x1 - 1"}],
            },
        )
    )
    payload = _decode_payload(result)
    assert payload.get("ok") is True, (
        f"optimize_minimize returned an error payload: {payload}"
    )
    value = payload["result"]
    assert value["success"] is True
    assert value["x"][0] == pytest.approx(0.5, abs=1e-5)
    assert value["x"][1] == pytest.approx(0.5, abs=1e-5)
    assert value["fun"] == pytest.approx(0.5, abs=1e-5)


def test_fft_analysis_accepts_file_reference(sci_output_dir: Path) -> None:
    """A 1 Hz sine stored with save_data can be read back via ``@file``."""
    signal = [math.sin(2.0 * math.pi * n / 16.0) for n in range(64)]
    saved = _call_ok("save_data", {"data": signal, "filename": "smoke_fft.csv"})
    assert Path(saved["path"]).is_file()
    assert Path(saved["path"]).parent == sci_output_dir.resolve()

    value = _call_ok(
        "fft_analysis", {"signal": "@smoke_fft.csv", "sample_rate": 16.0}
    )
    assert value["dominant_frequency"] == pytest.approx(1.0, abs=0.2)


def test_sparse_compute_conjugate_gradient(sci_output_dir: Path) -> None:
    """The CG iterative solver agrees with a dense solve on a 3x3 SPD system."""
    result = asyncio.run(
        mcp.call_tool(
            "sparse_compute",
            {
                "operation": "solve",
                "a": _SPARSE_CG_MATRIX,
                "b": [15, 6, 1],
                "solver": "cg",
            },
        )
    )
    payload = _decode_payload(result)
    assert payload.get("ok") is True, (
        f"sparse_compute[cg] returned an error payload: {payload}"
    )
    _check_sparse_compute_cg(payload["result"], sci_output_dir)
