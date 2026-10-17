"""Tests for the wavelet tools (public interface only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pywt

from sci_mcp.tools.wavelets import register, wavelet_denoise, wavelet_transform


def _sine(
    frequency: float = 2.0,
    sample_rate: float = 200.0,
    duration: float = 4.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def _write_csv(path, values) -> None:
    np.savetxt(path, np.asarray(values, dtype=float), delimiter=",")


def test_roundtrip_reconstruction_error_is_negligible(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine()

    # Independent ground truth: pywt's own decompose/reconstruct pair.
    coefficients = pywt.wavedec(signal, "db4", mode="symmetric")
    reconstructed = pywt.waverec(coefficients, "db4", mode="symmetric")
    overlap = min(signal.size, reconstructed.size)
    assert float(np.max(np.abs(reconstructed[:overlap] - signal[:overlap]))) < 1e-8

    # The coefficient arrays saved by the tool must reconstruct the same signal.
    result = wavelet_transform(signal.tolist(), "db4")
    assert result["ok"] is True
    payload = result["result"]
    level = payload["level"]
    with np.load(payload["path"]) as data:
        saved_coefficients = [data[f"cA{level}"]] + [
            data[f"cD{detail}"] for detail in range(level, 0, -1)
        ]
    from_saved = pywt.waverec(saved_coefficients, "db4", mode="symmetric")
    overlap = min(signal.size, from_saved.size)
    assert float(np.max(np.abs(from_saved[:overlap] - signal[:overlap]))) < 1e-8


def test_transform_level_and_coefficient_lengths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine(duration=5.12)  # 1024 samples at 200 Hz

    result = wavelet_transform(signal.tolist(), "db4")

    assert result["ok"] is True
    payload = result["result"]
    expected_level = int(pywt.dwt_max_level(signal.size, pywt.Wavelet("db4").dec_len))
    assert payload["level"] == expected_level
    assert len(payload["coefficients"]) == expected_level + 1

    expected = pywt.wavedec(signal, "db4", level=expected_level, mode="symmetric")
    assert [entry["length"] for entry in payload["coefficients"]] == [
        int(array.size) for array in expected
    ]
    for entry in payload["coefficients"]:
        assert entry["length"] > 0
        assert entry["energy"] >= 0.0
        assert len(entry["preview"]) <= 10
    assert payload["coefficients"][0]["kind"] == "approximation"
    assert all(entry["kind"] == "detail" for entry in payload["coefficients"][1:])


def test_transform_saves_full_coefficients_npz(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine()

    result = wavelet_transform(signal.tolist(), "db4", level=3)

    assert result["ok"] is True
    payload = result["result"]
    saved = Path(payload["path"])
    assert saved.is_file()
    assert saved.suffix == ".npz"

    with np.load(saved) as data:
        assert len(data.files) == payload["level"] + 1
        assert data["cA3"].size == payload["coefficients"][0]["length"]
        for index, detail in enumerate(range(3, 0, -1)):
            assert data[f"cD{detail}"].size == payload["coefficients"][index + 1]["length"]


def test_denoise_recovers_sine_from_noise(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    sample_rate = 200.0
    clean = _sine(frequency=2.0, sample_rate=sample_rate, duration=4.0)
    noisy = clean + 0.4 * rng.standard_normal(clean.size)

    result = wavelet_denoise(noisy.tolist(), "db4", level=3)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["mode"] == "soft"
    assert payload["original_length"] == noisy.size
    denoised = np.asarray(payload["denoised"])
    assert denoised.size == noisy.size
    correlation = float(np.corrcoef(denoised, clean)[0, 1])
    assert correlation > 0.9
    assert correlation > float(np.corrcoef(noisy, clean)[0, 1])


def test_explicit_threshold_is_applied(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 200.0
    high_frequency = _sine(frequency=40.0, sample_rate=sample_rate, duration=4.0)

    result = wavelet_denoise(high_frequency.tolist(), "db4", level=3, threshold=10.0)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["threshold"] == 10.0
    denoised = np.asarray(payload["denoised"])
    # A 40 Hz tone lives entirely in the detail bands, which a threshold of 10 zeroes.
    assert float(np.std(denoised)) < 0.1 * float(np.std(high_frequency))


def test_transform_accepts_at_file_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine()
    _write_csv(tmp_path / "wavelet_signal.csv", signal)

    from_file = wavelet_transform("@wavelet_signal.csv", "db4", level=3)
    from_list = wavelet_transform(signal.tolist(), "db4", level=3)

    assert from_file["ok"] is True
    assert from_list["ok"] is True
    assert from_file["result"]["level"] == from_list["result"]["level"]
    assert [entry["length"] for entry in from_file["result"]["coefficients"]] == [
        entry["length"] for entry in from_list["result"]["coefficients"]
    ]


def test_unknown_wavelet_returns_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine().tolist()

    transform = wavelet_transform(signal, "not_a_wavelet")
    denoise = wavelet_denoise(signal, "not_a_wavelet")

    assert transform["ok"] is False
    assert "error" in transform
    assert denoise["ok"] is False
    assert "error" in denoise


def test_level_too_high_returns_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine(duration=0.32).tolist()  # 64 samples

    transform = wavelet_transform(signal, "db4", level=100)
    denoise = wavelet_denoise(signal, "db4", level=100)

    assert transform["ok"] is False
    assert "error" in transform
    assert denoise["ok"] is False
    assert "error" in denoise


def test_outputs_are_json_serializable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    signal = _sine(duration=2.0)

    outputs = [
        wavelet_transform(signal.tolist(), "db4", level=3),
        wavelet_denoise(signal.tolist(), "db4", level=3),
    ]

    for output in outputs:
        assert output["ok"] is True
        json.dumps(output, allow_nan=False)


def test_register_registers_both_tools() -> None:
    class _FakeMCP:
        def __init__(self) -> None:
            self.names: list[str] = []

        def tool(self):
            def decorator(func):
                self.names.append(func.__name__)
                return func

            return decorator

    fake = _FakeMCP()
    register(fake)  # type: ignore[arg-type]

    assert set(fake.names) == {"wavelet_transform", "wavelet_denoise"}
