"""Tests for the signal processing tools (public interface only)."""

from __future__ import annotations

import json

import numpy as np

from sci_mcp.tools.signal_processing import (
    convolve,
    filter_signal,
    find_peaks,
    power_spectrum,
    register,
)


def _tone_mix(sample_rate: float = 100.0, duration: float = 4.0) -> np.ndarray:
    """1 Hz tone of amplitude 1 plus a 30 Hz tone of amplitude 0.5."""
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    return np.sin(2.0 * np.pi * 1.0 * t) + 0.5 * np.sin(2.0 * np.pi * 30.0 * t)


def _amplitude_at(samples: np.ndarray, frequency: float, sample_rate: float) -> float:
    """Single-sided amplitude of ``frequency`` measured with numpy's rfft."""
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    index = int(np.argmin(np.abs(frequencies - frequency)))
    return float(2.0 * np.abs(spectrum[index]) / samples.size)


def test_lowpass_rejects_30hz_and_keeps_1hz() -> None:
    sample_rate = 100.0
    signal = _tone_mix(sample_rate)

    result = filter_signal(signal.tolist(), "lowpass", 5.0, sample_rate, order=4)

    assert result["ok"] is True
    filtered = np.asarray(result["result"]["filtered"])
    assert filtered.size == signal.size
    # Original amplitudes: 1 Hz -> 1.0, 30 Hz -> 0.5.
    assert _amplitude_at(filtered, 1.0, sample_rate) > 0.8 * 1.0
    assert _amplitude_at(filtered, 30.0, sample_rate) < 0.2 * 0.5


def test_highpass_keeps_30hz_and_rejects_1hz() -> None:
    sample_rate = 100.0
    signal = _tone_mix(sample_rate)

    result = filter_signal(signal.tolist(), "highpass", 10.0, sample_rate, order=4)

    assert result["ok"] is True
    filtered = np.asarray(result["result"]["filtered"])
    assert filtered.size == signal.size
    assert _amplitude_at(filtered, 30.0, sample_rate) > 0.8 * 0.5
    assert _amplitude_at(filtered, 1.0, sample_rate) < 0.2 * 1.0


def test_bandpass_returns_same_length() -> None:
    sample_rate = 100.0
    signal = _tone_mix(sample_rate)

    result = filter_signal(signal.tolist(), "bandpass", [5.0, 20.0], sample_rate, order=4)

    assert result["ok"] is True
    assert len(result["result"]["filtered"]) == signal.size
    assert result["result"]["kind"] == "bandpass"
    assert result["result"]["cutoff"] == [5.0, 20.0]


def test_cutoff_above_nyquist_is_rejected() -> None:
    signal = _tone_mix().tolist()

    result = filter_signal(signal, "lowpass", 60.0, 100.0, order=4)

    assert result["ok"] is False
    assert "error" in result


def test_bandpass_low_not_below_high_is_rejected() -> None:
    signal = _tone_mix().tolist()

    result = filter_signal(signal, "bandpass", [20.0, 10.0], 100.0, order=4)

    assert result["ok"] is False
    assert "error" in result


def test_too_short_signal_fails_filtfilt() -> None:
    short = [0.0, 1.0, 0.0, -1.0, 0.0]

    result = filter_signal(short, "lowpass", 5.0, 100.0, order=4)

    assert result["ok"] is False
    assert "error" in result
    assert "short" in result["error"].lower()


def test_convolve_full_hand_computed() -> None:
    result = convolve([1.0, 2.0, 3.0], [1.0, 1.0], "full")

    assert result["ok"] is True
    assert result["result"]["result"] == [1.0, 3.0, 5.0, 3.0]
    assert result["result"]["mode"] == "full"


def test_convolve_same_and_valid_shapes() -> None:
    same = convolve([1.0, 2.0, 3.0], [1.0, 1.0], "same")
    valid = convolve([1.0, 2.0, 3.0], [1.0, 1.0], "valid")

    assert same["ok"] is True
    assert valid["ok"] is True
    assert len(same["result"]["result"]) == 3
    assert len(valid["result"]["result"]) == 2
    assert valid["result"]["result"] == [3.0, 5.0]


def test_power_spectrum_peak_is_1hz() -> None:
    sample_rate = 100.0
    t = np.arange(0.0, 10.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 1.0 * t)

    result = power_spectrum(signal.tolist(), sample_rate, nperseg=512)

    assert result["ok"] is True
    assert abs(result["result"]["peak_frequency"] - 1.0) <= 0.2
    assert len(result["result"]["frequencies"]) == len(result["result"]["psd"])


def test_nperseg_longer_than_signal_is_rejected() -> None:
    signal = np.zeros(100).tolist()

    result = power_spectrum(signal, 100.0, nperseg=150)

    assert result["ok"] is False
    assert "error" in result


def test_find_peaks_indices() -> None:
    result = find_peaks([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0])

    assert result["ok"] is True
    assert result["result"]["indices"] == [1, 3, 5]
    assert result["result"]["values"] == [1.0, 2.0, 1.5]
    assert result["result"]["count"] == 3


def test_find_peaks_with_height_filter() -> None:
    result = find_peaks([0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0], height=1.2)

    assert result["ok"] is True
    assert result["result"]["indices"] == [3, 5]
    assert result["result"]["count"] == 2
    assert result["result"]["properties"]["peak_heights"] == [2.0, 1.5]


def test_outputs_are_json_serializable() -> None:
    sample_rate = 100.0
    signal = _tone_mix(sample_rate, duration=3.0)

    outputs = [
        filter_signal(signal.tolist(), "lowpass", 5.0, sample_rate, order=4),
        convolve(signal.tolist()[:10], [1.0, 1.0], "full"),
        power_spectrum(signal.tolist(), sample_rate, nperseg=64),
        find_peaks([0.0, 1.0, 0.0, 2.0, 0.0]),
    ]

    for output in outputs:
        assert output["ok"] is True
        assert isinstance(json.dumps(output), str)


def test_long_filtered_output_is_downsampled() -> None:
    sample_rate = 100.0
    t = np.arange(0.0, 30.0, 1.0 / sample_rate)  # 3000 samples
    signal = np.sin(2.0 * np.pi * 3.0 * t)

    result = filter_signal(signal.tolist(), "lowpass", 10.0, sample_rate, order=2)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["downsampled"] is True
    assert payload["original_length"] == 3000
    assert len(payload["filtered"]) <= 2000


def test_register_registers_all_tools() -> None:
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

    assert set(fake.names) == {"filter_signal", "convolve", "power_spectrum", "find_peaks", "stft"}


# ---------------------------------------------------------------------------
# "@<path>" file-reference inputs (BRIEF section 12)
# ---------------------------------------------------------------------------


def _write_csv(path, values) -> None:
    np.savetxt(path, np.asarray(values, dtype=float), delimiter=",")


def test_filter_signal_accepts_at_file_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    signal = _tone_mix(sample_rate, duration=4.0)
    _write_csv(tmp_path / "tone.csv", signal)

    result = filter_signal("@tone.csv", "lowpass", 5.0, sample_rate, order=4)

    assert result["ok"] is True
    assert len(result["result"]["filtered"]) == signal.size


def test_convolve_accepts_at_file_inputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    _write_csv(tmp_path / "first.csv", [1.0, 2.0, 3.0])
    _write_csv(tmp_path / "second.csv", [1.0, 1.0])

    from_file = convolve("@first.csv", "@second.csv", "full")
    from_list = convolve([1.0, 2.0, 3.0], [1.0, 1.0], "full")

    assert from_file["ok"] is True
    assert from_list["ok"] is True
    assert from_file["result"]["result"] == from_list["result"]["result"]
    assert from_file["result"]["result"] == [1.0, 3.0, 5.0, 3.0]


def test_power_spectrum_accepts_at_file_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    t = np.arange(0.0, 10.0, 1.0 / sample_rate)
    _write_csv(tmp_path / "tone_1hz.csv", np.sin(2.0 * np.pi * 1.0 * t))

    result = power_spectrum("@tone_1hz.csv", sample_rate, nperseg=512)

    assert result["ok"] is True
    assert abs(result["result"]["peak_frequency"] - 1.0) <= 0.2


def test_find_peaks_accepts_at_file_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    _write_csv(tmp_path / "peaks.csv", [0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0])

    result = find_peaks("@peaks.csv")

    assert result["ok"] is True
    assert result["result"]["indices"] == [1, 3, 5]
    assert result["result"]["values"] == [1.0, 2.0, 1.5]


def test_missing_at_file_returns_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))

    result = filter_signal("@does_not_exist.csv", "lowpass", 5.0, 100.0, order=4)

    assert result["ok"] is False
    assert "error" in result
    assert "filter_signal failed" in result["error"]


# ---------------------------------------------------------------------------
# Short-time Fourier transform (STFT)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from sci_mcp.tools.signal_processing import stft  # noqa: E402


def test_stft_peak_frequency_is_1hz(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 1.0 * t)

    result = stft(signal.tolist(), sample_rate, nperseg=200)

    assert result["ok"] is True
    payload = result["result"]
    assert abs(payload["peak_frequency"] - 1.0) <= 0.2
    assert payload["shape"] == [len(payload["frequencies"]), len(payload["times"])]
    assert len(payload["magnitude"]) == payload["shape"][0]
    assert all(len(row) == payload["shape"][1] for row in payload["magnitude"])


def test_stft_tracks_chirp_frequency_increase(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    # Linear chirp from 1 Hz to 20 Hz over 2 s: phase = 2*pi*(f0*t + (f1-f0)/(2*T)*t^2).
    signal = np.sin(2.0 * np.pi * (1.0 * t + (20.0 - 1.0) / (2.0 * 2.0) * t**2))

    result = stft(signal.tolist(), sample_rate, nperseg=64)

    assert result["ok"] is True
    payload = result["result"]
    frequencies = np.asarray(payload["frequencies"])
    times = np.asarray(payload["times"])
    magnitude = np.asarray(payload["magnitude"])

    early_column = int(np.argmin(np.abs(times - 0.5)))
    late_column = int(np.argmin(np.abs(times - 1.5)))
    early_peak = float(frequencies[int(np.argmax(magnitude[:, early_column]))])
    late_peak = float(frequencies[int(np.argmax(magnitude[:, late_column]))])

    assert early_peak < 10.0
    assert late_peak > 10.0
    assert late_peak > early_peak + 3.0


def test_stft_large_matrix_is_downsampled_and_saved(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 1000.0
    t = np.arange(0.0, 1.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 50.0 * t)

    result = stft(signal.tolist(), sample_rate, nperseg=256)

    assert result["ok"] is True
    payload = result["result"]
    assert payload["downsampled"] is True
    assert len(payload["magnitude"]) <= 100
    assert len(payload["frequencies"]) <= 100
    assert len(payload["times"]) <= 100
    assert payload["path"] is not None

    saved = Path(payload["path"])
    assert saved.is_file()
    with np.load(saved) as data:
        assert tuple(data["magnitude"].shape) == tuple(payload["original_shape"])
        assert data["magnitude"].shape[0] > 100
        assert data["frequencies"].shape[0] == data["magnitude"].shape[0]
        assert data["times"].shape[0] == data["magnitude"].shape[1]
    assert json.dumps(result, allow_nan=False)


def test_stft_saves_spectrogram_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    signal = np.sin(2.0 * np.pi * 5.0 * t)

    result = stft(
        signal.tolist(),
        sample_rate,
        nperseg=64,
        save_spectrogram=True,
        filename="stft_spectrogram.png",
    )

    assert result["ok"] is True
    output = result["result"]["spectrogram_path"]
    assert output is not None
    assert Path(output).is_file()
    assert Path(output).suffix == ".png"


def test_stft_accepts_at_file_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    sample_rate = 100.0
    t = np.arange(0.0, 2.0, 1.0 / sample_rate)
    _write_csv(tmp_path / "stft_tone.csv", np.sin(2.0 * np.pi * 1.0 * t))

    result = stft("@stft_tone.csv", sample_rate, nperseg=200)

    assert result["ok"] is True
    assert abs(result["result"]["peak_frequency"] - 1.0) <= 0.2


def test_stft_rejects_nperseg_longer_than_signal() -> None:
    result = stft(np.zeros(50).tolist(), 100.0, nperseg=64)

    assert result["ok"] is False
    assert "error" in result


def test_stft_rejects_invalid_window() -> None:
    result = stft(np.zeros(200).tolist(), 100.0, nperseg=64, window="not_a_window")

    assert result["ok"] is False
    assert "error" in result


def test_stft_rejects_invalid_noverlap() -> None:
    result = stft(np.zeros(200).tolist(), 100.0, nperseg=64, noverlap=64)

    assert result["ok"] is False
    assert "error" in result
