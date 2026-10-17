"""Signal processing tools: filtering, convolution, power spectral density, peak finding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.mcpserver import MCPServer
from scipy import signal as sp_signal

from ..core import err, load_array, ok, output_dir

# BRIEF section 11: never return more than 2000 numeric points in one array.
_MAX_POINTS = 2000

# STFT time-frequency matrices are capped at 100 x 100 returned cells.
_STFT_MAX_CELLS = 100

_ARTIFACT_COUNTER = 0

_SUPPORTED_FILTER_KINDS = ("lowpass", "highpass", "bandpass", "bandstop")
_BAND_KINDS = ("bandpass", "bandstop")
_SUPPORTED_CONVOLVE_MODES = ("full", "same", "valid")


def _coerce_series(value: Any, name: str) -> np.ndarray:
    """Validate a list or ``@<path>`` file reference into a 1-D float array."""
    if not isinstance(value, (list, tuple, str)):
        raise ValueError(f"{name} must be a list of numbers or an '@<path>' reference")
    array = np.asarray(load_array(value, dtype=float), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional series")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _series_payload(values: np.ndarray) -> tuple[list[float], dict[str, Any]]:
    """Convert a series to a list, evenly downsampling it when above 2000 points."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size <= _MAX_POINTS:
        return flat.tolist(), {}
    step = (flat.size + _MAX_POINTS - 1) // _MAX_POINTS  # ceil(size / limit)
    reduced = flat[::step]
    meta: dict[str, Any] = {
        "downsampled": True,
        "original_length": int(flat.size),
        "downsample_step": int(step),
    }
    return reduced.tolist(), meta


def _as_positive_rate(sample_rate: Any) -> float:
    """Validate a sample rate as a finite positive float."""
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be a finite number greater than 0")
    return rate


def filter_signal(
    signal: list[float] | str,
    kind: str,
    cutoff: float | list[float],
    sample_rate: float,
    order: int = 4,
) -> dict:
    """Filter a signal with a Butterworth design and zero-phase filtfilt.

    ``kind`` is lowpass, highpass, bandpass or bandstop. ``cutoff`` is in Hz;
    lowpass/highpass take a scalar and bandpass/bandstop take ``[low, high]``.
    """
    try:
        if not isinstance(kind, str) or kind.strip().lower() not in _SUPPORTED_FILTER_KINDS:
            return err(
                f"filter_signal failed: unknown kind {kind!r}; "
                f"supported kinds are {list(_SUPPORTED_FILTER_KINDS)}"
            )
        selected_kind = kind.strip().lower()

        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            return err("filter_signal failed: order must be an integer between 1 and 8")
        selected_order = int(order)
        if not 1 <= selected_order <= 8:
            return err(f"filter_signal failed: order must be between 1 and 8 (got {selected_order})")

        rate = _as_positive_rate(sample_rate)
        nyquist = rate / 2.0
        samples = _coerce_series(signal, "signal")

        if selected_kind in _BAND_KINDS:
            if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
                return err(
                    f"filter_signal failed: {selected_kind} requires cutoff as a [low, high] list"
                )
            low = float(cutoff[0])
            high = float(cutoff[1])
            if not (np.isfinite(low) and np.isfinite(high)):
                return err("filter_signal failed: cutoff values must be finite")
            if not 0.0 < low < high < nyquist:
                return err(
                    "filter_signal failed: cutoff must satisfy 0 < low < high < "
                    f"sample_rate/2 ({nyquist})"
                )
            critical = [low, high]
            cutoff_payload: float | list[float] = [low, high]
        else:
            if isinstance(cutoff, (list, tuple)):
                return err(
                    f"filter_signal failed: {selected_kind} requires a scalar cutoff in Hz"
                )
            value = float(cutoff)
            if not np.isfinite(value):
                return err("filter_signal failed: cutoff must be finite")
            if not 0.0 < value < nyquist:
                return err(
                    "filter_signal failed: cutoff must satisfy 0 < cutoff < "
                    f"sample_rate/2 ({nyquist})"
                )
            critical = value
            cutoff_payload = value

        b, a = sp_signal.butter(selected_order, critical, btype=selected_kind, fs=rate)
        padlen = 3 * max(len(a), len(b))
        try:
            filtered = sp_signal.filtfilt(b, a, samples)
        except ValueError as exc:
            message = str(exc)
            if "padlen" in message.lower() or "length" in message.lower():
                return err(
                    f"filter_signal failed: signal is too short for filtfilt with order "
                    f"{selected_order} ({samples.size} samples); a longer signal "
                    f"(more than {padlen} samples) is required"
                )
            return err(f"filter_signal failed: {message}")

        filtered_payload, meta = _series_payload(filtered)
        payload: dict[str, Any] = {
            "filtered": filtered_payload,
            "kind": selected_kind,
            "cutoff": cutoff_payload,
            "order": selected_order,
            "sample_rate": rate,
        }
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"filter_signal failed: {exc}")


def convolve(
    signal1: list[float] | str, signal2: list[float] | str, mode: str = "full"
) -> dict:
    """Compute the discrete convolution of two signals with numpy.convolve."""
    try:
        if not isinstance(mode, str) or mode.strip().lower() not in _SUPPORTED_CONVOLVE_MODES:
            return err(
                f"convolve failed: unknown mode {mode!r}; "
                f"supported modes are {list(_SUPPORTED_CONVOLVE_MODES)}"
            )
        selected_mode = mode.strip().lower()

        first = _coerce_series(signal1, "signal1")
        second = _coerce_series(signal2, "signal2")

        result = np.convolve(first, second, mode=selected_mode)
        result_payload, meta = _series_payload(result)
        payload: dict[str, Any] = {"result": result_payload, "mode": selected_mode}
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"convolve failed: {exc}")


def power_spectrum(
    signal: list[float] | str, sample_rate: float, nperseg: int = 256
) -> dict:
    """Estimate the power spectral density with Welch's method."""
    try:
        rate = _as_positive_rate(sample_rate)
        samples = _coerce_series(signal, "signal")

        if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
            return err("power_spectrum failed: nperseg must be an integer")
        selected_nperseg = int(nperseg)
        if selected_nperseg < 8:
            return err(f"power_spectrum failed: nperseg must be at least 8 (got {selected_nperseg})")
        if selected_nperseg > samples.size:
            return err(
                "power_spectrum failed: nperseg must not exceed the signal length "
                f"(nperseg={selected_nperseg}, length={samples.size})"
            )

        frequencies, psd = sp_signal.welch(samples, fs=rate, nperseg=selected_nperseg)
        peak_frequency = float(frequencies[int(np.argmax(psd))])

        frequencies_payload, meta = _series_payload(frequencies)
        psd_payload, _ = _series_payload(psd)
        payload: dict[str, Any] = {
            "frequencies": frequencies_payload,
            "psd": psd_payload,
            "peak_frequency": peak_frequency,
            "nperseg": selected_nperseg,
            "sample_rate": rate,
        }
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"power_spectrum failed: {exc}")


def find_peaks(
    signal: list[float] | str,
    height: float | None = None,
    distance: int | None = None,
    prominence: float | None = None,
) -> dict:
    """Find local maxima in a signal, optionally filtered by height, distance or prominence."""
    try:
        samples = _coerce_series(signal, "signal")

        options: dict[str, Any] = {}
        if height is not None:
            selected_height = float(height)
            if not np.isfinite(selected_height):
                return err("find_peaks failed: height must be finite")
            options["height"] = selected_height
        if distance is not None:
            if isinstance(distance, bool) or not isinstance(distance, (int, np.integer)):
                return err("find_peaks failed: distance must be a positive integer")
            selected_distance = int(distance)
            if selected_distance < 1:
                return err("find_peaks failed: distance must be at least 1")
            options["distance"] = selected_distance
        if prominence is not None:
            selected_prominence = float(prominence)
            if not np.isfinite(selected_prominence):
                return err("find_peaks failed: prominence must be finite")
            if selected_prominence < 0:
                return err("find_peaks failed: prominence must be non-negative")
            options["prominence"] = selected_prominence

        indices, properties = sp_signal.find_peaks(samples, **options)
        indices = np.asarray(indices, dtype=np.int64)

        meta: dict[str, Any] = {}
        if indices.size > _MAX_POINTS:
            original_count = int(indices.size)
            step = (original_count + _MAX_POINTS - 1) // _MAX_POINTS
            keep = np.arange(0, original_count, step)
            indices = indices[keep]
            properties = {
                key: np.asarray(value).reshape(-1)[keep].tolist()
                for key, value in properties.items()
            }
            meta = {
                "downsampled": True,
                "original_length": original_count,
                "downsample_step": int(step),
            }
        else:
            properties = {
                key: np.asarray(value).tolist() for key, value in properties.items()
            }

        values = samples[indices]
        payload: dict[str, Any] = {
            "indices": indices.tolist(),
            "values": values.tolist(),
            "properties": properties,
            "count": int(indices.size),
        }
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"find_peaks failed: {exc}")


def _evenly_spaced_indices(size: int, limit: int) -> np.ndarray:
    """Return evenly spaced indices covering the full range, at most ``limit`` of them."""
    if size <= limit:
        return np.arange(size)
    return np.unique(np.rint(np.linspace(0.0, size - 1, limit)).astype(int))


def _new_artifact_path(prefix: str, suffix: str) -> Path:
    """Allocate a unique output path inside output_dir() without overwriting files."""
    global _ARTIFACT_COUNTER
    directory = output_dir()
    for _ in range(1000):
        _ARTIFACT_COUNTER += 1
        candidate = directory / f"{prefix}_{_ARTIFACT_COUNTER:06d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


def _resolve_spectrogram_path(filename: str | None) -> Path:
    """Validate a caller-supplied PNG basename and return its path under output_dir()."""
    if filename is None:
        return _new_artifact_path("stft_spectrogram", ".png")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    name = filename.strip()
    if name != Path(name).name or ".." in name:
        raise ValueError("filename must be a bare file name without path separators or '..'")
    if any(char in name for char in ("/", "\\", ":")):
        raise ValueError("filename must be a bare file name without path separators or '..'")
    if not Path(name).suffix:
        name = f"{name}.png"
    return output_dir() / name


def stft(
    signal: list[float] | str,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    save_spectrogram: bool = False,
    filename: str | None = None,
) -> dict:
    """Compute a short-time Fourier transform and return the magnitude spectrogram.

    The full magnitude matrix is saved as .npz when it exceeds 100 x 100 cells.
    ``save_spectrogram`` additionally renders a PNG with matplotlib (Agg).
    """
    try:
        rate = _as_positive_rate(sample_rate)
        samples = _coerce_series(signal, "signal")

        if isinstance(nperseg, bool) or not isinstance(nperseg, (int, np.integer)):
            return err("stft failed: nperseg must be an integer")
        selected_nperseg = int(nperseg)
        if selected_nperseg < 8:
            return err(f"stft failed: nperseg must be at least 8 (got {selected_nperseg})")
        if selected_nperseg > samples.size:
            return err(
                "stft failed: nperseg must not exceed the signal length "
                f"(nperseg={selected_nperseg}, length={samples.size})"
            )

        if noverlap is None:
            selected_noverlap = selected_nperseg // 2
        else:
            if isinstance(noverlap, bool) or not isinstance(noverlap, (int, np.integer)):
                return err("stft failed: noverlap must be an integer")
            selected_noverlap = int(noverlap)
            if not 0 <= selected_noverlap <= selected_nperseg - 1:
                return err(
                    "stft failed: noverlap must satisfy 0 <= noverlap <= nperseg - 1 "
                    f"(noverlap={selected_noverlap}, nperseg={selected_nperseg})"
                )

        # scipy validates the window specification (name or get_window arguments).
        try:
            sp_signal.get_window(window, selected_nperseg)
        except Exception as exc:
            return err(f"stft failed: invalid window {window!r}: {exc}")

        frequencies, times, zxx = sp_signal.stft(
            samples,
            fs=rate,
            window=window,
            nperseg=selected_nperseg,
            noverlap=selected_noverlap,
        )

        magnitude = np.abs(zxx)
        n_frequencies, n_times = magnitude.shape

        # Peaks are always taken from the full matrix, before any downsampling.
        flat_index = int(np.argmax(magnitude))
        peak_row, peak_column = np.unravel_index(flat_index, magnitude.shape)
        peak_frequency = float(frequencies[peak_row])
        peak_time = float(times[peak_column])

        row_indices = _evenly_spaced_indices(n_frequencies, _STFT_MAX_CELLS)
        column_indices = _evenly_spaced_indices(n_times, _STFT_MAX_CELLS)
        downsampled = (
            row_indices.size < n_frequencies or column_indices.size < n_times
        )

        path: str | None = None
        if downsampled:
            returned_magnitude = magnitude[np.ix_(row_indices, column_indices)]
            returned_frequencies = frequencies[row_indices]
            returned_times = times[column_indices]
            npz_path = _new_artifact_path("stft", ".npz")
            np.savez(
                npz_path,
                frequencies=frequencies,
                times=times,
                magnitude=magnitude,
            )
            path = str(npz_path.resolve())
        else:
            returned_magnitude = magnitude
            returned_frequencies = frequencies
            returned_times = times

        payload: dict[str, Any] = {
            "frequencies": returned_frequencies.tolist(),
            "times": returned_times.tolist(),
            "magnitude": returned_magnitude.tolist(),
            "shape": [int(returned_magnitude.shape[0]), int(returned_magnitude.shape[1])],
            "peak_frequency": peak_frequency,
            "peak_time": peak_time,
            "path": path,
            "downsampled": bool(downsampled),
            "spectrogram_path": None,
        }
        if downsampled:
            payload["original_shape"] = [int(n_frequencies), int(n_times)]

        if save_spectrogram:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_path = _resolve_spectrogram_path(filename)
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            mesh = ax.pcolormesh(times, frequencies, magnitude, shading="auto")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
            fig.colorbar(mesh, ax=ax, label="Magnitude")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            payload["spectrogram_path"] = str(plot_path.resolve())

        return ok(payload)
    except Exception as exc:
        return err(f"stft failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(filter_signal)
    mcp.tool()(convolve)
    mcp.tool()(power_spectrum)
    mcp.tool()(find_peaks)
    mcp.tool()(stft)
