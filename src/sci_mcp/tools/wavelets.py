"""Wavelet tools: multilevel decomposition and wavelet shrinkage denoising."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pywt
from mcp.server.mcpserver import MCPServer

from ..core import err, load_array, ok, output_dir

# BRIEF section 11: never return more than 2000 numeric points in one array.
_MAX_POINTS = 2000

# Threshold modes accepted by pywt.threshold.
_THRESHOLD_MODES = ("soft", "hard", "garrote", "greater", "less")

_OUTPUT_COUNTER = 0


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
    step = (flat.size + _MAX_POINTS - 1) // _MAX_POINTS
    reduced = flat[::step]
    meta: dict[str, Any] = {
        "downsampled": True,
        "downsample_step": int(step),
    }
    return reduced.tolist(), meta


def _new_output_path(prefix: str, suffix: str) -> Path:
    """Allocate a unique output path inside output_dir() without overwriting files."""
    global _OUTPUT_COUNTER
    directory = output_dir()
    for _ in range(1000):
        _OUTPUT_COUNTER += 1
        candidate = directory / f"{prefix}_{_OUTPUT_COUNTER:06d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a unique output filename")


def _resolve_wavelet(wavelet: Any) -> pywt.Wavelet:
    """Validate a wavelet name and return the pywt.Wavelet instance."""
    if not isinstance(wavelet, str) or not wavelet.strip():
        raise ValueError("wavelet must be a non-empty wavelet name string")
    try:
        return pywt.Wavelet(wavelet.strip())
    except Exception as exc:
        raise ValueError(f"unknown wavelet {wavelet!r}: {exc}") from exc


def _resolve_level(level: Any, samples: np.ndarray, wavelet: pywt.Wavelet) -> int:
    """Validate an explicit decomposition level or derive the maximum one."""
    maximum = int(pywt.dwt_max_level(int(samples.size), wavelet.dec_len))
    if level is None:
        return maximum
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
        raise ValueError("level must be an integer")
    selected = int(level)
    if selected < 0:
        raise ValueError(f"level must be non-negative (got {selected})")
    if selected > maximum:
        raise ValueError(
            f"level {selected} is too high for a signal of length {samples.size} "
            f"with wavelet {wavelet.name!r}; the maximum level is {maximum}"
        )
    return selected


def _coefficient_descriptors(coeffs: list[np.ndarray], level: int) -> list[dict[str, Any]]:
    """Describe each coefficient array (approximation first, then details)."""
    descriptors: list[dict[str, Any]] = []
    for index, coeff in enumerate(coeffs):
        if index == 0:
            kind = "approximation"
            coefficient_level = level
        else:
            kind = "detail"
            coefficient_level = level - index + 1
        descriptors.append(
            {
                "level": int(coefficient_level),
                "kind": kind,
                "length": int(coeff.size),
                "energy": float(np.sum(np.square(coeff))),
                "preview": [float(value) for value in coeff[:10]],
            }
        )
    return descriptors


def wavelet_transform(
    signal: list[float] | str,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "symmetric",
) -> dict:
    """Decompose a signal with a discrete wavelet transform (pywt.wavedec).

    The maximum level is used when ``level`` is omitted. Full coefficient arrays
    are saved to a .npz file; the return value carries lengths, energies and a
    short preview of each level.
    """
    try:
        wave = _resolve_wavelet(wavelet)
        samples = _coerce_series(signal, "signal")
        selected_level = _resolve_level(level, samples, wave)

        coeffs = pywt.wavedec(
            samples, wave, mode=mode, level=selected_level
        )
        descriptors = _coefficient_descriptors(coeffs, selected_level)

        arrays: dict[str, np.ndarray] = {f"cA{selected_level}": coeffs[0]}
        for index in range(1, len(coeffs)):
            detail_level = selected_level - index + 1
            arrays[f"cD{detail_level}"] = coeffs[index]

        path = _new_output_path("wavelet_coeffs", ".npz")
        np.savez(path, **arrays)

        return ok(
            {
                "wavelet": wave.name,
                "level": int(selected_level),
                "mode": str(mode),
                "length": int(samples.size),
                "coefficients": descriptors,
                "path": str(path.resolve()),
            }
        )
    except Exception as exc:
        return err(f"wavelet_transform failed: {exc}")


def wavelet_denoise(
    signal: list[float] | str,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: float | None = None,
    mode: str = "soft",
) -> dict:
    """Denoise a signal with wavelet shrinkage.

    Detail coefficients are thresholded with :func:`pywt.threshold`. When
    ``threshold`` is omitted, the universal threshold ``sigma*sqrt(2*log(n))``
    is used with ``sigma`` estimated from the finest detail coefficients via MAD.
    """
    try:
        wave = _resolve_wavelet(wavelet)
        samples = _coerce_series(signal, "signal")
        selected_level = _resolve_level(level, samples, wave)

        if not isinstance(mode, str) or mode.strip().lower() not in _THRESHOLD_MODES:
            return err(
                f"wavelet_denoise failed: unknown threshold mode {mode!r}; "
                f"supported modes are {list(_THRESHOLD_MODES)}"
            )
        threshold_mode = mode.strip().lower()

        coeffs = pywt.wavedec(samples, wave, mode="symmetric", level=selected_level)

        if threshold is None:
            # Universal threshold with sigma estimated from the finest detail band.
            if selected_level > 0:
                finest = np.asarray(coeffs[-1], dtype=float)
                center = np.median(finest)
                sigma = float(np.median(np.abs(finest - center)) / 0.6744897501960817)
                threshold_value = sigma * float(
                    np.sqrt(2.0 * np.log(max(samples.size, 2)))
                )
            else:
                threshold_value = 0.0
        else:
            if isinstance(threshold, bool) or not isinstance(
                threshold, (int, float, np.integer, np.floating)
            ):
                return err("wavelet_denoise failed: threshold must be a non-negative number")
            threshold_value = float(threshold)
            if not np.isfinite(threshold_value) or threshold_value < 0:
                return err("wavelet_denoise failed: threshold must be finite and non-negative")

        thresholded = [coeffs[0]]
        for detail in coeffs[1:]:
            thresholded.append(
                pywt.threshold(np.asarray(detail, dtype=float), threshold_value, mode=threshold_mode)
            )

        reconstructed = pywt.waverec(thresholded, wave, mode="symmetric")
        reconstructed = np.asarray(reconstructed, dtype=float).reshape(-1)
        if reconstructed.size > samples.size:
            reconstructed = reconstructed[: samples.size]
        elif reconstructed.size < samples.size:
            reconstructed = np.pad(
                reconstructed, (0, samples.size - reconstructed.size), mode="constant"
            )

        denoised_payload, meta = _series_payload(reconstructed)
        payload: dict[str, Any] = {
            "denoised": denoised_payload,
            "threshold": float(threshold_value),
            "level": int(selected_level),
            "wavelet": wave.name,
            "mode": threshold_mode,
            "original_length": int(samples.size),
        }
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"wavelet_denoise failed: {exc}")


def register(mcp: MCPServer) -> None:
    mcp.tool()(wavelet_transform)
    mcp.tool()(wavelet_denoise)
