"""Shared helpers for sci-mcp: output directory, JSON-safe conversion, result envelopes."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:  # sympy is a project dependency, but core should stay importable without it
    from sympy import Basic as _SympyBasic
    from sympy import Float as _SympyFloat
    from sympy import Integer as _SympyInteger
    from sympy import Rational as _SympyRational
except Exception:  # pragma: no cover - defensive only
    _SympyBasic = None  # type: ignore[assignment]
    _SympyFloat = None  # type: ignore[assignment]
    _SympyInteger = None  # type: ignore[assignment]
    _SympyRational = None  # type: ignore[assignment]

_DEFAULT_OUTPUT_DIR = Path.home() / ".sci-mcp" / "outputs"

# Maximum size of a file referenced through load_array's "@<path>" syntax.
_MAX_ARRAY_FILE_BYTES = 50 * 1024 * 1024


def output_dir() -> Path:
    """Return the output directory, reading SCI_MCP_OUTPUT_DIR on every call."""
    raw = os.environ.get("SCI_MCP_OUTPUT_DIR")
    path = Path(raw).expanduser() if raw and raw.strip() else _DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/sympy/complex/pathlib values into JSON-safe objects."""
    # numpy scalars (np.bool_, np.float64, np.int64, np.complex128, ...)
    # Must be checked before the generic Python types: np.float64 is a float
    # subclass and np.bool_ is not a bool, so isinstance checks on builtins lie.
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return to_jsonable(obj.item())
        return [to_jsonable(item) for item in obj.tolist()]

    # bool is a subclass of int; leave booleans untouched before any numeric handling.
    if isinstance(obj, bool):
        return obj

    if isinstance(obj, float):
        # Defensive: standard JSON has no NaN/Infinity literals. ok() records the
        # conversion in its warnings; direct to_jsonable callers simply get null.
        return obj if math.isfinite(obj) else None

    if isinstance(obj, complex):
        return {"real": to_jsonable(obj.real), "imag": to_jsonable(obj.imag)}

    if isinstance(obj, Path):
        return str(obj)

    if _SympyBasic is not None and isinstance(obj, _SympyBasic):
        # Check concrete numeric types before the generic str() fallback so that
        # callers can keep computing with the values. Integer must precede
        # Rational because sympy.Integer is a Rational subclass.
        if _SympyInteger is not None and isinstance(obj, _SympyInteger):
            return int(obj)
        if _SympyFloat is not None and isinstance(obj, _SympyFloat):
            return to_jsonable(float(obj))
        if _SympyRational is not None and isinstance(obj, _SympyRational):
            return float(obj)
        return str(obj)

    if isinstance(obj, dict):
        converted: dict[Any, Any] = {}
        for key, value in obj.items():
            new_key = to_jsonable(key)
            if not isinstance(new_key, (str, int, float, bool, type(None))):
                new_key = str(new_key)
            converted[new_key] = to_jsonable(value)
        return converted

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]

    # Anything already serializable passes through unchanged; the rest falls back to str.
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)
    return obj


def _nonfinite_label(value: float) -> str:
    """Return the human-facing JSON name for a non-finite float."""
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _sanitize(value: Any, path: str) -> tuple[Any, list[str]]:
    """Recursively convert a value to JSON-safe data, recording non-finite floats.

    Returns ``(cleaned, warnings)``. Every ``NaN``/``Infinity``/``-Infinity``
    found under ``path`` becomes ``None`` and produces one warning such as
    ``"result.t_values.x0: Infinity converted to null"``.
    """
    warnings: list[str] = []

    # numpy scalars: float64 is a float subclass, so unwrap before isinstance.
    if isinstance(value, np.generic):
        return _sanitize(value.item(), path)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _sanitize(value.item(), path)
        return _sanitize(value.tolist(), path)

    # bool is an int subclass; never treat booleans as numeric values.
    if isinstance(value, bool):
        return value, warnings

    if isinstance(value, float):
        if math.isfinite(value):
            return value, warnings
        warnings.append(f"{path}: {_nonfinite_label(value)} converted to null")
        return None, warnings

    if isinstance(value, complex):
        real, real_warnings = _sanitize(value.real, f"{path}.real")
        imag, imag_warnings = _sanitize(value.imag, f"{path}.imag")
        return {"real": real, "imag": imag}, real_warnings + imag_warnings

    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            new_key = to_jsonable(key)
            if not isinstance(new_key, (str, int, float, bool, type(None))):
                new_key = str(new_key)
            child_path = f"{path}.{new_key}" if path else str(new_key)
            cleaned, item_warnings = _sanitize(item, child_path)
            converted[new_key] = cleaned
            warnings.extend(item_warnings)
        return converted, warnings

    if isinstance(value, (list, tuple)):
        converted_items = []
        for index, item in enumerate(value):
            cleaned, item_warnings = _sanitize(item, f"{path}[{index}]")
            converted_items.append(cleaned)
            warnings.extend(item_warnings)
        return converted_items, warnings

    if _SympyBasic is not None and isinstance(value, _SympyBasic):
        # Keep to_jsonable's sympy conversions (int/float/str), but record
        # non-finite concrete numbers (e.g. sympy.Float("nan") or sympy.oo).
        if getattr(value, "is_number", False):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = None
            if number is not None and not math.isfinite(number):
                warnings.append(
                    f"{path}: {_nonfinite_label(number)} converted to null"
                )
                return None, warnings
        return to_jsonable(value), warnings

    return to_jsonable(value), warnings


def ok(value: Any = None, **meta: Any) -> dict:
    """Build a success envelope: {"ok": True, "result": ..., **meta}.

    Non-finite floats in ``value`` are recursively converted to ``None`` and
    reported in a top-level ``"warnings"`` list. Warnings supplied through
    ``meta`` are preserved and extended rather than overwritten.
    """
    cleaned, warnings = _sanitize(value, "result")
    envelope: dict[str, Any] = {"ok": True, "result": cleaned, **meta}
    if warnings:
        existing = envelope.get("warnings")
        if existing is None:
            envelope["warnings"] = warnings
        elif isinstance(existing, (list, tuple)):
            envelope["warnings"] = list(existing) + warnings
        else:
            envelope["warnings"] = [existing] + warnings
    return envelope


def _array_from_text(path: Path, dtype: Any) -> np.ndarray:
    """Parse numeric rows from a CSV/TSV/TXT file, skipping non-numeric rows."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # Comma-delimited files yield separate cells; tabs/spaces inside a cell
        # are split below, so all three separators work with one code path.
        for record in csv.reader(handle):
            tokens: list[str] = []
            for cell in record:
                tokens.extend(cell.split())
            if not tokens:
                continue
            try:
                numbers = [float(token) for token in tokens]
            except ValueError:
                # Header or other non-numeric row.
                continue
            rows.append(numbers)

    if not rows:
        raise ValueError(f"load_array: no numeric data found in {path}")

    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"load_array: inconsistent row lengths in {path}")

    array = np.asarray(rows, dtype=dtype)
    if array.ndim == 2 and array.shape[1] == 1:
        return array.reshape(-1)
    return array


def _array_from_json(payload: Any, path: Path, dtype: Any) -> np.ndarray:
    """Convert a JSON list or {column: list} mapping into a numeric array."""
    if isinstance(payload, list):
        try:
            array = np.asarray(payload, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"load_array: JSON in {path} is not a numeric array"
            ) from exc
        if array.ndim in (1, 2):
            return array
        raise ValueError(
            f"load_array: JSON array in {path} must be one- or two-dimensional"
        )

    if isinstance(payload, dict):
        if not payload:
            raise ValueError(f"load_array: JSON object in {path} is empty")
        columns: list[np.ndarray] = []
        for name, values in payload.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(
                    f"load_array: JSON key {name!r} in {path} must contain a list"
                )
            try:
                column = np.asarray(values, dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"load_array: JSON key {name!r} in {path} is not numeric"
                ) from exc
            if column.ndim != 1:
                raise ValueError(
                    f"load_array: JSON key {name!r} in {path} must be one-dimensional"
                )
            columns.append(column)

        lengths = {column.size for column in columns}
        if len(lengths) != 1:
            raise ValueError(
                f"load_array: JSON columns in {path} have different lengths"
            )
        return np.column_stack(columns)

    raise ValueError(f"load_array: unsupported JSON structure in {path}")


def _load_array_file(raw_path: str, dtype: Any) -> np.ndarray:
    """Resolve and read an ``@<path>`` array file reference."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = output_dir() / path

    if not path.is_file():
        raise FileNotFoundError(f"load_array: file not found: {path}")
    if path.stat().st_size > _MAX_ARRAY_FILE_BYTES:
        raise ValueError(f"load_array: file exceeds the 50 MB limit: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return np.asarray(array, dtype=dtype)

    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        return _array_from_json(payload, path, dtype)

    if suffix in {".csv", ".tsv", ".txt"}:
        return _array_from_text(path, dtype)

    raise ValueError(f"load_array: unsupported file extension {suffix!r}")


def load_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Load a numeric array from inline data or an ``@<path>`` file reference.

    Inline values may be numpy arrays, lists or tuples. File references accept
    an absolute path or one relative to :func:`output_dir`, with ``.npy``,
    ``.json``, ``.csv``, ``.tsv`` or ``.txt`` input. Failures raise; tool
    wrappers are expected to translate them into ``err(...)`` envelopes.
    """
    if isinstance(value, (np.ndarray, list, tuple)):
        return np.asarray(value, dtype=dtype)

    if isinstance(value, str):
        if not value.startswith("@"):
            raise ValueError(
                "load_array: string values must be file references starting with '@'"
            )
        raw_path = value[1:]
        if not raw_path:
            raise ValueError("load_array: '@' reference is missing a path")
        return _load_array_file(raw_path, dtype)

    raise ValueError(
        f"load_array: unsupported value type {type(value).__name__!r}"
    )


def err(message: str, /, **meta: Any) -> dict:
    """Build a failure envelope: {"ok": False, "error": str(message), **meta}."""
    return {"ok": False, "error": str(message), **meta}
