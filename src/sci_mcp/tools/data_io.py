"""Data file IO tools: load/save CSV, TSV, TXT, JSON and NPY, plus file-based describe/plot."""

from __future__ import annotations

import csv
import json
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import err, ok, output_dir
from .stats_plot import describe, plot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer


_MAX_FILE_BYTES = 50 * 1024 * 1024
_MAX_ROWS = 1_000_000
_MAX_POINTS = 2000
_SAVE_FORMATS = {"csv", "json", "npy"}
_PLOT_FILE_KINDS = {"line", "scatter", "hist", "bar", "errorbar"}

_SUFFIX_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "txt",
    ".json": "json",
    ".npy": "npy",
}

_FULL_DATA_HINT = (
    "full data not returned because the file has more rows than max_rows; "
    "pass column to select one column, or use another tool that accepts the file path"
)


# ---------------------------------------------------------------------------
# Path, format and numeric helpers
# ---------------------------------------------------------------------------


def _resolve_input_path(path: Any) -> Path:
    """Resolve an input path (relative paths are anchored at output_dir()) and check limits."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir() / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"not a regular file: {candidate}")
    size = candidate.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"file is too large ({size} bytes); the limit is {_MAX_FILE_BYTES} bytes"
        )
    return candidate


def _format_of(path: Path) -> str:
    """Return the canonical format name for a supported file extension."""
    suffix = path.suffix.lower()
    fmt = _SUFFIX_FORMATS.get(suffix)
    if fmt is None:
        raise ValueError(
            f"unsupported file format {suffix or '(no extension)'}; "
            "supported: .csv, .tsv, .txt, .json, .npy"
        )
    return fmt


def _resolve_output_path(filename: Any, fmt: str) -> Path:
    """Validate a bare output file name and return its path under output_dir()."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    name = filename.strip()
    if ".." in name:
        raise ValueError("filename must not contain '..'")
    if any(char in name for char in ("/", "\\", ":")):
        raise ValueError(
            "filename must be a bare file name without path separators or drive letters"
        )
    if Path(name).name != name or Path(name).drive:
        raise ValueError(
            "filename must be a bare file name without path separators or drive letters"
        )
    if not Path(name).suffix:
        name = f"{name}.{fmt}"
    path = output_dir() / name
    if path.exists() and path.is_dir():
        raise ValueError(f"output path is a directory: {path}")
    return path


def _detect_encoding(path: Path) -> str:
    """Pick a text encoding that can decode the first 64 KB of the file."""
    raw = path.open("rb").read(65536)
    for encoding in ("utf-8-sig", "gbk"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _detect_delimiter(sample: str, fmt: str) -> str | None:
    """Return the column delimiter for a text table, or None for whitespace splitting."""
    if fmt == "tsv":
        return "\t"
    if fmt == "csv":
        return ","
    first_line = next((line for line in sample.splitlines() if line.strip()), "")
    for delimiter in ("\t", ",", ";", "|"):
        if delimiter in first_line:
            return delimiter
    return None


def _is_numeric_text(text: str) -> bool:
    """Return True if the cell text parses as a float."""
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _unique_names(cells: list[str]) -> list[str]:
    """Make header names non-empty and unique while preserving order."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, cell in enumerate(cells):
        base = cell.strip() or f"col{index}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}_{count}")
    return names


# ---------------------------------------------------------------------------
# Readers: every reader returns (array, columns)
# ---------------------------------------------------------------------------


def _read_delimited(path: Path, fmt: str) -> tuple[np.ndarray, list[Any]]:
    """Read a CSV/TSV/TXT numeric table, detecting the header row."""
    encoding = _detect_encoding(path)
    with path.open("r", encoding=encoding, newline="") as handle:
        sample = handle.read(65536)

    delimiter = _detect_delimiter(sample, fmt)
    header: list[str] | None = None
    ncols: int | None = None
    first_row = True
    rows: list[list[float]] = []

    with path.open("r", encoding=encoding, newline="") as handle:
        if delimiter is None:
            raw_rows: Any = (line.split() for line in handle)
        else:
            raw_rows = csv.reader(handle, delimiter=delimiter)

        for raw in raw_rows:
            cells = [str(cell).strip() for cell in raw]
            if not any(cells):
                continue

            if first_row:
                first_row = False
                if all(_is_numeric_text(cell) for cell in cells):
                    ncols = len(cells)
                else:
                    header = _unique_names(cells)
                    ncols = len(header)
                    continue

            if ncols is None:
                ncols = len(cells)
            if len(cells) != ncols:
                raise ValueError(
                    f"row {len(rows) + 1} has {len(cells)} columns; expected {ncols}"
                )
            if not all(_is_numeric_text(cell) for cell in cells):
                raise ValueError(f"non-numeric value in data row {len(rows) + 1}: {cells}")
            rows.append([float(cell) for cell in cells])
            if len(rows) > _MAX_ROWS:
                raise ValueError(f"file has more than {_MAX_ROWS} rows")

    if not rows:
        raise ValueError("no numeric data rows found")

    array = np.asarray(rows, dtype=float)
    if array.shape[1] == 1:
        columns: list[Any] = [header[0]] if header else [0]
        return array[:, 0], columns
    columns = list(header) if header else list(range(array.shape[1]))
    return array, columns


def _read_json(path: Path) -> tuple[np.ndarray, list[Any]]:
    """Read a JSON file shaped as list[float], list[list[float]] or {name: list}."""
    with path.open("r", encoding=_detect_encoding(path)) as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if not payload:
            raise ValueError("JSON object has no columns")
        names = [str(key) for key in payload.keys()]
        columns_values: list[list[float]] = []
        for name, value in payload.items():
            if not isinstance(value, list):
                raise ValueError(f"JSON column {name!r} must be a list")
            if not value:
                raise ValueError(f"JSON column {name!r} is empty")
            if any(isinstance(item, (list, dict)) for item in value):
                raise ValueError(f"JSON column {name!r} must contain plain numbers")
            try:
                columns_values.append([float(item) for item in value])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"JSON column {name!r} contains non-numeric values"
                ) from exc
        lengths = {len(values) for values in columns_values}
        if len(lengths) != 1:
            raise ValueError("JSON columns must have equal lengths")
        n_rows = lengths.pop()
        if n_rows > _MAX_ROWS:
            raise ValueError(f"file has more than {_MAX_ROWS} rows")
        array = np.asarray(columns_values, dtype=float).T
        if array.shape[1] == 1:
            return array[:, 0], [names[0]]
        return array, names

    if isinstance(payload, list):
        if not payload:
            raise ValueError("JSON array is empty")
        if len(payload) > _MAX_ROWS:
            raise ValueError(f"file has more than {_MAX_ROWS} rows")

        if all(isinstance(item, list) for item in payload):
            width = len(payload[0])
            if width == 0:
                raise ValueError("JSON rows must not be empty")
            if any(len(row) != width for row in payload):
                raise ValueError("JSON rows must all have the same length")
            try:
                array = np.asarray(payload, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("JSON rows contain non-numeric values") from exc
            if array.ndim != 2:
                raise ValueError("JSON data must be numeric")
            if array.shape[1] == 1:
                return array[:, 0], [0]
            return array, list(range(array.shape[1]))

        try:
            values = [float(item) for item in payload]
        except (TypeError, ValueError) as exc:
            raise ValueError("JSON array contains non-numeric values") from exc
        return np.asarray(values, dtype=float), [0]

    raise ValueError("JSON must be a list or an object of named columns")


def _read_npy(path: Path) -> tuple[np.ndarray, list[Any]]:
    """Read a numeric .npy array of 0, 1 or 2 dimensions."""
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if array.ndim > 2:
        raise ValueError(f"NPY arrays with {array.ndim} dimensions are not supported")
    if array.size == 0:
        raise ValueError("NPY array is empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"NPY dtype {array.dtype} is not numeric")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("complex NPY arrays are not supported")
    if array.ndim >= 1 and array.shape[0] > _MAX_ROWS:
        raise ValueError(f"file has more than {_MAX_ROWS} rows")

    array = array.astype(float, copy=False)
    if array.ndim == 2:
        if array.shape[1] == 1:
            return array[:, 0], [0]
        return array, list(range(array.shape[1]))
    if array.ndim == 1:
        return array, [0]
    return array, []


def _read_file(path: Path) -> tuple[np.ndarray, list[Any]]:
    """Dispatch to the reader matching the file extension."""
    fmt = _format_of(path)
    if fmt in {"csv", "tsv", "txt"}:
        return _read_delimited(path, fmt)
    if fmt == "json":
        return _read_json(path)
    return _read_npy(path)


# ---------------------------------------------------------------------------
# Column selection, summaries and downsampling
# ---------------------------------------------------------------------------


def _resolve_column_index(columns: list[Any], column: Any) -> int:
    """Resolve a column name or integer index to a positional index."""
    if isinstance(column, bool):
        raise ValueError("column must be a column name or index")
    if isinstance(column, numbers.Integral):
        index = int(column)
        if index < 0 or index >= len(columns):
            raise ValueError(
                f"column index {index} out of range (0..{max(len(columns) - 1, 0)})"
            )
        return index
    if isinstance(column, str):
        wanted = column.strip()
        for index, name in enumerate(columns):
            if isinstance(name, str) and name == wanted:
                return index
        for index, name in enumerate(columns):
            if str(name) == wanted:
                return index
        if wanted.lstrip("+-").isdigit():
            index = int(wanted)
            if 0 <= index < len(columns):
                return index
        raise ValueError(f"unknown column {column!r}; available columns: {columns}")
    raise ValueError("column must be a column name or index")


def _select_column(
    array: np.ndarray, columns: list[Any], column: Any
) -> tuple[np.ndarray, Any]:
    """Return a one-dimensional column and its label from a 1-D or 2-D array."""
    if array.ndim == 0:
        raise ValueError("cannot select a column from a scalar array")
    if array.ndim == 1:
        if column is None:
            return array, columns[0] if columns else 0
        index = _resolve_column_index(columns, column)
        if index != 0:
            raise ValueError(f"column index {index} out of range for one-dimensional data")
        return array, columns[0] if columns else 0
    if array.ndim != 2:
        raise ValueError("column selection requires one- or two-dimensional data")
    if column is None:
        if array.shape[1] == 1:
            return array[:, 0], columns[0] if columns else 0
        raise ValueError("column is required for multi-column data")
    index = _resolve_column_index(columns, column)
    return array[:, index], columns[index]


def _summary(array: np.ndarray) -> dict[str, Any]:
    """Return n/mean/std/min/max for a numeric array."""
    values = np.asarray(array, dtype=float).ravel()
    n = int(values.size)
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _downsample(items: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Limit a list to 2000 evenly spaced points, reporting the original length."""
    n = len(items)
    if n <= _MAX_POINTS:
        return list(items), {}
    indices = np.linspace(0, n - 1, _MAX_POINTS).round().astype(np.int64)
    sampled = [items[int(index)] for index in indices]
    return sampled, {"downsampled": True, "original_length": n}


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def load_data(
    path: str, column: str | int | None = None, max_rows: int = 20
) -> dict:
    """Load a CSV/TSV/TXT/JSON/NPY file and return its shape, preview, summary and data."""
    try:
        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, numbers.Integral)
            or int(max_rows) < 1
        ):
            return err("load_data failed: max_rows must be a positive integer")
        max_rows = int(max_rows)

        resolved = _resolve_input_path(path)
        fmt = _format_of(resolved)
        array, columns = _read_file(resolved)

        if array.ndim == 0:
            value = float(array)
            return ok(
                {
                    "path": str(resolved),
                    "format": fmt,
                    "shape": [],
                    "columns": [],
                    "preview": [value],
                    "summary": _summary(array.reshape(1)),
                    "data": value,
                }
            )

        n_rows = int(array.shape[0])
        payload: dict[str, Any] = {
            "path": str(resolved),
            "format": fmt,
            "shape": [int(dim) for dim in array.shape],
            "columns": list(columns),
        }
        meta: dict[str, Any] = {}

        if column is not None:
            selected, label = _select_column(array, columns, column)
            values = selected.tolist()
            preview, preview_meta = _downsample(values[:max_rows])
            payload["shape"] = [n_rows]
            payload["columns"] = [label]
            payload["column"] = label
            payload["preview"] = preview
            payload["summary"] = _summary(selected)
            meta.update(preview_meta)
            if n_rows <= max_rows:
                data, data_meta = _downsample(values)
                payload["data"] = data
                meta.update(data_meta)
            else:
                payload["hint"] = _FULL_DATA_HINT
            payload.update(meta)
            return ok(payload)

        if array.ndim == 1:
            values = array.tolist()
            payload["columns"] = [columns[0]] if columns else [0]
            payload["shape"] = [n_rows]
            preview, preview_meta = _downsample(values[:max_rows])
            payload["preview"] = preview
            payload["summary"] = _summary(array)
            meta.update(preview_meta)
            if n_rows <= max_rows:
                data, data_meta = _downsample(values)
                payload["data"] = data
                meta.update(data_meta)
            else:
                payload["hint"] = _FULL_DATA_HINT
            payload.update(meta)
            return ok(payload)

        # Multi-column table: no single column requested.
        labels = [str(name) for name in columns]
        preview, preview_meta = _downsample(array[:max_rows].tolist())
        payload["preview"] = preview
        payload["summary"] = {
            label: _summary(array[:, index]) for index, label in enumerate(labels)
        }
        meta.update(preview_meta)
        if n_rows <= max_rows:
            data: dict[str, Any] = {}
            for index, label in enumerate(labels):
                values, data_meta = _downsample(array[:, index].tolist())
                data[label] = values
                meta.update(data_meta)
            payload["data"] = data
        else:
            payload["hint"] = _FULL_DATA_HINT
        payload.update(meta)
        return ok(payload)
    except Exception as exc:
        return err(f"load_data failed: {exc}")


def _to_1d_floats(values: Any, name: str) -> list[float]:
    """Convert a JSON-style sequence to a non-empty list of floats."""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of numbers")
    converted: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise ValueError(f"{name} contains a non-numeric value: {item!r}")
        converted.append(float(item))
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


def _normalize_save_data(
    data: Any, column_names: Any
) -> tuple[np.ndarray, list[str] | None]:
    """Normalize accepted save_data payloads to a 1-D/2-D float array and header names."""
    names: list[str] | None = None
    if column_names is not None:
        if not isinstance(column_names, (list, tuple)) or not column_names:
            raise ValueError("column_names must be a non-empty list of names")
        names = [str(item) for item in column_names]
        if len(set(names)) != len(names):
            raise ValueError("column_names must be unique")

    if isinstance(data, dict):
        if not data:
            raise ValueError("data must not be empty")
        columns: list[str] = []
        columns_values: list[list[float]] = []
        for key, value in data.items():
            columns.append(str(key))
            columns_values.append(_to_1d_floats(value, f"data[{key!r}]"))
        lengths = {len(values) for values in columns_values}
        if len(lengths) != 1:
            raise ValueError("all columns must have the same length")
        array = np.asarray(columns_values, dtype=float).T
        if names is not None:
            if len(names) != array.shape[1]:
                raise ValueError(
                    f"column_names has {len(names)} names but data has {array.shape[1]} columns"
                )
            columns = names
        return array, columns

    if isinstance(data, np.ndarray):
        data = data.tolist()
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("data must be a non-empty list or dict of columns")

    first = data[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        width = len(first)
        if width == 0:
            raise ValueError("data rows must not be empty")
        rows: list[list[float]] = []
        for index, row in enumerate(data):
            values = _to_1d_floats(row, f"data[{index}]")
            if len(values) != width:
                raise ValueError("data rows must all have the same length")
            rows.append(values)
        array = np.asarray(rows, dtype=float)
        if names is not None and len(names) != width:
            raise ValueError(
                f"column_names has {len(names)} names but data has {width} columns"
            )
        return array, names

    values = _to_1d_floats(data, "data")
    if names is not None and len(names) != 1:
        raise ValueError("column_names must contain exactly one name for one-dimensional data")
    return np.asarray(values, dtype=float), names


def save_data(
    data: list[float] | list[list[float]] | dict[str, list[float]],
    filename: str,
    format: str = "csv",
    column_names: list[str] | None = None,
) -> dict:
    """Save numeric data as CSV, JSON or NPY under the configured output directory."""
    try:
        fmt = str(format).strip().lower()
        if fmt not in _SAVE_FORMATS:
            return err(
                f"save_data failed: unsupported format {format!r}; use csv, json or npy"
            )
        # A supported extension is authoritative so that a file named ``x.npy``
        # never receives CSV bytes (and vice versa).
        if isinstance(filename, str):
            suffix = Path(filename.strip()).suffix.lower().lstrip(".")
            if suffix in _SAVE_FORMATS:
                fmt = suffix
        resolved = _resolve_output_path(filename, fmt)
        array, columns = _normalize_save_data(data, column_names)

        if fmt == "csv":
            with resolved.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if columns is not None:
                    writer.writerow(columns)
                if array.ndim == 1:
                    for value in array:
                        writer.writerow([float(value)])
                else:
                    for row in array:
                        writer.writerow([float(value) for value in row])
        elif fmt == "json":
            if isinstance(data, dict):
                payload = {
                    columns[index]: [float(value) for value in array[:, index]]
                    for index in range(array.shape[1])
                }
            elif array.ndim == 1:
                payload = [float(value) for value in array]
            else:
                payload = [[float(value) for value in row] for row in array]
            with resolved.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        else:
            np.save(resolved, array)

        return ok(
            {
                "path": str(resolved.resolve()),
                "format": fmt,
                "shape": [int(dim) for dim in array.shape],
            }
        )
    except Exception as exc:
        return err(f"save_data failed: {exc}")


def describe_file(path: str, column: str | int | None = None) -> dict:
    """Describe one numeric column of a CSV/TSV/TXT/JSON/NPY file."""
    try:
        resolved = _resolve_input_path(path)
        array, columns = _read_file(resolved)
        if array.ndim == 0:
            return err("describe_file failed: cannot describe a scalar NPY array")
        if array.ndim == 2 and array.shape[1] > 1 and column is None:
            return err(
                "describe_file failed: file has multiple columns; specify column by name or index"
            )
        selected, label = _select_column(array, columns, column)
        described = describe(selected.tolist())
        if not described.get("ok"):
            return described
        payload = dict(described["result"])
        payload["path"] = str(resolved)
        payload["column"] = label
        return ok(payload)
    except Exception as exc:
        return err(f"describe_file failed: {exc}")


def _first_matching_column(columns: list[Any], candidates: tuple[str, ...]) -> Any | None:
    """Return the first existing column whose normalized name is in candidates."""
    for candidate in candidates:
        for name in columns:
            if str(name).strip().lower() == candidate:
                return name
    return None


def plot_file(
    kind: str,
    path: str,
    x: str | int | None = None,
    y: str | int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    filename: str | None = None,
    dpi: int = 150,
) -> dict:
    """Plot columns from a CSV/TSV/TXT/JSON/NPY file to a PNG in the output directory."""
    try:
        plot_kind = str(kind).strip().lower() if isinstance(kind, str) else ""
        if plot_kind not in _PLOT_FILE_KINDS:
            return err(
                f"plot_file failed: unsupported plot kind {kind!r}; "
                "use line, scatter, hist, bar or errorbar"
            )

        resolved = _resolve_input_path(path)
        array, columns = _read_file(resolved)
        if array.ndim == 0:
            return err("plot_file failed: cannot plot a scalar NPY array")

        data: dict[str, Any]
        if plot_kind in {"line", "scatter", "errorbar"}:
            if x is None or y is None:
                return err(f"plot_file failed: {plot_kind} requires both x and y columns")
            x_values, _ = _select_column(array, columns, x)
            y_values, _ = _select_column(array, columns, y)
            data = {"x": x_values.tolist(), "y": y_values.tolist()}
            if plot_kind == "errorbar":
                error_column = _first_matching_column(columns, ("yerr", "err", "error"))
                if error_column is not None:
                    error_values, _ = _select_column(array, columns, error_column)
                    data["yerr"] = error_values.tolist()
        elif plot_kind == "hist":
            if y is not None:
                values, _ = _select_column(array, columns, y)
            elif x is not None:
                values, _ = _select_column(array, columns, x)
            else:
                values, _ = _select_column(array, columns, None)
            data = {"values": values.tolist()}
        else:  # bar
            if x is not None and y is not None:
                x_values, _ = _select_column(array, columns, x)
                y_values, _ = _select_column(array, columns, y)
                data = {"x": x_values.tolist(), "y": y_values.tolist()}
            elif y is not None:
                values, _ = _select_column(array, columns, y)
                data = {"values": values.tolist()}
            elif x is not None:
                values, _ = _select_column(array, columns, x)
                data = {"values": values.tolist()}
            else:
                values, _ = _select_column(array, columns, None)
                data = {"values": values.tolist()}

        rendered = plot(
            plot_kind,
            data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            filename=filename,
            dpi=dpi,
        )
        if not rendered.get("ok"):
            return rendered
        payload = dict(rendered["result"])
        payload["source_path"] = str(resolved)
        return ok(payload)
    except Exception as exc:
        return err(f"plot_file failed: {exc}")


def register(mcp: "MCPServer") -> None:
    """Register the data IO tools on the MCP server."""
    mcp.tool()(load_data)
    mcp.tool()(save_data)
    mcp.tool()(describe_file)
    mcp.tool()(plot_file)
