"""Tests for the data file IO tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sci_mcp.tools.data_io import describe_file, load_data, plot_file, save_data


@pytest.fixture
def out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SCI_MCP_OUTPUT_DIR at a temporary directory and return it."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_save_load_csv_roundtrip(out_dir: Path) -> None:
    saved = save_data([1.5, -2.0, 3.25], "values.csv")
    assert saved["ok"] is True
    assert saved["result"]["format"] == "csv"
    assert saved["result"]["shape"] == [3]
    path = Path(saved["result"]["path"])
    assert path.exists()
    assert path.parent == out_dir

    loaded = load_data(str(path))
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["format"] == "csv"
    assert payload["shape"] == [3]
    assert payload["data"] == pytest.approx([1.5, -2.0, 3.25])
    assert payload["summary"]["n"] == 3
    assert payload["summary"]["mean"] == pytest.approx(0.9166666666666666)

    # Relative input paths resolve against SCI_MCP_OUTPUT_DIR.
    relative = load_data("values.csv")
    assert relative["ok"] is True
    assert relative["result"]["data"] == pytest.approx([1.5, -2.0, 3.25])


def test_save_load_npy_roundtrip(out_dir: Path) -> None:
    saved = save_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "matrix.npy")
    assert saved["ok"] is True
    assert saved["result"]["format"] == "npy"
    assert saved["result"]["shape"] == [3, 2]

    loaded = load_data(saved["result"]["path"])
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["format"] == "npy"
    assert payload["shape"] == [3, 2]
    assert payload["columns"] == [0, 1]
    assert payload["data"] == {"0": pytest.approx([1, 3, 5]), "1": pytest.approx([2, 4, 6])}


def test_save_load_json_dict_roundtrip(out_dir: Path) -> None:
    saved = save_data({"a": [1.0, 2.0], "b": [3.5, 4.5]}, "columns.json")
    assert saved["ok"] is True
    assert saved["result"]["format"] == "json"
    assert saved["result"]["shape"] == [2, 2]

    loaded = load_data(saved["result"]["path"])
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["columns"] == ["a", "b"]
    assert payload["shape"] == [2, 2]
    assert payload["data"]["a"] == pytest.approx([1.0, 2.0])
    assert payload["data"]["b"] == pytest.approx([3.5, 4.5])


# ---------------------------------------------------------------------------
# CSV parsing and column selection
# ---------------------------------------------------------------------------


def test_csv_multi_column_selection(out_dir: Path) -> None:
    saved = save_data(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], "table.csv", column_names=["x", "y"]
    )
    path = saved["result"]["path"]

    full = load_data(path)
    assert full["ok"] is True
    assert full["result"]["columns"] == ["x", "y"]
    assert full["result"]["shape"] == [3, 2]

    by_name = load_data(path, column="y")
    assert by_name["ok"] is True
    assert by_name["result"]["shape"] == [3]
    assert by_name["result"]["columns"] == ["y"]
    assert by_name["result"]["data"] == pytest.approx([10.0, 20.0, 30.0])

    by_index = load_data(path, column=1)
    assert by_index["ok"] is True
    assert by_index["result"]["data"] == pytest.approx([10.0, 20.0, 30.0])


def test_csv_header_detection(out_dir: Path) -> None:
    path = out_dir / "manual.csv"
    path.write_text("time,value\n0,1.0\n1,2.5\n", encoding="utf-8")

    loaded = load_data(str(path))
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["columns"] == ["time", "value"]
    assert payload["shape"] == [2, 2]
    assert payload["summary"]["value"]["n"] == 2
    assert payload["summary"]["value"]["mean"] == pytest.approx(1.75)


def test_headerless_csv_uses_indices(out_dir: Path) -> None:
    path = out_dir / "noheader.tsv"
    path.write_text("0\t5\n1\t6\n", encoding="utf-8")

    loaded = load_data(str(path))
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["columns"] == [0, 1]
    assert payload["format"] == "tsv"
    assert payload["data"]["0"] == pytest.approx([0.0, 1.0])


# ---------------------------------------------------------------------------
# Full-data versus preview behaviour
# ---------------------------------------------------------------------------


def test_small_data_returns_full_data(out_dir: Path) -> None:
    saved = save_data([4.0, 5.0, 6.0, 7.0], "small.csv")
    loaded = load_data(saved["result"]["path"], max_rows=20)
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["data"] == pytest.approx([4.0, 5.0, 6.0, 7.0])
    assert "hint" not in payload


def test_large_data_omits_full_data(out_dir: Path) -> None:
    values = [float(index) for index in range(100)]
    saved = save_data(values, "large.csv")

    loaded = load_data(saved["result"]["path"], max_rows=10)
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert "data" not in payload
    assert len(payload["preview"]) == 10
    assert payload["summary"]["n"] == 100
    assert payload["summary"]["mean"] == pytest.approx(49.5)
    assert payload["summary"]["min"] == pytest.approx(0.0)
    assert payload["summary"]["max"] == pytest.approx(99.0)
    assert "hint" in payload


def test_long_output_is_downsampled(out_dir: Path) -> None:
    values = [float(index) for index in range(2500)]
    saved = save_data(values, "long.csv")

    loaded = load_data(saved["result"]["path"], max_rows=5000)
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["downsampled"] is True
    assert payload["original_length"] == 2500
    assert len(payload["data"]) == 2000
    assert len(payload["preview"]) == 2000
    assert payload["data"][0] == pytest.approx(0.0)
    assert payload["data"][-1] == pytest.approx(2499.0)


# ---------------------------------------------------------------------------
# describe_file
# ---------------------------------------------------------------------------


def test_describe_file_known_mean(out_dir: Path) -> None:
    saved = save_data([1.0, 2.0, 3.0, 4.0], "stats.csv")
    described = describe_file(saved["result"]["path"])
    assert described["ok"] is True
    payload = described["result"]
    assert payload["n"] == 4
    assert payload["mean"] == pytest.approx(2.5)
    assert payload["std"] == pytest.approx(1.2909944487358056, abs=1e-9)
    assert payload["path"].endswith("stats.csv")
    assert payload["column"] == 0


def test_describe_file_multi_column_requires_selection(out_dir: Path) -> None:
    saved = save_data([[1.0, 2.0], [3.0, 4.0]], "multi.csv", column_names=["a", "b"])
    path = saved["result"]["path"]

    rejected = describe_file(path)
    assert rejected["ok"] is False
    assert "error" in rejected

    selected = describe_file(path, column="b")
    assert selected["ok"] is True
    assert selected["result"]["column"] == "b"
    assert selected["result"]["mean"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# plot_file
# ---------------------------------------------------------------------------


def test_plot_file_line_creates_png(out_dir: Path) -> None:
    saved = save_data(
        [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], "line.csv", column_names=["x", "y"]
    )
    rendered = plot_file(
        "line", saved["result"]["path"], x="x", y="y", title="line plot"
    )
    assert rendered["ok"] is True
    payload = rendered["result"]
    assert payload["kind"] == "line"
    assert payload["source_path"].endswith("line.csv")
    image = Path(payload["path"])
    assert image.exists()
    assert image.suffix == ".png"
    assert image.stat().st_size > 0


def test_plot_file_hist_creates_png(out_dir: Path) -> None:
    saved = save_data([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], "hist.csv")
    rendered = plot_file("hist", saved["result"]["path"])
    assert rendered["ok"] is True
    image = Path(rendered["result"]["path"])
    assert image.exists()
    assert image.stat().st_size > 0


def test_plot_file_requires_x_and_y(out_dir: Path) -> None:
    saved = save_data([[0.0, 1.0], [1.0, 2.0]], "xy.csv", column_names=["x", "y"])
    rendered = plot_file("line", saved["result"]["path"], x="x")
    assert rendered["ok"] is False
    assert "error" in rendered


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_load_missing_file(out_dir: Path) -> None:
    result = load_data(str(out_dir / "does_not_exist.csv"))
    assert result["ok"] is False
    assert "error" in result


def test_load_unsupported_format(out_dir: Path) -> None:
    path = out_dir / "data.xyz"
    path.write_text("1,2\n3,4\n", encoding="utf-8")
    result = load_data(str(path))
    assert result["ok"] is False
    assert "error" in result


def test_load_non_numeric_csv(out_dir: Path) -> None:
    path = out_dir / "bad.csv"
    path.write_text("x,y\n1,oops\n", encoding="utf-8")
    result = load_data(str(path))
    assert result["ok"] is False
    assert "error" in result


@pytest.mark.parametrize(
    "filename",
    ["../escape.csv", "sub/dir.csv", "C:escape.csv", "..\\escape.csv"],
)
def test_save_rejects_unsafe_filenames(out_dir: Path, filename: str) -> None:
    result = save_data([1.0, 2.0], filename)
    assert result["ok"] is False
    assert "error" in result


def test_save_rejects_unknown_format(out_dir: Path) -> None:
    result = save_data([1.0, 2.0], "values.xlsx", format="xlsx")
    assert result["ok"] is False
    assert "error" in result


def test_save_appends_extension_from_format(out_dir: Path) -> None:
    result = save_data({"v": [1.0, 2.0]}, "noext", format="json")
    assert result["ok"] is True
    path = Path(result["result"]["path"])
    assert path.name == "noext.json"
    assert path.exists()


# ---------------------------------------------------------------------------
# NPY edge cases
# ---------------------------------------------------------------------------


def test_load_scalar_npy(out_dir: Path) -> None:
    path = out_dir / "scalar.npy"
    np.save(path, np.asarray(3.5))

    loaded = load_data(str(path))
    assert loaded["ok"] is True
    payload = loaded["result"]
    assert payload["shape"] == []
    assert payload["data"] == pytest.approx(3.5)
    assert payload["summary"]["n"] == 1
