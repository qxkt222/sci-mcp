"""Tests for the run_python sandbox tool and its registration."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sci_mcp.tools.sandbox import run_python


def test_print_and_numpy_available(sci_output_dir: Path) -> None:
    result = run_python("print(2 + 3)\nprint(int(np.array([1, 2, 3]).sum()))")
    assert result["ok"] is True
    payload = result["result"]
    assert payload["returncode"] == 0
    assert "5" in payload["stdout"]
    assert "6" in payload["stdout"]
    assert payload["stderr"] == ""
    assert payload["files"] == []


def test_user_exception_is_reported_but_tool_succeeds(sci_output_dir: Path) -> None:
    result = run_python('raise ValueError("boom")')
    assert result["ok"] is True
    payload = result["result"]
    assert payload["returncode"] != 0
    assert "ValueError" in payload["stderr"]
    assert "boom" in payload["stderr"]


def test_saved_figure_is_listed_in_files(sci_output_dir: Path) -> None:
    code = (
        "plt.plot([0, 1, 2], [0, 1, 4])\n"
        'plt.savefig("sandbox_plot.png")\n'
        'print("saved")\n'
    )
    result = run_python(code)
    assert result["ok"] is True
    payload = result["result"]
    assert payload["returncode"] == 0
    assert "saved" in payload["stdout"]

    files = payload["files"]
    pngs = [f for f in files if f.endswith("sandbox_plot.png")]
    assert len(pngs) == 1
    assert Path(pngs[0]).is_file()
    assert Path(pngs[0]).parent == sci_output_dir.resolve()
    assert len(payload["files"]) == 1


def test_timeout_returns_error(sci_output_dir: Path) -> None:
    result = run_python("import time\ntime.sleep(30)", timeout_seconds=1)
    assert result["ok"] is False
    assert "error" in result
    assert "timeout" in result["error"].lower()


@pytest.mark.parametrize("bad_timeout", [0, -5, 301, 1000])
def test_timeout_out_of_range_returns_error(sci_output_dir: Path, bad_timeout: int) -> None:
    result = run_python("print(1)", timeout_seconds=bad_timeout)
    assert result["ok"] is False
    assert "timeout_seconds" in result["error"]


def test_sympy_available_in_subprocess(sci_output_dir: Path) -> None:
    code = 'x = sp.Symbol("x")\nprint(sp.diff(x**3, x))\nprint(sp.integrate(x, x))\n'
    result = run_python(code)
    assert result["ok"] is True
    payload = result["result"]
    assert payload["returncode"] == 0
    assert "3*x**2" in payload["stdout"]
    assert "x**2/2" in payload["stdout"]


def test_long_output_is_truncated(sci_output_dir: Path) -> None:
    result = run_python('print("x" * 25000)')
    assert result["ok"] is True
    stdout = result["result"]["stdout"]
    assert stdout.endswith("...[truncated]")
    assert len(stdout) == 20000 + len("...[truncated]")


def test_output_goes_to_configured_directory(sci_output_dir: Path) -> None:
    result = run_python('f = open("note.txt", "w", encoding="utf-8")\nf.write("hi")\nf.close()')
    assert result["ok"] is True
    files = result["result"]["files"]
    assert len(files) == 1
    assert files[0].endswith("note.txt")
    assert (sci_output_dir / "note.txt").read_text(encoding="utf-8") == "hi"


def test_server_registers_run_python() -> None:
    from sci_mcp.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}
    assert "run_python" in names


def test_core_helpers_contract() -> None:
    import numpy as np

    from sci_mcp.core import err, ok, to_jsonable

    converted = to_jsonable(
        {
            "scalar": np.float64(1.5),
            "flag": np.bool_(True),
            "int": np.int64(3),
            "array": np.arange(3),
            "complex": 1 + 2j,
            "path": Path("a/b.txt"),
        }
    )
    assert converted["scalar"] == 1.5
    assert converted["flag"] is True
    assert converted["int"] == 3
    assert converted["array"] == [0, 1, 2]
    assert converted["complex"] == {"real": 1.0, "imag": 2.0}
    assert converted["path"] == str(Path("a/b.txt"))

    assert ok(3)["result"] == 3
    assert err("nope") == {"ok": False, "error": "nope"}


def test_core_err_allows_message_meta_keyword() -> None:
    from sci_mcp.core import err

    # ``message`` is positional-only, so it can also be a metadata key.
    assert err("boom", message="detail") == {
        "ok": False,
        "error": "boom",
        "message": "detail",
    }


def test_core_to_jsonable_converts_sympy_numbers_and_symbols() -> None:
    import json

    import sympy as sp

    from sci_mcp.core import to_jsonable

    integer = to_jsonable(sp.Integer(2))
    assert integer == 2
    assert type(integer) is int

    floating = to_jsonable(sp.Float("2.5"))
    assert floating == 2.5
    assert type(floating) is float

    rational = to_jsonable(sp.Rational(1, 3))
    assert isinstance(rational, float)
    assert rational == pytest.approx(1.0 / 3.0)

    assert to_jsonable(sp.Symbol("x")) == "x"

    payload = {
        "integer": sp.Integer(2),
        "float": sp.Float("2.5"),
        "rational": sp.Rational(1, 3),
        "symbol": sp.Symbol("x"),
    }
    assert json.dumps(to_jsonable(payload))


# ---------------------------------------------------------------------------
# INF: core.load_array file/inline input and ok() non-finite sanitisation.
# ---------------------------------------------------------------------------


def test_load_array_accepts_inline_list() -> None:
    import numpy as np

    from sci_mcp.core import load_array

    array = load_array([1, 2, 3])
    assert isinstance(array, np.ndarray)
    assert np.issubdtype(array.dtype, np.floating)
    assert array.tolist() == [1.0, 2.0, 3.0]


def test_load_array_accepts_numpy_input() -> None:
    import numpy as np

    from sci_mcp.core import load_array

    source = np.array([[1, 2], [3, 4]], dtype=int)
    array = load_array(source)
    assert array.dtype == float
    assert array.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_array_reads_csv_without_header(sci_output_dir: Path) -> None:
    from sci_mcp.core import load_array

    (sci_output_dir / "x.csv").write_text("1,2\n3,4\n", encoding="utf-8")
    array = load_array("@x.csv")
    assert array.shape == (2, 2)
    assert array.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_array_reads_csv_with_header(sci_output_dir: Path) -> None:
    from sci_mcp.core import load_array

    (sci_output_dir / "x.csv").write_text(
        "time,value\n0,1\n1,2\n2,4\n", encoding="utf-8"
    )
    array = load_array("@x.csv")
    assert array.shape == (3, 2)
    assert array.tolist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]]


def test_load_array_reads_tsv(sci_output_dir: Path) -> None:
    from sci_mcp.core import load_array

    (sci_output_dir / "x.tsv").write_text("1\t2\n3\t4\n", encoding="utf-8")
    array = load_array("@x.tsv")
    assert array.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_load_array_reads_txt_single_column(sci_output_dir: Path) -> None:
    from sci_mcp.core import load_array

    (sci_output_dir / "x.txt").write_text("1\n2\n3\n", encoding="utf-8")
    array = load_array("@x.txt")
    assert array.shape == (3,)
    assert array.tolist() == [1.0, 2.0, 3.0]


def test_load_array_reads_npy(sci_output_dir: Path) -> None:
    import numpy as np

    from sci_mcp.core import load_array

    np.save(sci_output_dir / "x.npy", np.array([1.5, 2.5, 3.5]))
    array = load_array("@x.npy")
    assert array.tolist() == [1.5, 2.5, 3.5]


def test_load_array_reads_json_list(sci_output_dir: Path) -> None:
    import json

    from sci_mcp.core import load_array

    (sci_output_dir / "x.json").write_text(
        json.dumps([1.0, 2.0, 3.0]), encoding="utf-8"
    )
    array = load_array("@x.json")
    assert array.tolist() == [1.0, 2.0, 3.0]


def test_load_array_reads_json_column_dict(sci_output_dir: Path) -> None:
    import json

    from sci_mcp.core import load_array

    (sci_output_dir / "x.json").write_text(
        json.dumps({"a": [1, 2], "b": [3, 4]}), encoding="utf-8"
    )
    array = load_array("@x.json")
    assert array.shape == (2, 2)
    assert array.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_load_array_rejects_mismatched_json_columns(sci_output_dir: Path) -> None:
    import json

    from sci_mcp.core import load_array

    (sci_output_dir / "x.json").write_text(
        json.dumps({"a": [1, 2], "b": [3]}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_array("@x.json")


def test_load_array_missing_file_raises(sci_output_dir: Path) -> None:
    from sci_mcp.core import load_array

    with pytest.raises((FileNotFoundError, ValueError)):
        load_array("@does_not_exist.csv")


def test_load_array_rejects_plain_string() -> None:
    from sci_mcp.core import load_array

    with pytest.raises(ValueError):
        load_array("x.csv")


def test_ok_converts_nonfinite_floats_to_null() -> None:
    from sci_mcp.core import ok

    result = ok({"std": float("nan"), "t": float("inf"), "x": 1.0})
    assert result["ok"] is True
    assert result["result"]["std"] is None
    assert result["result"]["t"] is None
    assert result["result"]["x"] == 1.0
    assert len(result["warnings"]) == 2
    assert any("std" in warning and "NaN" in warning for warning in result["warnings"])
    assert any(
        "t" in warning and "Infinity" in warning for warning in result["warnings"]
    )


def test_ok_converts_nested_nonfinite_floats_to_null() -> None:
    from sci_mcp.core import ok

    result = ok({"a": [float("-inf")]})
    assert result["result"]["a"][0] is None
    assert any("a[0]" in warning for warning in result["warnings"])
    assert any("-Infinity" in warning for warning in result["warnings"])


def test_ok_omits_warnings_when_all_values_are_finite() -> None:
    from sci_mcp.core import ok

    result = ok({"x": 1.0, "items": [1, 2, 3]})
    assert result["result"] == {"x": 1.0, "items": [1, 2, 3]}
    assert "warnings" not in result


def test_ok_preserves_meta_arguments() -> None:
    from sci_mcp.core import ok

    result = ok("v", extra=1)
    assert result == {"ok": True, "result": "v", "extra": 1}


def test_ok_appends_to_existing_warnings_meta() -> None:
    from sci_mcp.core import ok

    result = ok(float("nan"), warnings=["existing"])
    assert result["warnings"][0] == "existing"
    assert len(result["warnings"]) == 2


def test_err_structure_is_unchanged() -> None:
    from sci_mcp.core import err

    assert err("boom") == {"ok": False, "error": "boom"}


# ---------------------------------------------------------------------------
# SBX: resource controls and process-tree cleanup.
# ---------------------------------------------------------------------------


def test_timeout_returns_error_with_captured_output(sci_output_dir: Path) -> None:
    # 5 s leaves room for the matplotlib/numpy preamble (~2 s here) before the
    # snippet prints and starts sleeping.
    result = run_python(
        'import time\nprint("before-timeout", flush=True)\ntime.sleep(30)',
        timeout_seconds=5,
    )
    assert result["ok"] is False
    assert "timeout" in result["error"].lower()
    assert result["returncode"] is None
    assert "before-timeout" in result["stdout"]
    assert result["elapsed_seconds"] >= 4.5
    assert isinstance(result["files"], list)


def test_code_over_limit_is_rejected(sci_output_dir: Path) -> None:
    code = "x = 1\n" * 16_700
    assert len(code) > 100_000
    result = run_python(code)
    assert result["ok"] is False
    assert "code too large" in result["error"]


def test_normal_execution_still_returns_stdout(sci_output_dir: Path) -> None:
    result = run_python('print("sandbox-ok")')
    assert result["ok"] is True
    payload = result["result"]
    assert payload["stdout"] == "sandbox-ok\n"
    assert payload["stderr"] == ""
    assert payload["returncode"] == 0
    assert isinstance(payload["elapsed_seconds"], float)
    assert payload["files"] == []


@pytest.mark.skipif(os.name != "nt", reason="taskkill-based tree kill is Windows-specific")
def test_timeout_kills_spawned_process_tree(sci_output_dir: Path) -> None:
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(f'GRANDCHILD_PID={child.pid}', flush=True)\n"
        "time.sleep(60)\n"
    )
    # The preamble takes ~2 s; 5 s guarantees the grandchild is spawned (and the
    # marker flushed) before the timeout fires.
    result = run_python(code, timeout_seconds=5)
    assert result["ok"] is False
    assert "timeout" in result["error"].lower()

    match = re.search(r"GRANDCHILD_PID=(\d+)", result["stdout"])
    assert match is not None, f"grandchild pid was not captured: {result!r}"
    pid = int(match.group(1))

    # taskkill /F is synchronous, but allow a short grace period for teardown.
    deadline = time.monotonic() + 5.0
    listing = None
    while True:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if str(pid) not in listing.stdout or time.monotonic() >= deadline:
            break
        time.sleep(0.2)

    assert listing is not None
    assert listing.returncode == 0, listing.stderr
    assert str(pid) not in listing.stdout, f"grandchild process {pid} survived timeout"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object memory cap")
def test_memory_limit_is_enforced(
    sci_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scientific preamble peaks near 1.1 GiB of commit charge; 1400 MiB
    # leaves headroom for imports while still rejecting a 800 MiB allocation.
    monkeypatch.setenv("SCI_MCP_SANDBOX_MEMORY_MB", "1400")
    result = run_python(
        'big = bytearray(800 * 1024 * 1024)\nprint("ALLOCATED", len(big))',
        timeout_seconds=60,
    )
    assert result["ok"] is True
    payload = result["result"]
    assert payload["returncode"] != 0
    assert "MemoryError" in payload["stderr"]
    assert "ALLOCATED" not in payload["stdout"]
