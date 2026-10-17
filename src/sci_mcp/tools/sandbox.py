"""Sandboxed Python executor: runs user code in a subprocess rooted at the output dir."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from sci_mcp.core import err, ok, output_dir

_MAX_CAPTURE_CHARS = 20000
_MAX_TIMEOUT_SECONDS = 300
_MAX_CODE_CHARS = 100_000
_KILL_GRACE_SECONDS = 5

# Memory cap for the child process tree.  The scientific preamble alone peaks at
# roughly 1.1 GiB of commit charge on Windows, so a lower default would break
# normal imports.  The value can be overridden per call via the environment
# variable SCI_MCP_SANDBOX_MEMORY_MB (16 .. 65536 MiB).
_DEFAULT_MEMORY_LIMIT_BYTES = 2048 * 1024 * 1024
_MIN_MEMORY_LIMIT_MB = 16
_MAX_MEMORY_LIMIT_MB = 65536

# Windows Job Object constants / structures used for the best-effort memory cap.
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

_kernel32 = None

if os.name == "nt":
    try:
        from ctypes import wintypes

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        _kernel32.SetInformationJobObject.restype = wintypes.BOOL
        _kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        _kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        _kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    except Exception:  # pragma: no cover - defensive: degrade to no hard cap
        _kernel32 = None

# Pre-imported in every subprocess so user snippets can use the scientific stack
# without paying repeated import boilerplate.
_PREAMBLE = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import sympy as sp
"""


def _truncate(text: str) -> str:
    if len(text) > _MAX_CAPTURE_CHARS:
        return text[:_MAX_CAPTURE_CHARS] + "...[truncated]"
    return text


def _decode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Map relative path -> (size, mtime_ns) for every regular file under directory."""
    state: dict[str, tuple[int, int]] = {}
    try:
        entries = directory.rglob("*")
        for path in entries:
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            state[str(path.relative_to(directory))] = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        pass
    return state


def _changed_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]], directory: Path) -> list[str]:
    changed = []
    for relative, signature in after.items():
        if before.get(relative) != signature:
            changed.append(str((directory / relative).resolve()))
    changed.sort()
    return changed


def _memory_limit_bytes() -> int:
    """Return the configured per-process commit-charge cap in bytes."""
    raw = os.environ.get("SCI_MCP_SANDBOX_MEMORY_MB")
    if raw is not None:
        try:
            megabytes = int(raw)
        except (TypeError, ValueError):
            megabytes = 0
        if _MIN_MEMORY_LIMIT_MB <= megabytes <= _MAX_MEMORY_LIMIT_MB:
            return megabytes * 1024 * 1024
    return _DEFAULT_MEMORY_LIMIT_BYTES


def _create_memory_limit_job(proc: subprocess.Popen) -> object | None:
    """Best-effort: put ``proc`` in a Windows Job Object with a memory cap.

    Returns the job handle on success and ``None`` when the platform does not
    support it or any Win32 step fails.  A failure never blocks execution; the
    child then simply runs without a hard memory cap.
    """
    if _kernel32 is None:
        return None
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    try:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = _memory_limit_bytes()
        if not _kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not _kernel32.AssignProcessToJobObject(job, proc._handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    except Exception:
        _kernel32.CloseHandle(job)
        return None
    return job


def _close_job(job: object | None) -> None:
    if job is None or _kernel32 is None:
        return
    try:
        _kernel32.CloseHandle(job)
    except Exception:
        pass


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and its descendants; on Windows via ``taskkill /F /T``."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass

    # Fallback in case taskkill is unavailable or partially failed.
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _collect_after_timeout(proc: subprocess.Popen, exc: subprocess.TimeoutExpired) -> tuple[object, object]:
    """Drain whatever the child wrote before it was terminated."""
    try:
        return proc.communicate(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired as second:
        stdout = second.stdout if second.stdout is not None else exc.stdout
        stderr = second.stderr if second.stderr is not None else exc.stderr
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        return stdout, stderr


def run_python(code: str, timeout_seconds: int = 30) -> dict:
    """Execute Python code in a subprocess and return captured output plus produced files.

    Resource controls: code is limited to 100,000 characters; stdout and stderr
    are each truncated to 20,000 characters.  On timeout the whole process tree
    is terminated (Windows: ``taskkill /F /T``) and the error result carries the
    output captured so far plus ``elapsed_seconds`` and changed ``files``.

    On Windows the child (and its descendants) is assigned to a Job Object with
    a per-process memory cap: 2048 MiB by default, overridable with the
    environment variable ``SCI_MCP_SANDBOX_MEMORY_MB`` (16..65536).  Exceeding
    the cap surfaces as ``MemoryError`` in ``stderr``.  If the Job Object cannot
    be created or assigned, execution continues without a hard memory cap.  On
    non-Windows platforms no hard memory cap is applied (the standard library
    ``resource`` module is not available on Windows and no portable equivalent
    is used); rely on ``timeout_seconds`` and the caller's own limits there.
    """
    if not isinstance(code, str):
        return err("run_python failed: code must be a string")
    if len(code) > _MAX_CODE_CHARS:
        return err(
            f"run_python failed: code too large ({len(code)} characters; limit {_MAX_CODE_CHARS})"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= _MAX_TIMEOUT_SECONDS
    ):
        return err(
            f"run_python failed: timeout_seconds must be an integer between 1 and {_MAX_TIMEOUT_SECONDS}"
        )

    try:
        workdir = output_dir()
    except Exception as exc:
        return err(f"run_python failed: cannot prepare output directory: {exc}")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    before = _snapshot(workdir)
    started = time.perf_counter()

    script_path: str | None = None
    job: object | None = None
    try:
        # A script file avoids the ~32K Windows command-line limit, so the
        # documented 100,000-character code ceiling is actually usable.
        try:
            fd, script_path = tempfile.mkstemp(prefix="sci_mcp_sandbox_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(_PREAMBLE + code)
        except OSError as exc:
            elapsed = time.perf_counter() - started
            return err(
                f"run_python failed: cannot write temporary script: {exc}",
                stdout="",
                stderr="",
                returncode=None,
                elapsed_seconds=elapsed,
                files=[],
            )

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            elapsed = time.perf_counter() - started
            return err(
                f"run_python failed: could not start subprocess: {exc}",
                stdout="",
                stderr="",
                returncode=None,
                elapsed_seconds=elapsed,
                files=[],
            )

        job = _create_memory_limit_job(proc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(proc)
            stdout, stderr = _collect_after_timeout(proc, exc)
            elapsed = time.perf_counter() - started
            after = _snapshot(workdir)
            return err(
                f"run_python failed: timeout after {timeout_seconds} seconds",
                stdout=_truncate(_decode(stdout)),
                stderr=_truncate(_decode(stderr)),
                returncode=None,
                elapsed_seconds=elapsed,
                files=_changed_files(before, after, workdir),
            )
        except OSError as exc:
            _terminate_process_tree(proc)
            elapsed = time.perf_counter() - started
            after = _snapshot(workdir)
            return err(
                f"run_python failed: subprocess communication error: {exc}",
                stdout="",
                stderr="",
                returncode=None,
                elapsed_seconds=elapsed,
                files=_changed_files(before, after, workdir),
            )

        elapsed = time.perf_counter() - started
        after = _snapshot(workdir)
        return ok(
            {
                "stdout": _truncate(stdout),
                "stderr": _truncate(stderr),
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "files": _changed_files(before, after, workdir),
            }
        )
    finally:
        _close_job(job)
        if script_path is not None:
            try:
                os.unlink(script_path)
            except OSError:
                pass


def register(mcp: MCPServer) -> None:
    mcp.tool()(run_python)
