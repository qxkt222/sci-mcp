"""Health check tool: report tool module import status and the exposed tool surface."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from mcp.server.mcpserver import MCPServer

import sci_mcp.tools as _tools_pkg

from ..core import ok


def _tool_function_names(module: object) -> list[str]:
    """Return public synchronous functions defined by the module itself.

    Imported helpers (for example ``core.ok`` or ``sympy.sympify``) are ignored
    so that the count reflects the module's own tool surface.
    """
    names: list[str] = []
    for name, attribute in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or name == "register":
            continue
        if inspect.iscoroutinefunction(attribute):
            continue
        if attribute.__module__ != module.__name__:
            continue
        names.append(name)
    return sorted(names)


def _module_record(module_name: str) -> dict:
    """Import one tool module and describe its status without ever raising."""
    qualified = f"{_tools_pkg.__name__}.{module_name}"
    record: dict = {
        "module": module_name,
        "imported": False,
        "error": None,
        "has_register": False,
        "tool_functions": [],
    }

    try:
        module = importlib.import_module(qualified)
    except Exception as exc:  # a broken module must not break the health check
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["imported"] = True
    record["has_register"] = callable(getattr(module, "register", None))

    try:
        record["tool_functions"] = _tool_function_names(module)
    except Exception as exc:
        record["error"] = f"inspection failed: {type(exc).__name__}: {exc}"

    return record


def health_check() -> dict:
    """Report tool module import status and the current tool surface."""
    try:
        module_names = sorted(
            info.name
            for info in pkgutil.iter_modules(_tools_pkg.__path__)
            if not info.name.startswith("_")
        )
    except Exception as exc:
        # Record discovery failure in a way that still satisfies the envelope contract.
        return ok(
            {
                "modules": [],
                "module_count": 0,
                "failed_modules": ["<discovery>"],
                "tool_function_count": 0,
                "healthy": False,
                "error": f"module discovery failed: {type(exc).__name__}: {exc}",
            }
        )

    modules = [_module_record(name) for name in module_names]
    failed_modules = [record["module"] for record in modules if not record["imported"]]
    tool_function_count = sum(len(record["tool_functions"]) for record in modules)

    return ok(
        {
            "modules": modules,
            "module_count": len(modules),
            "failed_modules": failed_modules,
            "tool_function_count": tool_function_count,
            "healthy": failed_modules == [],
        }
    )


def register(mcp: MCPServer) -> None:
    """Register the health check tool with the MCP server."""
    mcp.tool()(health_check)
