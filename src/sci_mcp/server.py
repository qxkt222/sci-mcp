"""MCP server assembly: discover tool modules under sci_mcp.tools and register them."""

from __future__ import annotations

import importlib
import pkgutil
import sys

from mcp.server.mcpserver import MCPServer

import sci_mcp.tools as _tools_pkg

INSTRUCTIONS = (
    "Local scientific computing server. Provides symbolic math, physics/units, "
    "numerical simulation, statistics and plotting tools, plus run_python for "
    "arbitrary Python execution in a subprocess sandbox. Files produced by tools "
    "are written to the configured output directory and returned as absolute paths."
)

mcp = MCPServer("sci-mcp", instructions=INSTRUCTIONS)


def _iter_tool_modules() -> list[str]:
    """Return tool module names in deterministic (sorted) order."""
    names = [
        info.name
        for info in pkgutil.iter_modules(_tools_pkg.__path__)
        if not info.name.startswith("_")
    ]
    return sorted(names)


def _register_tool_modules() -> None:
    for name in _iter_tool_modules():
        qualified = f"{_tools_pkg.__name__}.{name}"
        try:
            module = importlib.import_module(qualified)
        except Exception as exc:  # keep one broken module from taking down the server
            print(f"[sci-mcp] failed to import tool module {qualified}: {exc}", file=sys.stderr)
            continue
        register = getattr(module, "register", None)
        if callable(register):
            register(mcp)


_register_tool_modules()


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()
