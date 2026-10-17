"""Shared pytest fixtures for sci-mcp tests."""

from pathlib import Path

import pytest


@pytest.fixture
def sci_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SCI_MCP_OUTPUT_DIR at a temporary directory and return it."""
    monkeypatch.setenv("SCI_MCP_OUTPUT_DIR", str(tmp_path))
    return tmp_path
