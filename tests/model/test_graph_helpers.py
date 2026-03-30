"""Tests for graph artifact helper behavior."""

from __future__ import annotations

import logging
from pathlib import Path

from ems_prepared.policies.pydantic_graph.graph_helpers import save_mermaid_graph


class _FailingMermaidGraph:
    """Test double that fails during image export."""

    def mermaid_code(self) -> str:
        return "graph TD; A-->B;"

    def mermaid_save(self, path: Path) -> None:
        raise RuntimeError("simulated mermaid render timeout")


def test_save_mermaid_graph_keeps_markdown_when_image_export_fails(
    caplog,
    tmp_path: Path,
) -> None:
    """Image export failure should not crash session artifact persistence."""
    caplog.set_level(logging.WARNING)

    save_mermaid_graph(_FailingMermaidGraph(), tmp_path)  # pyright: ignore[reportArgumentType]

    markdown_path = tmp_path / "graph.md"
    assert markdown_path.exists()
    assert "```mermaid" in markdown_path.read_text(encoding="utf-8")
    assert "Mermaid image export failed" in caplog.text
