from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic_graph.graph import Graph
from pydantic_graph.nodes import BaseNode
from pydantic_graph.persistence.file import FileStatePersistence

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import Settings
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    setup_file_persistence,
)

logger = logging.getLogger(__name__)


def save_mermaid_graph(
    graph: Graph[Any, Any, Any],
    save_path: Path,
) -> None:
    """Save mermaid diagram as markdown and JPG."""
    mermaid_file_path = (save_path / "graph").with_suffix(".md")
    mermaid_code = graph.mermaid_code()
    mermaid_content = f"```mermaid\n{mermaid_code}\n```"
    mermaid_file_path.write_text(mermaid_content, encoding="utf-8")

    image_path = (save_path / "graph").with_suffix(".jpg")
    try:
        graph.mermaid_save(image_path)
    except Exception as error:
        logger.warning(
            "Mermaid image export failed for %s: %s. Continuing without image.",
            image_path,
            error,
        )


async def init_graph(
    node_list: list[Any],
    init_node: BaseNode[Any, Any, Any],
    deps: Settings,
    init_state: EmergencyCall,
    prefix: str,
) -> tuple[Graph[EmergencyCall, Settings, EmergencyCall], FileStatePersistence]:
    graph = Graph[EmergencyCall, Settings, EmergencyCall](
        nodes=node_list,
    )
    persistence = await setup_file_persistence(
        graph, deps.storage.save_path, prefix=prefix
    )
    await graph.initialize(init_node, persistence=persistence, state=init_state)
    return graph, persistence
