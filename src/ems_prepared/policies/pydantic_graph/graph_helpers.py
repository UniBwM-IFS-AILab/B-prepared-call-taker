from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph

from ems_prepared.model.context import Settings
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    ResumableFilePersistence,
)

logger = logging.getLogger(__name__)
StateT = TypeVar("StateT")
RunEndT = TypeVar("RunEndT")


def save_mermaid_graph(
    graph: Graph[Any, Any, Any],
    save_path: Path,
    *,
    stem: str = "graph",
) -> None:
    """Save mermaid diagram as markdown and JPG."""
    mermaid_file_path = (save_path / stem).with_suffix(".md")
    mermaid_code = graph.mermaid_code()
    mermaid_content = f"```mermaid\n{mermaid_code}\n```"
    mermaid_file_path.write_text(mermaid_content, encoding="utf-8")

    image_path = (save_path / stem).with_suffix(".jpg")
    try:
        graph.mermaid_save(image_path)
    except Exception as error:
        logger.warning(
            "Mermaid image export failed for %s: %s. Continuing without image.",
            image_path,
            error,
        )


@overload
async def restore_graph(
    graph: Graph[StateT, Settings, RunEndT],
    persistence: ResumableFilePersistence[StateT, RunEndT],
    *,
    init_node: None = None,
    init_state: None = None,
) -> tuple[BaseNode[StateT, Settings, RunEndT], StateT] | None: ...


@overload
async def restore_graph(
    graph: Graph[StateT, Settings, RunEndT],
    persistence: ResumableFilePersistence[StateT, RunEndT],
    *,
    init_node: BaseNode[StateT, Settings, RunEndT],
    init_state: StateT,
) -> tuple[BaseNode[StateT, Settings, RunEndT], StateT]: ...


async def restore_graph(
    graph: Graph[StateT, Settings, RunEndT],
    persistence: ResumableFilePersistence[StateT, RunEndT],
    *,
    init_node: BaseNode[StateT, Settings, RunEndT] | None = None,
    init_state: StateT | None = None,
) -> tuple[BaseNode[StateT, Settings, RunEndT], StateT] | None:
    """Load the next persisted node or initialize a new run if one is requested.

    `load_next()` moves a snapshot from `created` to `pending`. We immediately
    re-queue the chosen node/state as a fresh `created` snapshot so
    `iter_from_persistence()` can resume it on the next line.
    """
    snapshot = await persistence.load_next()
    if snapshot is None:
        if init_node is None or init_state is None:
            return None
        node = init_node
        state = init_state
    else:
        node = snapshot.node
        state = snapshot.state

    await graph.initialize(node, persistence=persistence, state=state)
    return node, state
