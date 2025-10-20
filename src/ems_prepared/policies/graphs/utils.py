import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_graph.graph import Graph, GraphRunResult
from pydantic_graph.nodes import BaseNode, StateT
from pydantic_graph.persistence.file import FileStatePersistence

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.policies.graphs.custom_persistence import setup_file_persistence
from ems_prepared.util.settings import Settings


async def save_mermaid_graph(
    graph: Graph[Any, Any, Any],
    save_path: Path,
) -> None:
    # Save the markdown file
    mermaid_file_path = (save_path / "graph").with_suffix(".md")
    mermaid_code = graph.mermaid_code()
    mermaid_content = f"```mermaid\n{mermaid_code}\n```"
    _ = mermaid_file_path.write_text(mermaid_content, encoding="utf-8")
    graph.mermaid_save((save_path / "graph").with_suffix(".jpg"))


async def save_state_json(
    result: GraphRunResult[Any, StateT],
    save_path: Path,
) -> None:
    # log final state
    state: BaseModel = result.state
    json_content = state.model_dump_json(indent=2)
    _ = (save_path / "final_state.json").write_text(data=json_content, encoding="utf-8")

    # log state model schema
    schema_content = result.state.model_json_schema()
    formatted_schema_content = json.dumps(schema_content, indent=2)
    _ = (save_path / "state_schema.json").write_text(
        formatted_schema_content, encoding="utf-8"
    )


async def init_graph(
    node_list: list[Any],
    init_node: BaseNode[Any, Any, Any],
    deps: Settings,
    init_state: BaseModel,
    prefix: str,
):
    graph = Graph[BaseModel, BaseModel, BaseModel](
        nodes=node_list,
    )
    persistence = await setup_file_persistence(graph, deps.save_path, prefix=prefix)
    try:
        snapshots = await persistence.load_all()
        if len(snapshots) == 0:
            print("No snapshots found, initializing new graph.")
            await graph.initialize(init_node, persistence=persistence, state=init_state)
    except Exception as e:
        print(f"Error initializing graph: {e}")

    return graph, persistence
