import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_graph.graph import Graph, GraphRunResult
from pydantic_graph.nodes import StateT
from pydantic_graph.persistence.file import FileStatePersistence

from ems_prepared.settings import Settings
from ems_prepared.state_model.emergency_call_state import EmergencyCall


async def save_mermaid_graph(
    graph: Graph[EmergencyCall, Settings, EmergencyCall],
    save_path: Path,
) -> None:
    mermaid_file_path = save_path.with_suffix(".md")
    mermaid_content = f"```mermaid\n{graph.mermaid_code()}\n```"
    _ = mermaid_file_path.write_text(mermaid_content, encoding="utf-8")

    graph.mermaid_save(save_path.with_suffix(".jpg"))


async def save_state_json(
    result: GraphRunResult[BaseModel, StateT],
    save_path: Path,
) -> None:
    # log final state
    state_file_path = Path(f"{save_path}_final_state.json")
    state: BaseModel = result.state
    json_content = state.model_dump_json(indent=2)
    _ = state_file_path.write_text(data=json_content, encoding="utf-8")

    # log state model schema
    schema_file_path = Path(f"{save_path}_state_schema.json")
    schema_content = result.state.model_json_schema()
    formatted_schema_content = json.dumps(schema_content, indent=2)
    _ = schema_file_path.write_text(formatted_schema_content, encoding="utf-8")


def setup_file_persistence(
    graph: Graph[EmergencyCall, Settings, EmergencyCall], save_path: Path
):
    persistence = FileStatePersistence[EmergencyCall, EmergencyCall](
        json_file=(save_path)
    )
    persistence.set_graph_types(graph)
    return persistence
