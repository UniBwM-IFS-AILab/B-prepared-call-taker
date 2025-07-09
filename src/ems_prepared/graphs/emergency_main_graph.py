"""Module for asynchronous operations in the emergency call workflow."""

import asyncio
import json
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

from pydantic_graph.graph import Graph, GraphRunResult
from pydantic_graph.persistence.file import FileStatePersistence
from rich import print

from ems_prepared.graphs.nodes import (
    RD1,
    RD2,
    TCPR,
    AskCaller,
    ChooseQuestion,
    ChooseSubGraph,
    Disposition,
    EvaluateAgentOutput,
    EvaluateState,
    Greeting,
    HighUrgency,
)
from ems_prepared.state_model.emergency_call_state import EmergencyCall

questions: list[str] = iter(
    [
        # "Hier ist der Notruf für Feuerwehr und Rettungsdienst.",
        "Mit wem spreche ich bitte?",
        "Wo genau ist der Einsatzort / die Einsatzstelle?",
        "Was ist jetzt akut passiert?",
        # "Sind Sie der Patient / beim Patienten?" # FIXME: only do this if needed, maybe a different data format (with a schema that defines a list of optional questions the llm can choose from)
    ]
)


class RunMode(Enum):
    """Enumeration for input modes in the emergency call workflow.

    Defines whether input is collected via CLI or as a request.
    """

    CLI = auto()
    MAIN = auto()
    TEST = auto()


async def save_mermaid_graph(graph: Graph, save_dir: Path, file_name_base: str) -> None:
    mermaid_output = graph.mermaid_code(start_node=Greeting)
    mermaid_file_path = save_dir / (file_name_base + ".md")
    mermaid_content = f"```mermaid\n{mermaid_output}\n```"
    _ = mermaid_file_path.write_text(mermaid_content, encoding="utf-8")

    graph.mermaid_save(save_dir / (file_name_base + ".jpg"))


async def save_state_json(
    result: GraphRunResult, save_dir: Path, file_name_base: str
) -> None:
    # log final state
    state_file_path = save_dir / f"{file_name_base}_final_state.json"
    _ = state_file_path.write_text(
        result.state.model_dump_json(indent=2), encoding="utf-8"
    )

    # log state model schema
    schema_file_path = save_dir / f"{file_name_base}_state_schema.json"
    schema_content = result.state.model_json_schema()
    formatted_schema_content = json.dumps(schema_content, indent=2)
    _ = schema_file_path.write_text(formatted_schema_content, encoding="utf-8")


def setup_file_persistence(graph: Graph, save_path: Path):
    persistence = FileStatePersistence(json_file=(save_path))
    persistence.set_graph_types(graph)
    return persistence


async def main(call_origin: RunMode = RunMode.MAIN) -> None:  # pragma: no cover
    """Run the main graph synchronously for demonstration purposes."""
    main_graph: Graph[EmergencyCall] = Graph[EmergencyCall](
        nodes=[
            Greeting,
            AskCaller,
            RD1,
            RD2,
            Disposition,
            ChooseSubGraph,
            EvaluateState,
            EvaluateAgentOutput,
            TCPR,
            HighUrgency,
            ChooseQuestion,
        ],
    )
    state: EmergencyCall = EmergencyCall()

    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_log_dir: Path = Path("logs") / "emergency_call" / timestamp
    current_log_dir.mkdir(parents=True, exist_ok=True)
    save_file_base = f"emergency_call_{timestamp}_{call_origin.name}"
    asyncio.create_task(
        save_mermaid_graph(
            main_graph,
            current_log_dir,
            f"{save_file_base}_mermaid",
        )
    )

    persistence = setup_file_persistence(
        main_graph, current_log_dir / f"{save_file_base}_persistence.json"
    )
    async with main_graph.iter(Greeting(), state=state, persistence=persistence) as run:
        async for node in run:
            try:
                print(f"Node: {node.get_node_id()}")
            except Exception as _:
                print(node)

    # result: GraphRunResult[EmergencyCall] = await main_graph.run(
    #     start_node=Greeting(),
    #     state=state,  # persistence=persistence
    # )
    result = run.result

    asyncio.create_task(save_state_json(result, current_log_dir, save_file_base))


if __name__ == "__main__":
    asyncio.run(main(RunMode.CLI))
