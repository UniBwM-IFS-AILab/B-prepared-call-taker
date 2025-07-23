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
from ems_prepared.graphs.utils import (
    save_mermaid_graph,
    save_state_json,
    setup_file_persistence,
)
from ems_prepared.settings import RunMode, Settings
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

    graph_name: str = "emergency_call"
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_log_dir: Path = Path("logs") / graph_name / timestamp
    current_log_dir.mkdir(parents=True, exist_ok=True)
    save_file_base = f"{graph_name}_{timestamp}_{call_origin.name}"
    asyncio.create_task(
        save_mermaid_graph(
            main_graph,
            current_log_dir / f"{save_file_base}_mermaid",
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
    result = run.result
    if result is not None:
        asyncio.create_task(save_state_json(result, current_log_dir / save_file_base))


if __name__ == "__main__":
    asyncio.run(main(RunMode.CLI))
