"""Module for asynchronous operations in the emergency call workflow."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph
from rich import print

from ems_prepared.policies.graphs.nodes import (
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
from ems_prepared.policies.graphs.utils import (
    save_mermaid_graph,
    save_state_json,
    setup_file_persistence,
)
from ems_prepared.util.settings import RunMode, Settings
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall

async def main(call_origin: RunMode = RunMode.MAIN) -> None:  # pragma: no cover
    """Run the main graph synchronously for demonstration purposes."""
    main_graph = Graph[EmergencyCall, Settings, EmergencyCall](
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
    deps = Settings(name="main_graph")
    state: EmergencyCall = EmergencyCall()

    graph_name: str = "emergency_call"
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_log_dir: Path = Path("logs") / graph_name / timestamp
    current_log_dir.mkdir(parents=True, exist_ok=True)
    save_file_base = f"{graph_name}_{timestamp}_{call_origin.name}"
    # _ = asyncio.create_task(
    #     save_mermaid_graph(
    #         main_graph,
    #         current_log_dir / f"{save_file_base}_mermaid",
    #     )
    # )

    persistence = setup_file_persistence(
        main_graph, current_log_dir / f"{save_file_base}_persistence.json"
    )
    async with main_graph.iter(
        Greeting(), state=state, deps=deps, persistence=persistence
    ) as run:
        async for node in run:
            if isinstance(node, BaseNode):
                print(f"Node: {node.get_node_id()}")  # type: ignore
            else:
                print(node)
    result = run.result
    if result is not None:
        _ = asyncio.create_task(
            save_state_json(result, current_log_dir / save_file_base)
        )


if __name__ == "__main__":
    asyncio.run(main(RunMode.CLI))
