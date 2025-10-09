"""Module for asynchronous operations in the emergency call workflow."""

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph
from rich import print

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.policies.graphs.custom_persistence import setup_file_persistence
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
)
from ems_prepared.util.settings import Settings


async def init_graph(user_id: UUID, session_id: UUID):
    graph = Graph[EmergencyCall, Settings, EmergencyCall](
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
    save_path: Path = Path("logs") / user_id.hex / session_id.hex

    # breakpoint()

    # This one is only initialised to provd
    persistence = await setup_file_persistence(graph, save_path)

    try:
        snapshots = await persistence.load_all()
        if len(snapshots) == 0:
            print("No snapshots found, initializing new graph.")
            await graph.initialize(
                Greeting(), persistence=persistence, state=EmergencyCall()
            )
    except Exception as e:
        print(f"Error initializing graph: {e}")

    return graph


# async def main(call_origin: RunMode = RunMode.MAIN) -> None:  # pragma: no cover
async def main(user_id: UUID, session_id: UUID) -> None:  # pragma: no cover
    """Run the main graph synchronously for demonstration purposes."""

    deps = Settings(name="emergency_call", user_id=user_id, session_id=session_id)

    graph = await init_graph(user_id, session_id)
    save_path: Path = Path("logs") / user_id.hex / session_id.hex
    persistence = await setup_file_persistence(graph, save_path)
    # breakpoint()

    async with graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
        print(run.state)
        # breakpoint()

        async for node in run:
            if isinstance(node, BaseNode):
                print(f"Node: {node.get_node_id()}")  # type: ignore
            else:
                print(node)

    _ = asyncio.create_task(
        save_mermaid_graph(
            graph,
            save_path,
        )
    )
    if run.result is not None:
        _ = asyncio.create_task(save_state_json(run.result, save_path))


if __name__ == "__main__":
    # TODO: use settings here for uuids
    import sys

    session_id = UUID(int=int(sys.argv[1])) if len(sys.argv) > 1 else UUID(int=0)
    user_id = UUID(int=0)

    print(f"{user_id} / {session_id}")
    import json

    file_path = Path(f"logs/{user_id.hex}/{session_id.hex}/persistence.json")
    if file_path.exists():
        prev_run_finished = False
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            prev_run_finished = state[-1]["kind"] == "end"

        if prev_run_finished:
            import os

            # os.rmdir(f"logs/{user_id.hex}/{session_id.hex}")
            os.remove(f"logs/{user_id.hex}/{session_id.hex}/persistence.json")

    # use 0000... as the deterministic user for tests
    asyncio.run(main(user_id, session_id))
