"""Module for asynchronous operations in the emergency call workflow."""

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic.main import BaseModel
from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph, GraphRunResult
from rich import print

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
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
    init_graph,
    save_mermaid_graph,
    save_state_json,
)
from ems_prepared.util.settings import Settings
from ems_prepared.util.user_interaction import tell_user


async def run_graph(
    deps: Settings, init_state: EmergencyCall
) -> GraphRunResult[Any, Any] | None:  # pragma: no cover
    """Run the graph."""

    graph, persistence = await init_graph(
        node_list=[
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
        init_node=Greeting(),
        deps=deps,
        init_state=init_state,
        prefix="main_",
    )

    async with graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
        print(run.state)

        async for node in run:
            if isinstance(node, BaseNode):
                print(f"Node: {node.get_node_id()}")  # type: ignore
            else:
                print(node)

    result: GraphRunResult[EmergencyCall, EmergencyCall] = run.result  # type: ignore
    if run.result is not None:
        asyncio.create_task(save_state_json(result, deps.save_path))
        asyncio.create_task(save_mermaid_graph(graph, deps.save_path))
        (deps.save_path / "deps.json").write_text(
            data=deps.model_dump_json(indent=2), encoding="utf-8"
        )

        result_state: EmergencyCall = result.state

        if result_state.cpr_needed:
            from ems_prepared.policies.graphs import tcpr_subgraph

            await tcpr_subgraph.run_graph(
                deps=deps,
                # state=EmergencyCall.model_validate(result.model_dump()),
                init_state=result_state,
            )
        await tell_user("The emergency call has been processed.", deps)

    return run.result


async def main(deps=None, state=None):  # pragma: no cover
    ### parse user override # TODO better CLI, use typer, argparse or pydantic-cli
    import sys

    session_id = (
        UUID(int=int(sys.argv[1]))
        if len(sys.argv) > 1
        # use 0000... as the deterministic user for tests
        else UUID(int=0)
    )
    user_id = UUID(int=0)
    print(f"{user_id} / {session_id}")
    ###

    ### fix resume
    import json

    file_paths = [
        Path(f"logs/{user_id.hex}/{session_id.hex}/{'main_'}persistence.json"),
        Path(f"logs/{user_id.hex}/{session_id.hex}/{'tcpr_'}persistence.json"),
    ]
    for file_path in file_paths:
        if file_path.exists():
            prev_run_finished = False
            with open(file_path, "r", encoding="utf-8") as f:
                json_state = json.load(f)
                prev_run_finished = json_state[-1]["kind"] == "end"

            if prev_run_finished:
                import os

                os.remove(f"logs/{user_id.hex}/{session_id.hex}/persistence.json")
    ###

    deps: Settings = deps or Settings(
        name="emergency_call", user_id=user_id, session_id=session_id
    )
    state: EmergencyCall = state or EmergencyCall()

    _: GraphRunResult[EmergencyCall, EmergencyCall] | None = await run_graph(
        deps, state
    )


if __name__ == "__main__":
    asyncio.run(main())
