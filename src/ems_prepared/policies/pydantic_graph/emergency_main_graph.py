"""Module for asynchronous operations in the emergency call workflow."""

import asyncio
import logging
from typing import Any
from uuid import UUID

# logger = logging.getLogger(__name__)
from loguru import logger
from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph, GraphRunResult
from pydantic_graph.nodes import End, StateT
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.file import FileStatePersistence

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    clear_old_run,
)
from ems_prepared.policies.pydantic_graph.nodes import (
    RD1,
    RD2,
    TCPR,
    AskCaller,
    ChooseQuestion,
    ChooseSubGraph,
    Disposition,
    EvaluateAgentOutput,
    EvaluateState,
    ExtractState,
    Greeting,
    HighUrgency,
    MergeState,
    MessageNode,
    QuestionNode,
    Start,
)
from ems_prepared.policies.pydantic_graph.utils import (
    save_mermaid_graph,
    save_state_json,
)
from ems_prepared.util.helpers import async_wrapper
from ems_prepared.util.logger import flush_logger, setup_console_logging
from ems_prepared.util.settings import Settings
from ems_prepared.util.user_interaction import prompt_user


async def async_print(msg: str):
    logger.info(msg)


async def build_graph():
    graph: Graph[GraphState, Settings, EmergencyCall] = Graph[
        GraphState, Settings, EmergencyCall
    ](
        nodes=[
            Start,
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
            ExtractState,
            MergeState,
        ]
    )

    return graph


async def debug_cli(deps: Settings | None = None):
    """Main function for cli debug usage."""
    import sys

    # use 0 as the user id for tests
    session_id = UUID(int=0)
    user_id = UUID(int=0)
    logger.info(f"{user_id} / {session_id}")

    if len(sys.argv) > 1:
        logger.info("Clearing old run...")
        await clear_old_run(user_id)

    deps = deps or Settings(
        name="emergency_call", user_id=user_id, session_id=session_id, emit=async_print
    )
    logger.info(f"log dir: {deps.save_path}")
    await loop_graph(deps)


async def loop_graph(deps: Settings):
    graph: Graph[GraphState, Settings, EmergencyCall] = await build_graph()

    save_mermaid_graph(graph, deps.save_path)

    answer: str | None = None
    while not isinstance(result := await run_graph(graph, deps, answer), End):
        answer = None

        logger.debug(f"Graph returned {type(result)}")

        if hasattr(result, "question") and isinstance(result.question, str):
            logger.debug(f"Prompting user: {result.question}")
            answer = await prompt_user(result.question, deps=deps)

        elif hasattr(result, "messages") and type(result.messages) is dict:
            logger.debug("Emitting message to user.")
            await async_wrapper(
                deps.emit(
                    f"Message: {result.messages.get(deps.locale, 'No message for this locale.')}"
                )
            )
        else:
            raise ValueError(
                "Unhandled graph result type. Graph should not have exited."
            )

    logger.info("Graph ended")
    return result


async def resume_from_persistence(
    deps: Settings,
    graph: Graph[GraphState, Settings, EmergencyCall],
):
    persistence = FileStatePersistence[GraphState, EmergencyCall](
        json_file=(deps.save_path / "main_persistence.json")
    )
    if persistence.should_set_types():
        persistence.set_graph_types(graph)

    if snapshot := await persistence.load_next():
        logger.info("Resuming from persisted graph state...")
        logger.info(f"[Node] {snapshot.node.get_node_id()}")

        state: GraphState = snapshot.state
        logger.debug(
            f"Resumed State:\t{state.call_state.model_dump(exclude_none=True)}"
        )

        node: BaseNode[GraphState, Settings, EmergencyCall] | End[EmergencyCall] = (
            snapshot.node
        )
    else:
        logger.info("Initializing new graph run...")

        state = GraphState()
        node = Start()

    if not isinstance(node, End):
        await graph.initialize(node, persistence=persistence, state=state)

    return node, state, persistence


async def run_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    answer: str | None = None,
) -> (
    GraphRunResult[Any, Any]
    | End[EmergencyCall]
    | BaseNode[GraphState, Settings, EmergencyCall]
    | None
):
    """Run the graph."""

    node: BaseNode[GraphState, Settings, EmergencyCall] | End[EmergencyCall]
    node, state, persistence = await resume_from_persistence(deps, graph)
    if answer is not None and hasattr(node, "question"):
        node = ExtractState(node.question, answer)

    if isinstance(node, End):
        logger.info("Graph already finished.")

        # FIXME: adopt new logic for TCPR subgraph
        # if graph_run.result is not None and graph_run.result.state.cpr_needed:
        #     from ems_prepared.policies.pydantic_graph import tcpr_subgraph

        #     await tcpr_subgraph.run_graph(
        #         deps=deps,
        #         init_state=graph_run.result.state,
        #     )
        return node

    # async with graph.iter(
    #     node, state=state, persistence=persistence, deps=deps
    # ) as graph_run:
    async with graph.iter_from_persistence(
        persistence=persistence, deps=deps
    ) as graph_run:
        while not isinstance(node := await graph_run.next(node), End):
            logger.debug(f"[Node] {node.get_node_id()}")

            if isinstance(node, QuestionNode):
                logger.debug(str(node))
                return node

            elif isinstance(node, MessageNode):
                return node

    if isinstance(node, End):
        await async_wrapper(
            graph_run.deps.emit("The emergency call has been processed.")
        )
        # Flush log handlers before saving
        flush_logger(deps.messages_logger)
        flush_logger(deps.state_logger)

        save_run(graph, graph_run, deps)
        return node


def save_run(graph, graph_run, deps):
    """Save graph run results and state to disk.

    Note: Logging should be completed before calling this function.
    """
    if graph_run.result is not None:
        save_state_json(graph_run.result, deps.save_path)
        save_mermaid_graph(graph, deps.save_path)
        (deps.save_path / "deps.json").write_text(
            data=deps.model_dump_json(indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    asyncio.run(debug_cli())
