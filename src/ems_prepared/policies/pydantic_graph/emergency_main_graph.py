"""Module for asynchronous operations in the emergency call workflow."""

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic_graph import BaseNode
from pydantic_graph.graph import Graph
from pydantic_graph.nodes import End

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import SessionResumeError
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    ResumableFilePersistence,
)
from ems_prepared.policies.pydantic_graph.graph_helpers import save_mermaid_graph
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
from ems_prepared.util.helpers import async_wrapper
from ems_prepared.util.logger import flush_logger
from ems_prepared.util.user_interaction import prompt_user

RunGraphNode = QuestionNode | MessageNode | End[EmergencyCall]


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
    # use 0 as the user id for tests
    session_id = UUID(int=0)
    user_id = UUID(int=0)

    deps = deps or Settings(
        name="emergency_call",
        user_id=user_id,
        session_id=session_id,
        policy_name="graph",
    )
    deps.telemetry.logger.info(f"{user_id} / {session_id}")
    deps.telemetry.logger.info(f"log dir: {deps.storage.save_path}")
    await loop_graph(deps)


async def loop_graph(deps: Settings):
    graph: Graph[GraphState, Settings, EmergencyCall] = await build_graph()

    save_mermaid_graph(graph, deps.storage.save_path)

    answer: str | None = None
    result: RunGraphNode = await run_graph(graph, deps, answer)
    while not isinstance(result, End):
        answer = None

        deps.telemetry.logger.debug(f"Graph returned {type(result)}")

        if isinstance(result, QuestionNode):
            deps.telemetry.logger.debug(f"Prompting user: {result.question}")
            answer = await prompt_user(result.question, deps=deps)

        elif isinstance(result, MessageNode):
            deps.telemetry.logger.debug("Emitting message to user.")
            await async_wrapper(
                deps.emit(
                    deps,
                    f"Message: {result.messages.get(deps.locale, 'No message for this locale.')}",
                )
            )
        else:
            raise ValueError(
                "Unhandled graph result type. Graph should not have exited."
            )
        result = await run_graph(graph, deps, answer)

    deps.telemetry.logger.info("Graph ended")
    return result


async def resume_from_persistence(
    deps: Settings,
    graph: Graph[GraphState, Settings, EmergencyCall],
):
    persistence = ResumableFilePersistence[GraphState, EmergencyCall](
        json_file=(deps.storage.save_path / "main_persistence.json")
    )
    if persistence.should_set_types():
        persistence.set_graph_types(graph)

    try:
        snapshot = await persistence.load_next()
    except Exception as exc:
        if deps.resume_expected:
            raise SessionResumeError(
                f"Failed to restore graph persistence for session: {deps.session_id}"
            ) from exc
        raise

    if snapshot:
        deps.telemetry.logger.info("Resuming from persisted graph state...")
        deps.telemetry.logger.info(f"[Node] {snapshot.node.get_node_id()}")

        state: GraphState = snapshot.state
        deps.telemetry.logger.debug(
            f"Resumed State:\t{state.call_state.model_dump(exclude_none=True)}"
        )

        node: BaseNode[GraphState, Settings, EmergencyCall] | End[EmergencyCall] = (
            snapshot.node
        )
    else:
        if deps.resume_expected:
            raise SessionResumeError(
                f"Missing graph persistence for session: {deps.session_id}"
            )
        deps.telemetry.logger.info("Initializing new graph run...")

        state = GraphState()
        node = Start()

    if not isinstance(node, End):
        await graph.initialize(node, persistence=persistence, state=state)

    return node, state, persistence


async def run_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    answer: str | None = None,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
) -> RunGraphNode:
    """Run the graph."""

    node: BaseNode[GraphState, Settings, EmergencyCall] | End[EmergencyCall]
    node, _, persistence = await resume_from_persistence(deps, graph)
    if answer is not None and isinstance(node, QuestionNode):
        node = ExtractState(node.question, answer)

    if isinstance(node, End):
        deps.telemetry.logger.info("Graph already finished.")

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
            deps.telemetry.logger.debug(f"[Node] {node.get_node_id()}")

            if isinstance(node, QuestionNode):
                deps.telemetry.logger.debug(str(node))
                return node

            elif isinstance(node, MessageNode):
                return node

    if isinstance(node, End):
        await async_wrapper(
            graph_run.deps.emit(
                deps,
                "The emergency call has been processed.",
            )
        )
        # Flush log handlers before saving
        flush_logger(deps.telemetry.messages_logger)
        flush_logger(deps.telemetry.state_logger)

        save_graph_run_results(graph, graph_run, deps, on_complete=on_complete)
        return node

    raise RuntimeError("Graph finished without returning an End node.")


def save_graph_run_results(
    graph,
    graph_run,
    deps,
    *,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
):
    """Save graph run results, state, and mermaid diagram to disk.

    Note: Logging should be completed before calling this function.
    """
    if graph_run.result is not None:
        save_mermaid_graph(graph, deps.storage.save_path)
        if on_complete is not None:
            on_complete(
                graph_run.result.state.call_state,
                graph_run.result.state.message_history,
            )
