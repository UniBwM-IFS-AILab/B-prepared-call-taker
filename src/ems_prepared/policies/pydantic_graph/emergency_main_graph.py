"""Module for asynchronous operations in the emergency call workflow."""

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from pydantic_graph import BaseNode, End
from pydantic_graph.graph import Graph

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import SessionResumeError
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    setup_resumable_file_persistence,
)
from ems_prepared.policies.pydantic_graph.graph_helpers import (
    restore_graph,
    save_mermaid_graph,
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
    """Run the CLI debug flow."""
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
    persistence = await setup_resumable_file_persistence(
        graph, deps.storage.save_path, prefix="main_"
    )

    restored = await restore_graph(graph, persistence)

    if restored is not None:
        deps.telemetry.logger.info("Resuming from persisted graph state...")
        node, state = restored
        deps.telemetry.logger.info(f"[Node] {node.get_node_id()}")

        deps.telemetry.logger.debug(
            f"Resumed State:\t{state.call_state.model_dump(exclude_none=True)}"
        )
    else:
        deps.telemetry.logger.info("Initializing new graph run...")
        node, state = await restore_graph(
            graph,
            persistence,
            init_node=Start(),
            init_state=GraphState(),
        )

    return node, state, persistence


async def resume_existing_from_persistence(
    deps: Settings,
    graph: Graph[GraphState, Settings, EmergencyCall],
):
    persistence = await setup_resumable_file_persistence(
        graph, deps.storage.save_path, prefix="main_"
    )

    try:
        restored = await restore_graph(graph, persistence)
    except Exception as exc:
        raise SessionResumeError(
            f"Failed to restore graph persistence for session: {deps.session_id}"
        ) from exc

    if restored is None:
        raise SessionResumeError(
            f"Missing graph persistence for session: {deps.session_id}"
        )

    node, state = restored
    deps.telemetry.logger.info("Resuming from persisted graph state...")
    deps.telemetry.logger.info(f"[Node] {node.get_node_id()}")
    deps.telemetry.logger.debug(
        f"Resumed State:\t{state.call_state.model_dump(exclude_none=True)}"
    )
    return node, persistence


async def _run_loaded_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    node: BaseNode[GraphState, Settings, EmergencyCall],
    persistence,
    answer: str | None = None,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
) -> RunGraphNode:
    from ems_prepared.policies.pydantic_graph import tcpr_subgraph

    if answer is None and isinstance(node, QuestionNode):
        deps.telemetry.logger.debug(str(node))
        return node
    if answer is not None and isinstance(node, QuestionNode):
        node = ExtractState(node.question, answer)

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

    result = graph_run.result
    assert result is not None

    if result.state.call_state.cpr_needed and not result.state.call_state.ems_arrived:
        save_mermaid_graph(graph, deps.storage.save_path)

        tcpr_result = await tcpr_subgraph.run_graph(
            deps=deps,
            init_state=result.state,
        )
        if isinstance(tcpr_result, End):
            await async_wrapper(
                deps.emit(
                    deps,
                    "The emergency call has been processed.",
                )
            )
        return tcpr_result

    await async_wrapper(
        deps.emit(
            deps,
            "The emergency call has been processed.",
        )
    )
    flush_logger(deps.telemetry.messages_logger)
    flush_logger(deps.telemetry.state_logger)
    save_graph_run_results(graph, graph_run, deps, on_complete=on_complete)
    return node


async def resume_existing_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    answer: str | None = None,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
) -> RunGraphNode:
    """Resume an existing graph-backed session turn."""
    from ems_prepared.policies.pydantic_graph import tcpr_subgraph

    tcpr_result = await tcpr_subgraph.run_graph(deps=deps, answer=answer)
    if tcpr_result is not None:
        if isinstance(tcpr_result, End):
            await async_wrapper(
                deps.emit(
                    deps,
                    "The emergency call has been processed.",
                )
            )
        return tcpr_result

    node, persistence = await resume_existing_from_persistence(deps, graph)
    return await _run_loaded_graph(
        graph,
        deps,
        node,
        persistence,
        answer=answer,
        on_complete=on_complete,
    )


async def run_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    answer: str | None = None,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
) -> RunGraphNode:
    """Run the graph."""
    return await resume_existing_graph(
        graph,
        deps,
        answer=answer,
        on_complete=on_complete,
    ) if answer is not None else await _run_initial_graph(
        graph,
        deps,
        on_complete=on_complete,
    )


async def _run_initial_graph(
    graph: Graph[GraphState, Settings, EmergencyCall],
    deps: Settings,
    *,
    on_complete: Callable[[EmergencyCall, Sequence[Any]], None] | None = None,
) -> RunGraphNode:
    from ems_prepared.policies.pydantic_graph import tcpr_subgraph

    tcpr_result = await tcpr_subgraph.run_graph(deps=deps, answer=None)
    if tcpr_result is not None:
        if isinstance(tcpr_result, End):
            await async_wrapper(
                deps.emit(
                    deps,
                    "The emergency call has been processed.",
                )
            )
        return tcpr_result

    node: BaseNode[GraphState, Settings, EmergencyCall]
    node, _, persistence = await resume_from_persistence(deps, graph)
    return await _run_loaded_graph(
        graph,
        deps,
        node,
        persistence,
        answer=None,
        on_complete=on_complete,
    )


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
    result = graph_run.result
    assert result is not None
    save_mermaid_graph(graph, deps.storage.save_path)
    if on_complete is not None:
        on_complete(
            result.state.call_state,
            result.state.message_history,
        )
