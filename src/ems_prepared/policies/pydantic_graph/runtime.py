"""Graph policy runtime adapter and builder."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.graph import Graph
from pydantic_graph.nodes import End

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import BackendEvent, ConversationPolicy
from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
    clear_old_run,
)
from ems_prepared.policies.pydantic_graph.emergency_main_graph import (
    build_graph,
    run_graph,
)
from ems_prepared.policies.pydantic_graph.nodes import MessageNode, QuestionNode
from ems_prepared.policies.shared import (
    record_completion_artifacts,
    to_event_payload,
)
from ems_prepared.util.logger import flush_logger


@dataclass(slots=True)
class GraphConversationPolicy(ConversationPolicy):
    """Adapter from the pydantic-graph flow to `BackendEvent`."""

    graph: Graph[GraphState, Settings, EmergencyCall]
    deps: Settings
    name: str = "graph"

    async def start(self) -> list[BackendEvent]:
        """Run initial graph steps until question/completion."""
        return await self._run_turn(None)

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Advance graph execution with one caller message."""
        return await self._run_turn(text)

    async def close(self) -> None:
        """Flush session loggers."""
        flush_logger(self.deps.telemetry.messages_logger)
        flush_logger(self.deps.telemetry.state_logger)

    async def _run_turn(self, text: str | None) -> list[BackendEvent]:
        """Run graph steps until a question, completion, or error."""
        events: list[BackendEvent] = []
        next_text = text

        while result := await run_graph(
            self.graph,
            self.deps,
            next_text,
            on_complete=lambda state, message_history: record_completion_artifacts(
                self.deps,
                state,
                list(message_history),
            ),
        ):
            next_text = None
            if isinstance(result, End):
                events.append(
                    BackendEvent(
                        kind="completed",
                        text="The emergency call has been processed.",
                        payload=to_event_payload(result.data),
                    )
                )
                break

            if isinstance(result, QuestionNode):
                events.append(BackendEvent(kind="question", text=result.question))
                break

            if isinstance(result, MessageNode):
                message = result.messages.get(
                    self.deps.locale, "No message for this locale."
                )
                events.append(BackendEvent(kind="message", text=message))
                continue

            self.deps.telemetry.logger.warning(
                "Unexpected result type from run_graph: %s",
                type(result).__name__,
            )
            events.append(
                BackendEvent(
                    kind="error",
                    text=f"Unexpected graph result type: {type(result).__name__}",
                )
            )
            break

        return events


async def build_graph_policy(deps: Settings) -> GraphConversationPolicy:
    """Build one graph policy runtime."""
    await clear_old_run(deps.user_id, deps.storage.user_root)
    return GraphConversationPolicy(graph=await build_graph(), deps=deps)
