"""Agent policy runtime adapter and builder."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessagesTypeAdapter

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationPolicy,
    SessionResumeError,
)
from ems_prepared.policies.llm_only.agent import (
    AgentPolicy,
    build_emergency_agent,
    extract_last_model_response,
    run_agent_with_capture,
    update_history_and_merge_state,
)
from ems_prepared.policies.shared import (
    record_completion_artifacts,
    to_event_payload,
)
from ems_prepared.util.logger import flush_logger


@dataclass(slots=True)
class AgentConversationPolicy(ConversationPolicy):
    """Adapter from the LLM-only loop to `BackendEvent`."""

    policy: AgentPolicy
    deps: Settings
    name: str = "agent"

    async def start(self) -> list[BackendEvent]:
        """Run the initial agent turn."""
        return await self._run_turn(None)

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Advance the agent loop with one caller message."""
        return await self._run_turn(text)

    async def close(self) -> None:
        """Flush session loggers."""
        flush_logger(self.deps.telemetry.messages_logger)
        flush_logger(self.deps.telemetry.state_logger)

    async def _run_turn(self, text: str | None) -> list[BackendEvent]:
        """Run one agent turn and translate it into backend events."""
        agent = self.policy.agent
        state = self.policy.state
        agent_history = self.policy.history

        result, captured_messages = await run_agent_with_capture(
            agent,
            agent_history,
            text,
            self.deps,
        )
        if text:
            self.deps.telemetry.messages_logger.info(
                "",
                extra={"speaker": "caller", "msg_text": text},
            )
        self.deps.telemetry.logger.info(
            "agent_history_len=%s usage=%s",
            len(agent_history),
            result.usage(),
        )

        response = extract_last_model_response(result, captured_messages)
        if response is not None:
            self.deps.telemetry.logger.info(
                "provider=%s model=%s",
                response.provider_name,
                response.model_name,
            )
        else:
            self.deps.telemetry.logger.warning("No ModelResponse found for this run.")

        is_complete, next_question = update_history_and_merge_state(
            agent_history,
            result,
            state,
            self.deps,
            caller_msg=text,
        )
        self._persist_snapshot()

        if is_complete:
            self.deps.telemetry.logger.info("Outcome reached")
            record_completion_artifacts(self.deps, state, agent_history)
            flush_logger(self.deps.telemetry.messages_logger)
            flush_logger(self.deps.telemetry.state_logger)
            return [
                BackendEvent(
                    kind=BackendEventKind.COMPLETED,
                    text="The emergency call has been processed.",
                    payload=to_event_payload(state),
                )
            ]

        if not next_question or not next_question.strip():
            self.deps.telemetry.logger.error(
                "Agent returned neither completion nor next_question. "
                "state=%r, next_question=%r, len(agent_history)=%s",
                result.output.state,
                next_question,
                len(agent_history),
            )
            self.deps.telemetry.logger.error("Full agent result: %r", result)
            return [
                BackendEvent(
                    kind=BackendEventKind.ERROR,
                    text=(
                        "Internal error: agent returned no next question. "
                        "Please reset the session."
                    ),
                )
            ]

        self.deps.telemetry.logger.debug(next_question)
        return [BackendEvent(kind=BackendEventKind.QUESTION, text=next_question)]

    def _persist_snapshot(self) -> None:
        if self.deps.save_agent_snapshot is None:
            return
        snapshot = {
            "state": self.policy.state.model_dump(mode="json"),
            "history": ModelMessagesTypeAdapter.dump_python(
                self.policy.history,
                mode="json",
            ),
        }
        self.deps.save_agent_snapshot(snapshot)


async def build_agent_policy(deps: Settings) -> AgentConversationPolicy:
    """Build one agent policy runtime."""
    state = EmergencyCall()
    history: list = []
    snapshot: dict | None = None
    if deps.load_agent_snapshot is not None:
        snapshot = deps.load_agent_snapshot()

    if deps.resume_expected and snapshot is None:
        raise SessionResumeError(
            f"Missing agent runtime snapshot for session: {deps.session_id}"
        )

    if snapshot is not None:
        if not isinstance(snapshot, dict):
            raise SessionResumeError(
                f"Invalid agent runtime snapshot for session: {deps.session_id}"
            )
        try:
            raw_state = snapshot.get("state")
            raw_history = snapshot.get("history")
            if raw_state is not None:
                if not isinstance(raw_state, dict):
                    raise TypeError("snapshot.state must be a mapping")
                state = EmergencyCall.model_validate(raw_state)
            if raw_history is not None:
                history = list(ModelMessagesTypeAdapter.validate_python(raw_history))
        except Exception as exc:
            raise SessionResumeError(
                f"Failed to restore agent runtime snapshot for session: {deps.session_id}"
            ) from exc

    return AgentConversationPolicy(
        policy=AgentPolicy(
            agent=build_emergency_agent(deps),
            state=state,
            history=history,
            deps=deps,
        ),
        deps=deps,
    )
