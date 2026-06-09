"""Tests for CLI adapter argument parsing."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID, uuid4

from ems_prepared.adapters.cli import app as cli_app
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    MessageRole,
    MessageType,
    SessionHandle,
    SessionHistory,
    SessionParameters,
    SessionState,
    SessionStatus,
)


def test_register_arguments_accepts_resume_session_id() -> None:
    """CLI frontend plugin should register the explicit resume session flag."""
    plugin = cli_app.CliFrontend()
    parser = argparse.ArgumentParser()
    plugin.register_arguments(parser)
    args = parser.parse_args([])
    assert args.session_id is None
    assert not hasattr(args, "scenario_name")


class _InterruptingManager:
    """Session manager fake that returns one prompt, then supports close."""

    def __init__(self) -> None:
        self.session_id = uuid4()
        self.ended = False

    async def start_session(
        self, request: SessionParameters
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        return (
            SessionHandle(
                user_id=request.user_id or uuid4(),
                session_id=self.session_id,
                locale=request.locale,
                frontend_name=request.frontend_name,
                policy_name=request.policy_name,
                scenario_name=request.scenario_name,
            ),
            [BackendEvent(kind=BackendEventKind.QUESTION, text="Where are you?")],
        )

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        _ = (session_id, responses, feedback, metadata)
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        _ = session_id
        return None


def test_run_terminal_session_exits_on_keyboard_interrupt(monkeypatch) -> None:
    """One Ctrl-C during prompt input should exit with code 130."""
    manager = _InterruptingManager()
    args = argparse.Namespace(
        policy="graph",
        locale=Locale.EN.value,
        experiment_name=None,
    )

    def _raise_interrupt(_question: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_app.Prompt, "ask", _raise_interrupt)

    code = cli_app.run_terminal_session(manager, args)

    assert code == 130
    assert manager.ended is False


class _ResumeManager:
    """Session manager fake that supports resuming an existing session."""

    def __init__(self) -> None:
        self.session_id = UUID(int=7)
        self.start_calls = 0
        self.handled: list[tuple[UUID, str]] = []

    async def start_session(
        self, request: SessionParameters
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        self.start_calls += 1
        raise AssertionError(f"start_session should not be called: {request}")

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        self.handled.append((session_id, text))
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        _ = (session_id, responses, feedback, metadata)
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        if session_id != self.session_id:
            return None
        return SessionHistory(
            handle=SessionHandle(
                user_id=UUID(int=1),
                session_id=self.session_id,
                locale=Locale.EN,
                frontend_name="cli",
                policy_name="graph",
                scenario_name=None,
            ),
            experiment_name="",
            status=SessionStatus.ACTIVE,
            created_at="2026-01-01T00:00:00",
            messages=[
                ConversationMessage(
                    id="msg_1",
                    type=MessageType.QUESTION,
                    role=MessageRole.ASSISTANT,
                    content="Where are you?",
                )
            ],
        )


def test_run_terminal_session_resumes_active_integer_session(monkeypatch) -> None:
    """CLI resume should reuse the pending question for an existing session."""
    manager = _ResumeManager()
    args = argparse.Namespace(
        policy="graph",
        locale=Locale.EN.value,
        experiment_name=None,
        session_id="7",
    )

    monkeypatch.setattr(cli_app.Prompt, "ask", lambda _question: "At home")

    code = cli_app.run_terminal_session(manager, args)

    assert code == 0
    assert manager.start_calls == 0
    assert manager.handled == [(UUID(int=7), "At home")]


def test_render_events_prints_message_before_completion(capsys) -> None:
    """CLI rendering should show final handover text before the completion line."""
    question, stop, code = cli_app._render_events(
        [
            BackendEvent(
                kind=BackendEventKind.MESSAGE,
                text="Please hand over to the paramedics",
            ),
            BackendEvent(
                kind=BackendEventKind.COMPLETED,
                text="The emergency call has been processed.",
            ),
        ]
    )

    output = capsys.readouterr().out

    assert question is None
    assert stop is True
    assert code == 0
    assert "Please hand over to the paramedics" in output
    assert "The emergency call has been processed." in output
