"""Tests for CLI adapter argument parsing."""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID, uuid4

from ems_prepared.adapters.cli import app as cli_app
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    SessionHandle,
    SessionParameters,
    SessionState,
)


def test_register_arguments_accepts_no_frontend_specific_options() -> None:
    """CLI frontend plugin should not register extra frontend-specific options."""
    plugin = cli_app.CliFrontend()
    parser = argparse.ArgumentParser()
    plugin.register_arguments(parser)
    args = parser.parse_args([])
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
            [BackendEvent(kind="question", text="Where are you?")],
        )

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    async def end_session(self, session_id: UUID) -> bool:
        self.ended = session_id == self.session_id
        return self.ended

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> Path:
        _ = (session_id, responses, metadata)
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
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
    assert manager.ended is True
