"""Smoke tests for Gradio adapter wiring."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr

from ems_prepared.adapters.gradio.app import build_demo
from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    SessionHandle,
    SessionParameters,
    SessionState,
)


class _FakeSessionManager:
    """Minimal async session manager stub for UI build-time wiring tests."""

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        return (
            SessionHandle(
                user_id=request.user_id or uuid4(),
                session_id=request.session_id or uuid4(),
                locale=request.locale,
                frontend_name=request.frontend_name,
                policy_name=request.policy_name,
                scenario_name=request.scenario_name,
            ),
            [],
        )

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        return None

    async def end_session(self, session_id: UUID) -> bool:
        return True

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> Path:
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        return None


def test_build_demo_wires_events_inside_blocks_context(tmp_path: Path) -> None:
    """`build_demo` should not raise context errors when wiring events."""
    (tmp_path / "Scenario_01.md").write_text("# Scenario 01\n", encoding="utf-8")

    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        experiment_name=None,
        exit_on_launch=False,
    )
    demo = build_demo(session_manager=_FakeSessionManager(), args=args)

    assert isinstance(demo, gr.Blocks)
    assert demo.mode == "blocks"
    assert isinstance(args.resolve_policy(), str)
    assert args.is_debug_enabled() is False
    assert args.picker_interactive is True
    assert Locale.EN.value == "english"
