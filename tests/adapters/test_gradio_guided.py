"""Tests for the guided Gradio flow."""

from __future__ import annotations
from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr
import pytest

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.guided import (
    advance_to_next_scenario,
    build_initial_guided_state,
    capture_guided_user_submit,
    current_context_markdown,
    current_scenario_markdown,
    enter_guided_survey,
    prepare_guided_survey_state,
    reset_guided_flow,
    restart_current_call,
    save_guided_survey,
    start_guided_session,
    resolve_guided_scenarios,
    stream_guided_turn,
)
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    SessionHandle,
    SessionParameters,
    SessionState,
)


class _RecordingSessionManager:
    """Small async session manager stub that records guided flow operations."""

    def __init__(self) -> None:
        self.start_calls: list[SessionParameters] = []
        self.end_calls: list[UUID] = []
        self.survey_calls: list[dict[str, object]] = []

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        self.start_calls.append(request)
        user_id = request.user_id or uuid4()
        handle = SessionHandle(
            user_id=user_id,
            session_id=uuid4(),
            locale=request.locale,
            frontend_name=request.frontend_name,
            policy_name=request.policy_name,
            scenario_name=request.scenario_name,
        )
        return handle, []

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return []

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    async def end_session(self, session_id: UUID) -> bool:
        self.end_calls.append(session_id)
        return True

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        self.survey_calls.append(
            {
                "session_id": session_id,
                "responses": responses,
                "feedback": feedback,
                "metadata": metadata,
            }
        )
        return Path("survey.json")

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None



def make_args(
    tmp_path: Path,
    scenario_names: tuple[str, ...] = (
        "Scenario_01.md",
        "Scenario_02.md",
        "Scenario_03.md",
        "Scenario_04.md",
    ),
    *,
    explicit_guided_scenarios: tuple[str, ...] | None = None,
) -> GradioAppArgs:
    """Create guided-mode args with temp scenarios."""
    for scenario_name in scenario_names:
        (tmp_path / scenario_name).write_text(f"# {scenario_name}\n", encoding="utf-8")
    return GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=False,
        random_scenario=False,
        ui="guided",
        guided_scenarios=explicit_guided_scenarios or (),
    )



def final_state(*, session_id: str | None = None) -> dict[str, object]:
    """Build a final-scenario guided state for survey tests."""
    return {
        "user_id": "user-1",
        "scenario_names": ("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
        "active_index": 2,
        "phase": "complete",
        "survey_submitted": False,
        "active_session": {
            "user_id": "user-1",
            "session_id": session_id or str(uuid4()),
            "policy_name": "graph",
            "scenario_name": "Scenario_03.md",
            "experiment_name": "",
        },
        "completed_sessions": [
            {"scenario_name": "Scenario_01.md", "session_id": str(uuid4())},
            {"scenario_name": "Scenario_02.md", "session_id": str(uuid4())},
        ],
    }



def test_resolve_guided_scenarios_uses_directory_when_cli_list_is_empty(
    tmp_path: Path,
) -> None:
    """Guided mode should expand to all directory scenarios when none are specified."""
    args = make_args(tmp_path)

    assert resolve_guided_scenarios(args) == (
        "Scenario_01.md",
        "Scenario_02.md",
        "Scenario_03.md",
        "Scenario_04.md",
    )



def test_build_initial_guided_state_starts_on_first_scenario() -> None:
    """The guided flow should begin on the first scenario with no active call."""
    state = build_initial_guided_state(("Scenario_01.md", "Scenario_02.md"))

    assert state["active_index"] == 0
    assert state["phase"] == "brief"
    assert state["survey_submitted"] is False
    assert state["active_session"] is None
    assert state["completed_sessions"] == []


def test_capture_guided_user_submit_appends_only_user_message() -> None:
    """Submitting a guided chat turn should immediately add only the user bubble."""
    (
        normalized_message,
        updated_history,
        turn_status,
        _input_update,
        _send_update,
        _restart_update,
        _status_update,
    ) = capture_guided_user_submit(" Hello ", [])

    assert normalized_message == "Hello"
    assert turn_status == "stream"
    assert len(updated_history) == 1
    assert updated_history[0].content == "Hello"


@pytest.mark.asyncio
async def test_stream_guided_turn_streams_assistant_message_without_manual_placeholder() -> None:
    """Guided streaming should let Gradio own the pending indicator, then stream output."""
    manager = _RecordingSessionManager()

    async def _handle_input(session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return [BackendEvent(kind="message", text="Hello there")]

    manager.handle_input = _handle_input  # type: ignore[method-assign]
    state = build_initial_guided_state(("Scenario_01.md",))
    state["user_id"] = "user-1"
    state["phase"] = "call"
    state["active_session"] = {
        "user_id": "user-1",
        "session_id": str(uuid4()),
        "policy_name": "graph",
        "scenario_name": "Scenario_01.md",
        "experiment_name": "",
    }

    snapshots: list[tuple[list[str], bool, str]] = []
    async for history, completion, status in stream_guided_turn(
        [gr.ChatMessage(role="user", content="Hello")],
        "stream",
        "Hello",
        state,
        scenario_index=0,
        session_manager=manager,
    ):
        snapshots.append(([message.content for message in history], completion, status))

    assert snapshots[0] == (["Hello", "H"], False, "stream")
    assert snapshots[-1] == (["Hello", "Hello there"], False, "stream")


@pytest.mark.asyncio
async def test_start_guided_session_uses_requested_scenario_index(tmp_path: Path) -> None:
    """Guided session start should use the scenario index chosen by the outer walkthrough."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_02.md", "Scenario_04.md"),
    )
    manager = _RecordingSessionManager()
    state = build_initial_guided_state(resolve_guided_scenarios(args))

    updated_state, events, completion, user_id = await start_guided_session(
        state,
        "",
        scenario_index=1,
        session_manager=manager,
        args=args,
    )

    assert manager.start_calls[0].scenario_name == "Scenario_04.md"
    assert updated_state["active_index"] == 1
    assert updated_state["phase"] == "starting"
    assert updated_state["active_session"] is not None
    assert updated_state["active_session"]["scenario_name"] == "Scenario_04.md"
    assert updated_state["user_id"] == user_id
    assert completion is False
    assert events == []



def test_advance_to_next_scenario_records_completion_and_moves_forward(
    tmp_path: Path,
) -> None:
    """Advancing should record the current session and select the next scenario."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    session_id = uuid4()
    state = build_initial_guided_state(resolve_guided_scenarios(args))
    state["user_id"] = "user-1"
    state["phase"] = "complete"
    state["active_session"] = {
        "user_id": "user-1",
        "session_id": str(session_id),
        "policy_name": "graph",
        "scenario_name": "Scenario_01.md",
        "experiment_name": "",
    }

    updated_state = advance_to_next_scenario(state)

    assert updated_state["active_index"] == 1
    assert updated_state["phase"] == "brief"
    assert updated_state["active_session"] is None
    assert updated_state["completed_sessions"] == [
        {"scenario_name": "Scenario_01.md", "session_id": str(session_id)}
    ]
    assert "Scenario_02.md" in current_scenario_markdown(updated_state, args=args)



def test_prepare_guided_survey_state_records_final_session_once() -> None:
    """Entering the survey should append the final session metadata only once."""
    session_id = str(uuid4())
    state = final_state(session_id=session_id)

    updated_state = prepare_guided_survey_state(state)
    deduped_state = prepare_guided_survey_state(updated_state)

    assert len(updated_state["completed_sessions"]) == 3
    assert updated_state["completed_sessions"][-1] == {
        "scenario_name": "Scenario_03.md",
        "session_id": session_id,
    }
    assert deduped_state["completed_sessions"] == updated_state["completed_sessions"]



def test_enter_guided_survey_marks_survey_phase() -> None:
    """Final continue should move the run into the survey phase without clearing the session."""
    state = final_state()

    updated_state = enter_guided_survey(state)

    assert updated_state["phase"] == "survey"
    assert updated_state["active_session"] == state["active_session"]
    assert len(updated_state["completed_sessions"]) == 3


def test_current_context_markdown_switches_from_scenario_to_survey(tmp_path: Path) -> None:
    """The persistent context panel should swap from scenario text to survey help."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    (tmp_path / "Scenario_01.md").write_text(
        "## Situation Description\n\n- Example guided scenario\n",
        encoding="utf-8",
    )
    initial_state = build_initial_guided_state(resolve_guided_scenarios(args))

    scenario_context = current_context_markdown(initial_state, args=args)
    survey_context = current_context_markdown(enter_guided_survey(final_state()), args=args)

    assert "Situation Description" in scenario_context
    assert "Final Survey" in survey_context
    assert "Please answer the survey for the full guided run" in survey_context


def test_restart_current_call_preserves_completed_progress(tmp_path: Path) -> None:
    """Restarting a call should only clear the active session, not prior progress."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    session_id = uuid4()
    completed_session_id = str(uuid4())
    state = {
        "user_id": "user-1",
        "scenario_names": resolve_guided_scenarios(args),
        "active_index": 1,
        "phase": "call",
        "survey_submitted": False,
        "active_session": {
            "user_id": "user-1",
            "session_id": str(session_id),
            "policy_name": "graph",
            "scenario_name": "Scenario_02.md",
            "experiment_name": "",
        },
        "completed_sessions": [
            {"scenario_name": "Scenario_01.md", "session_id": completed_session_id},
        ],
    }

    updated_state = restart_current_call(state)

    assert updated_state["active_index"] == 1
    assert updated_state["phase"] == "brief"
    assert updated_state["completed_sessions"] == [
        {"scenario_name": "Scenario_01.md", "session_id": completed_session_id}
    ]
    assert updated_state["active_session"] is None


@pytest.mark.asyncio
async def test_save_guided_survey_includes_aggregate_metadata(tmp_path: Path) -> None:
    """Guided survey submission should include all scenario IDs in metadata."""
    _ = make_args(tmp_path)
    manager = _RecordingSessionManager()
    final_session_id = str(uuid4())
    completed_sessions = [
        {"scenario_name": "Scenario_01.md", "session_id": str(uuid4())},
        {"scenario_name": "Scenario_02.md", "session_id": str(uuid4())},
        {"scenario_name": "Scenario_03.md", "session_id": final_session_id},
    ]
    state = {
        "user_id": "user-1",
        "scenario_names": ("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
        "active_index": 2,
        "phase": "survey",
        "survey_submitted": False,
        "active_session": {
            "user_id": "user-1",
            "session_id": final_session_id,
            "policy_name": "graph",
            "scenario_name": "Scenario_03.md",
            "experiment_name": "",
        },
        "completed_sessions": completed_sessions,
    }

    result = await save_guided_survey(
        state,
        "  useful session  ",
        *([5] * len(DEFAULT_SURVEY.questions)),
        session_manager=manager,
    )

    assert manager.survey_calls
    saved_call = manager.survey_calls[0]
    assert saved_call["feedback"] == "useful session"
    assert saved_call["metadata"] == {
        "ui_mode": "guided",
        "scenario_names": ["Scenario_01.md", "Scenario_02.md", "Scenario_03.md"],
        "session_ids": [
            completed_session["session_id"] for completed_session in completed_sessions
        ],
        "scenario_count": 3,
    }
    assert result[0]["survey_submitted"] is True
    assert any(
        isinstance(update, dict) and update.get("visible") is True
        for update in result[1:]
    )



def test_reset_guided_flow_returns_to_first_intro(tmp_path: Path) -> None:
    """Full guided reset should clear all progress and restore the first scenario."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    state = {
        "user_id": "user-1",
        "scenario_names": resolve_guided_scenarios(args),
        "active_index": 2,
        "phase": "survey",
        "survey_submitted": True,
        "active_session": {
            "user_id": "user-1",
            "session_id": str(uuid4()),
            "policy_name": "graph",
            "scenario_name": "Scenario_03.md",
            "experiment_name": "",
        },
        "completed_sessions": [
            {"scenario_name": "Scenario_01.md", "session_id": str(uuid4())},
            {"scenario_name": "Scenario_02.md", "session_id": str(uuid4())},
        ],
    }

    updated_state = reset_guided_flow(state, "user-1")

    assert updated_state["active_index"] == 0
    assert updated_state["phase"] == "brief"
    assert updated_state["survey_submitted"] is False
    assert updated_state["active_session"] is None
    assert updated_state["completed_sessions"] == []
    assert "Scenario_01.md" in current_scenario_markdown(updated_state, args=args)



def test_build_scenario_intro_markdown_avoids_duplicate_progress_copy() -> None:
    """Intro copy should rely on the outer walkthrough instead of repeating scenario counters."""
    from ems_prepared.adapters.gradio.guided import build_scenario_intro_markdown

    intro = build_scenario_intro_markdown(0, 3)

    assert "Scenario 1 of 3" not in intro
    assert "Ready to Start" in intro
