"""Tests for the guided Gradio flow."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr
import pytest

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.flow import (
    advance_to_next_scenario,
    archive_guided_runtime_artifacts,
    begin_guided_start,
    build_initial_guided_state,
    build_scenario_intro_markdown,
    capture_guided_user_submit,
    current_context_markdown,
    current_scenario_markdown,
    enter_guided_survey,
    prepare_guided_survey_state,
    reset_guided_flow,
    resolve_guided_scenarios,
    restart_current_call,
    save_guided_survey,
    start_guided_session,
    stream_guided_turn,
)
from ems_prepared.adapters.gradio.scenarios import COMPLETION_MESSAGE
from ems_prepared.adapters.gradio.app import build_demo
from ems_prepared.adapters.gradio.session_runtime import DEBUG_STATIC_REPLY
from ems_prepared.adapters.gradio.state import CompletedScenario, GuidedState, SessionRef
from ems_prepared.adapters.gradio.survey import DEFAULT_SURVEY
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    MessageFeedback,
    SessionHistory,
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
        self.feedback_calls: list[tuple[UUID, MessageFeedback]] = []

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

    async def submit_feedback(self, session_id: UUID, feedback: MessageFeedback) -> None:
        self.feedback_calls.append((session_id, feedback))

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        _ = session_id
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        _ = session_id
        return None


class _ArchiveViewStateManager:
    """Minimal manager stub exposing only get_view_state for archive helper tests."""

    def __init__(self, save_path: Path, session_id: UUID) -> None:
        self._save_path = save_path
        self._session_id = session_id

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        if session_id != self._session_id:
            return None
        return SessionState(
            handle=SessionHandle(
                user_id=UUID(int=1),
                session_id=self._session_id,
                locale=Locale.EN,
                frontend_name="gradio",
                policy_name="graph",
                scenario_name="Scenario_01.md",
            ),
            experiment_name="",
            save_path=self._save_path,
            is_complete=False,
        )


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


def final_state(*, session_id: str | None = None) -> GuidedState:
    """Build final-scenario guided state for survey tests."""
    return GuidedState(
        user_id="user-1",
        scenario_names=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
        active_index=2,
        phase="complete",
        survey_submitted=False,
        active_session=SessionRef(
            user_id="user-1",
            session_id=session_id or str(uuid4()),
            policy_name="graph",
            scenario_name="Scenario_03.md",
            experiment_name="",
        ),
        completed_sessions=[
            CompletedScenario(scenario_name="Scenario_01.md"),
            CompletedScenario(scenario_name="Scenario_02.md"),
        ],
    )


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
    """Guided flow should begin on first scenario with no active call."""
    state = build_initial_guided_state(("Scenario_01.md", "Scenario_02.md"))

    assert state.active_index == 0
    assert state.phase == "brief"
    assert state.survey_submitted is False
    assert state.active_session is None
    assert state.completed_sessions == []


def test_build_initial_guided_state_uses_consent_phase_when_required() -> None:
    """Consent-enabled guided runs should begin on the consent step."""
    state = build_initial_guided_state(
        ("Scenario_01.md", "Scenario_02.md"),
        consent_required=True,
    )

    assert state.phase == "consent"
    assert state.consent_accepted is False


def test_guided_navigation_callbacks_update_tabs_without_queue(tmp_path: Path) -> None:
    """Guided navigation callbacks should drive shared Tabs navigation directly."""
    scenario_names = ("Scenario_01.md", "Scenario_02.md")
    for scenario_name in scenario_names:
        (tmp_path / scenario_name).write_text(f"# {scenario_name}\n", encoding="utf-8")

    args = GradioAppArgs(
        scenario_dir=str(tmp_path),
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=True,
        random_scenario=False,
        ui="guided",
        require_consent=True,
    )
    demo = build_demo(session_manager=_RecordingSessionManager(), args=args)
    walkthrough = next(
        block for block in demo.blocks.values() if isinstance(block, gr.Walkthrough)
    )
    navigation_dependencies = [
        dep
        for dep in demo.config["dependencies"]
        if dep.get("outputs")
        and walkthrough._id in dep["outputs"]
    ]

    assert navigation_dependencies
    assert all(dep["queue"] is False for dep in navigation_dependencies)


def test_capture_guided_user_submit_appends_only_user_message() -> None:
    """Submitting a guided chat turn should immediately add only user bubble."""
    (
        normalized_message,
        updated_history,
        _input_update,
        _send_update,
        _restart_update,
        _status_update,
    ) = capture_guided_user_submit(" Hello ", [])

    assert normalized_message == "Hello"
    assert len(updated_history) == 1
    assert updated_history[0].content == "Hello"


def test_begin_guided_start_marks_consent_once_checked() -> None:
    """First guided start should persist consent acceptance in run state."""
    state = build_initial_guided_state(("Scenario_01.md", "Scenario_02.md"))

    updated = begin_guided_start(state, scenario_index=0, consent_checked=True)

    assert updated.phase == "starting"
    assert updated.consent_accepted is True


@pytest.mark.asyncio
async def test_stream_guided_turn_streams_assistant_message() -> None:
    """Guided streaming should append assistant output incrementally."""
    manager = _RecordingSessionManager()

    async def _handle_input(session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return [BackendEvent(kind=BackendEventKind.MESSAGE, text="Hello there")]

    manager.handle_input = _handle_input  # type: ignore[method-assign]
    state = build_initial_guided_state(("Scenario_01.md",))
    state.user_id = "user-1"
    state.phase = "call"
    state.active_session = SessionRef(
        user_id="user-1",
        session_id=str(uuid4()),
        policy_name="graph",
        scenario_name="Scenario_01.md",
        experiment_name="",
    )

    snapshots: list[tuple[list[str], bool]] = []
    async for history, completion in stream_guided_turn(
        [gr.ChatMessage(role="user", content="Hello")],
        "Hello",
        state,
        scenario_index=0,
        session_manager=manager,
        skip_policy_calls=False,
    ):
        snapshots.append(([message.content for message in history], completion))

    assert snapshots[0] == (["Hello", "H"], False)
    assert snapshots[-1] == (["Hello", "Hello there"], False)


@pytest.mark.asyncio
async def test_stream_guided_turn_skips_policy_calls_in_debug_session() -> None:
    """Guided debug sessions should produce static reply and completion."""
    manager = _RecordingSessionManager()
    state = build_initial_guided_state(("Scenario_01.md",))
    state.user_id = "user-1"
    state.phase = "call"
    state.active_session = SessionRef(
        user_id="user-1",
        session_id=str(uuid4()),
        policy_name="graph",
        scenario_name="Scenario_01.md",
        experiment_name="",
    )

    snapshots: list[tuple[list[str], bool]] = []
    async for history, completion in stream_guided_turn(
        [gr.ChatMessage(role="user", content="Hello")],
        "Hello",
        state,
        scenario_index=0,
        session_manager=manager,
        skip_policy_calls=True,
    ):
        snapshots.append(([message.content for message in history], completion))

    assert snapshots
    assert snapshots[-1] == (
        ["Hello", DEBUG_STATIC_REPLY, COMPLETION_MESSAGE.content],
        True,
    )


@pytest.mark.asyncio
async def test_start_guided_session_uses_requested_scenario_index(tmp_path: Path) -> None:
    """Guided start should use scenario index selected by outer walkthrough."""
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
    assert updated_state.active_index == 1
    assert updated_state.phase == "starting"
    assert updated_state.run_session_id == updated_state.active_session.session_id
    assert updated_state.active_session is not None
    assert updated_state.active_session.scenario_name == "Scenario_04.md"
    assert updated_state.user_id == user_id
    assert completion is False
    assert events == []


@pytest.mark.asyncio
async def test_start_guided_session_reuses_existing_run_session_id(tmp_path: Path) -> None:
    """Guided start should reuse one stable run session ID across scenarios."""
    args = make_args(tmp_path)
    manager = _RecordingSessionManager()
    state = build_initial_guided_state(resolve_guided_scenarios(args))
    stable_session_id = str(uuid4())
    state.run_session_id = stable_session_id

    updated_state, _events, _completion, _user_id = await start_guided_session(
        state,
        "",
        scenario_index=0,
        session_manager=manager,
        args=args,
    )

    assert manager.start_calls
    assert manager.start_calls[0].session_id == UUID(stable_session_id)
    assert updated_state.run_session_id == stable_session_id


def test_advance_to_next_scenario_records_completion_and_moves_forward(
    tmp_path: Path,
) -> None:
    """Advancing should record current session and select next scenario."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    session_id = uuid4()
    state = build_initial_guided_state(resolve_guided_scenarios(args))
    state.user_id = "user-1"
    state.phase = "complete"
    state.active_session = SessionRef(
        user_id="user-1",
        session_id=str(session_id),
        policy_name="graph",
        scenario_name="Scenario_01.md",
        experiment_name="",
    )

    updated_state = advance_to_next_scenario(state)

    assert updated_state.active_index == 1
    assert updated_state.phase == "brief"
    assert updated_state.active_session is None
    assert updated_state.completed_sessions == [
        CompletedScenario(scenario_name="Scenario_01.md")
    ]
    assert "Scenario_02.md" in current_scenario_markdown(updated_state, args=args)


def test_prepare_guided_survey_state_records_final_session_once() -> None:
    """Entering survey should append final session metadata only once."""
    session_id = str(uuid4())
    state = final_state(session_id=session_id)

    updated_state = prepare_guided_survey_state(state)
    deduped_state = prepare_guided_survey_state(updated_state)

    assert len(updated_state.completed_sessions) == 3
    assert updated_state.completed_sessions[-1] == CompletedScenario(
        scenario_name="Scenario_03.md",
    )
    assert deduped_state.completed_sessions == updated_state.completed_sessions


def test_enter_guided_survey_marks_survey_phase() -> None:
    """Final continue should move run into survey phase without clearing session."""
    state = final_state()

    updated_state = enter_guided_survey(state)

    assert updated_state.phase == "survey"
    assert updated_state.active_session == state.active_session
    assert len(updated_state.completed_sessions) == 3


def test_current_context_markdown_switches_from_scenario_to_survey(tmp_path: Path) -> None:
    """Persistent context panel should swap from scenario text to survey help."""
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


def test_current_context_markdown_uses_consent_copy_before_first_scenario(
    tmp_path: Path,
) -> None:
    """Consent-enabled guided runs should show consent copy before scenario 1."""
    args = make_args(tmp_path)
    state = build_initial_guided_state(resolve_guided_scenarios(args), consent_required=True)

    context = current_context_markdown(state, args=args, consent_required=True)

    assert "Please review the consent information" in context


def test_restart_current_call_preserves_completed_progress(tmp_path: Path) -> None:
    """Restarting call should clear only active session, not prior progress."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    session_id = uuid4()
    state = GuidedState(
        user_id="user-1",
        scenario_names=resolve_guided_scenarios(args),
        active_index=1,
        phase="call",
        survey_submitted=False,
        active_session=SessionRef(
            user_id="user-1",
            session_id=str(session_id),
            policy_name="graph",
            scenario_name="Scenario_02.md",
            experiment_name="",
        ),
        completed_sessions=[
            CompletedScenario(scenario_name="Scenario_01.md")
        ],
    )

    updated_state = restart_current_call(state)

    assert updated_state.active_index == 1
    assert updated_state.phase == "brief"
    assert updated_state.completed_sessions == [
        CompletedScenario(scenario_name="Scenario_01.md")
    ]
    assert updated_state.active_session is None


@pytest.mark.asyncio
async def test_save_guided_survey_includes_aggregate_metadata(tmp_path: Path) -> None:
    """Guided survey submission should include scenario aggregation metadata."""
    _ = make_args(tmp_path)
    manager = _RecordingSessionManager()
    final_session_id = str(uuid4())
    completed_sessions = [
        CompletedScenario(scenario_name="Scenario_01.md"),
        CompletedScenario(scenario_name="Scenario_02.md"),
        CompletedScenario(scenario_name="Scenario_03.md"),
    ]
    state = GuidedState(
        user_id="user-1",
        scenario_names=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
        active_index=2,
        phase="survey",
        survey_submitted=False,
        active_session=SessionRef(
            user_id="user-1",
            session_id=final_session_id,
            policy_name="graph",
            scenario_name="Scenario_03.md",
            experiment_name="",
        ),
        completed_sessions=completed_sessions,
    )

    result = await save_guided_survey(
        state,
        "  useful session  ",
        *([5] * len(DEFAULT_SURVEY.questions)),
        session_manager=manager,
        skip_policy_calls=False,
    )

    assert manager.survey_calls
    saved_call = manager.survey_calls[0]
    assert saved_call["feedback"] == "useful session"
    assert saved_call["metadata"] == {
        "ui_mode": "guided",
        "scenario_names": ["Scenario_01.md", "Scenario_02.md", "Scenario_03.md"],
        "completed_scenarios": [item.scenario_name for item in completed_sessions],
        "completed_count": 3,
        "scenario_count": 3,
    }
    assert result[0].survey_submitted is True


@pytest.mark.asyncio
async def test_save_guided_survey_skips_backend_submit_in_debug_mode() -> None:
    """Guided debug sessions should complete survey UI without backend submit."""
    manager = _RecordingSessionManager()
    state = final_state()
    state.phase = "survey"

    result = await save_guided_survey(
        state,
        "debug feedback",
        *([5] * len(DEFAULT_SURVEY.questions)),
        session_manager=manager,
        skip_policy_calls=True,
    )

    assert manager.survey_calls == []
    assert result[0].survey_submitted is True


def test_reset_guided_flow_returns_to_first_intro(tmp_path: Path) -> None:
    """Full guided reset should clear progress and restore first scenario."""
    args = make_args(
        tmp_path,
        explicit_guided_scenarios=("Scenario_01.md", "Scenario_02.md", "Scenario_03.md"),
    )
    state = GuidedState(
        user_id="user-1",
        scenario_names=resolve_guided_scenarios(args),
        active_index=2,
        phase="survey",
        survey_submitted=True,
        active_session=SessionRef(
            user_id="user-1",
            session_id=str(uuid4()),
            policy_name="graph",
            scenario_name="Scenario_03.md",
            experiment_name="",
        ),
        completed_sessions=[
            CompletedScenario(scenario_name="Scenario_01.md"),
            CompletedScenario(scenario_name="Scenario_02.md"),
        ],
    )

    updated_state = reset_guided_flow(state, "user-1", consent_required=False)

    assert updated_state.active_index == 0
    assert updated_state.phase == "brief"
    assert updated_state.survey_submitted is False
    assert updated_state.consent_accepted is False
    assert updated_state.run_session_id is None
    assert updated_state.active_session is None
    assert updated_state.completed_sessions == []
    assert "Scenario_01.md" in current_scenario_markdown(updated_state, args=args)


def test_reset_guided_flow_returns_to_consent_when_required(tmp_path: Path) -> None:
    """Full guided reset should return to the consent step when enabled."""
    args = make_args(tmp_path)
    state = GuidedState(
        user_id="user-1",
        scenario_names=resolve_guided_scenarios(args),
        active_index=1,
        phase="survey",
        survey_submitted=True,
        consent_accepted=True,
    )

    updated_state = reset_guided_flow(state, "user-1", consent_required=True)

    assert updated_state.phase == "consent"
    assert updated_state.consent_accepted is False


def test_build_scenario_intro_markdown_avoids_duplicate_progress_copy() -> None:
    """Intro copy should rely on outer walkthrough instead of repeated counters."""
    intro = build_scenario_intro_markdown(0, 3)

    assert "Scenario 1 of 3" not in intro
    assert "Ready to Start" in intro


def test_archive_guided_runtime_artifacts_moves_runtime_files(tmp_path: Path) -> None:
    """Guided archive helper should move scenario runtime files into subfolders."""
    session_id = uuid4()
    save_path = tmp_path / "run"
    save_path.mkdir(parents=True, exist_ok=True)
    persistence_file = save_path / "main_persistence.json"
    persistence_file.write_text("{}", encoding="utf-8")
    state_file = save_path / "final_state.json"
    state_file.write_text("{}", encoding="utf-8")

    manager = _ArchiveViewStateManager(save_path, session_id)
    archive_guided_runtime_artifacts(
        session_manager=manager,  # type: ignore[arg-type]
        run_session_id=str(session_id),
        scenario_index=0,
        scenario_name="Scenario_01.md",
    )

    archived_dir = save_path / "scenarios" / "01_Scenario_01.md"
    assert (archived_dir / "main_persistence.json").exists()
    assert (archived_dir / "final_state.json").exists()
    assert not persistence_file.exists()
    assert not state_file.exists()
