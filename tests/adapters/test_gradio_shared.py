"""Tests for shared Gradio runtime helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ems_prepared.adapters.gradio.args import GradioAppArgs
from ems_prepared.adapters.gradio.session_runtime import (
    DEBUG_STATIC_REPLY,
    end_active_session,
    start_session_for_gradio,
)
from ems_prepared.adapters.gradio.state import SessionRef
from ems_prepared.adapters.gradio.survey import (
    DEFAULT_SURVEY,
    build_survey_response_payload,
    submit_survey_for_gradio,
)
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    MessageFeedback,
    SessionHistory,
    SessionHandle,
    SessionParameters,
)


class _RecordingSessionManager:
    """Small session manager stub for shared helper tests."""

    def __init__(self) -> None:
        self.start_calls: list[SessionParameters] = []
        self.end_calls: list[UUID] = []
        self.survey_calls: list[dict[str, object]] = []
        self.feedback_calls: list[tuple[UUID, MessageFeedback]] = []
        self._initial_events: list[BackendEvent] = []
        self._handle_policy_name: str | None = None
        self._end_error: Exception | None = None
        self._survey_error: Exception | None = None

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        self.start_calls.append(request)
        handle = SessionHandle(
            user_id=request.user_id or uuid4(),
            session_id=request.session_id or uuid4(),
            locale=request.locale,
            frontend_name=request.frontend_name,
            policy_name=self._handle_policy_name,
            scenario_name=request.scenario_name,
        )
        return handle, list(self._initial_events)

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        _ = (session_id, text)
        return []

    async def resume_session(self, session_id: UUID):
        _ = session_id
        return None

    async def end_session(self, session_id: UUID) -> bool:
        self.end_calls.append(session_id)
        if self._end_error is not None:
            raise self._end_error
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
        if self._survey_error is not None:
            raise self._survey_error
        return Path("survey.json")

    async def submit_feedback(self, session_id: UUID, feedback: MessageFeedback) -> None:
        self.feedback_calls.append((session_id, feedback))

    def get_view_state(self, session_id: UUID):
        _ = session_id
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        _ = session_id
        return None


def make_args(*, debug: bool = False, verbose: bool = False) -> GradioAppArgs:
    """Create default Gradio args for shared helper tests."""
    return GradioAppArgs(
        scenario_dir="scenarios",
        user_id=0,
        policy="graph",
        locale=Locale.EN.value,
        debug=debug,
        random_scenario=False,
        verbose=verbose,
    )


@pytest.mark.asyncio
async def test_start_session_for_gradio_ends_previous_session_and_preserves_fields() -> None:
    """Shared start helper should end prior state and keep standard session fields."""
    manager = _RecordingSessionManager()
    stored_user_id = str(uuid4())
    previous_session = SessionRef(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        policy_name="graph",
        scenario_name="OldScenario.md",
        experiment_name="old-exp",
    )

    started = await start_session_for_gradio(
        scenario_name="Scenario_02.md",
        stored_user_id=stored_user_id,
        previous_session=previous_session,
        session_manager=manager,
        args=make_args(),
    )

    assert manager.end_calls == [UUID(previous_session.session_id)]
    assert manager.start_calls[0].scenario_name == "Scenario_02.md"
    assert manager.start_calls[0].policy_name == "graph"
    assert manager.start_calls[0].user_id == UUID(stored_user_id)
    assert started.user_id == stored_user_id
    assert started.session.user_id == stored_user_id
    assert started.session.policy_name == "graph"
    assert started.session.scenario_name == "Scenario_02.md"
    assert started.session.experiment_name == ""
    assert started.is_complete is False


@pytest.mark.asyncio
async def test_start_session_for_gradio_sets_completion_from_initial_events() -> None:
    """Shared start helper should surface initial completion emitted by backend."""
    manager = _RecordingSessionManager()
    manager._initial_events = [BackendEvent(kind=BackendEventKind.COMPLETED)]

    started = await start_session_for_gradio(
        scenario_name="Scenario_01.md",
        stored_user_id="",
        previous_session=None,
        session_manager=manager,
        args=make_args(),
    )

    assert started.events == [BackendEvent(kind=BackendEventKind.COMPLETED)]
    assert started.is_complete is True


@pytest.mark.asyncio
async def test_start_session_for_gradio_skips_policy_calls_in_debug_mode() -> None:
    """Debug mode should bypass session-manager start call."""
    manager = _RecordingSessionManager()

    started = await start_session_for_gradio(
        scenario_name="Scenario_01.md",
        stored_user_id="",
        previous_session=None,
        session_manager=manager,
        args=make_args(debug=True),
    )

    assert manager.start_calls == []
    assert started.events == [
        BackendEvent(kind=BackendEventKind.MESSAGE, text=DEBUG_STATIC_REPLY)
    ]
    assert started.is_complete is False


@pytest.mark.asyncio
async def test_start_session_for_gradio_reuses_requested_session_id() -> None:
    """Shared start helper should pass through an explicit requested session ID."""
    manager = _RecordingSessionManager()
    requested_session_id = str(uuid4())

    started = await start_session_for_gradio(
        scenario_name="Scenario_01.md",
        stored_user_id="",
        previous_session=None,
        requested_session_id=requested_session_id,
        session_manager=manager,
        args=make_args(),
    )

    assert manager.start_calls
    assert manager.start_calls[0].session_id == UUID(requested_session_id)
    assert started.session.session_id == requested_session_id


@pytest.mark.asyncio
async def test_end_active_session_noops_on_none_and_swallows_failures(caplog) -> None:
    """End helper should ignore missing sessions and log cleanup failures."""
    manager = _RecordingSessionManager()

    await end_active_session(None, session_manager=manager)

    await end_active_session(
        SessionRef(
            user_id=str(uuid4()),
            session_id=str(uuid4()),
            policy_name="graph",
            scenario_name="Scenario_01.md",
            experiment_name="",
        ),
        session_manager=manager,
        skip_policy_calls=True,
    )
    assert manager.end_calls == []

    manager._end_error = RuntimeError("boom")
    active_session = SessionRef(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        policy_name="graph",
        scenario_name="Scenario_01.md",
        experiment_name="",
    )
    with caplog.at_level("ERROR"):
        await end_active_session(active_session, session_manager=manager)

    assert manager.end_calls == [UUID(active_session.session_id)]
    assert f"Failed ending session {active_session.session_id}" in caplog.text


def test_build_survey_response_payload_preserves_current_schema() -> None:
    """Survey payload builder should preserve serialized shape."""
    responses = tuple(5 for _ in DEFAULT_SURVEY.questions)

    payload = build_survey_response_payload(DEFAULT_SURVEY, responses)

    assert len(payload) == len(DEFAULT_SURVEY.questions)
    assert payload[0] == {
        "label": DEFAULT_SURVEY.questions[0].label,
        "category": DEFAULT_SURVEY.questions[0].category,
        "question": DEFAULT_SURVEY.questions[0].text,
        "score": 5,
    }


@pytest.mark.asyncio
async def test_submit_survey_for_gradio_returns_expired_on_missing_session() -> None:
    """Survey submitter should map missing sessions to expired status."""
    manager = _RecordingSessionManager()
    manager._survey_error = LookupError("missing")

    status = await submit_survey_for_gradio(
        session_manager=manager,
        session_id=str(uuid4()),
        survey=DEFAULT_SURVEY,
        feedback="  helpful  ",
        responses=tuple(5 for _ in DEFAULT_SURVEY.questions),
    )

    assert status == "expired"
