"""Tests for the additive shared `SessionService`."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import pytest

from ems_prepared.model.context import InputMode, Locale, Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    ConversationPolicy,
    SessionParameters,
)
from ems_prepared.model.errors import SessionNotFoundError, UnsupportedPolicyError
from ems_prepared.model.session_service import SessionService
from ems_prepared.policies.runtime_shared import build_policy
from tests.fakes import FakeSessionRecorder


@dataclass
class FakeConversationPolicy:
    """Simple in-memory driver used to exercise `SessionService`."""

    start_events: list[BackendEvent]
    name: str = "fake"
    replies: dict[str, list[BackendEvent]] = field(default_factory=dict)
    close_calls: int = 0

    async def start(self) -> list[BackendEvent]:
        """Return the configured initial events."""
        return self.start_events

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Return the configured response for one caller message."""
        return self.replies.get(text, [])

    async def close(self) -> None:
        """Record one close call."""
        self.close_calls += 1


def build_fake_factory(policy: FakeConversationPolicy):
    """Create a policy-builder bound to one fake policy instance."""

    async def factory(policy_name: str, deps: Settings) -> ConversationPolicy:
        assert policy_name == "fake"
        assert deps.policy_name == "fake"
        return policy

    return factory


@pytest.mark.asyncio
async def test_start_session_returns_handle_and_initial_events(
    monkeypatch, tmp_path
) -> None:
    """Starting a session should register it and expose view state."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(
        start_events=[BackendEvent(kind="message", text="hello caller")]
    )
    session_recorder = FakeSessionRecorder()
    service = SessionService(
        policy_builder=build_fake_factory(policy),
        session_recorder=session_recorder,
    )

    handle, events = await service.start_session(
        SessionParameters(
            frontend_name="tests",
            policy_name="fake",
            scenario_name="demo.md",
            user_id=UUID(int=1),
            session_id=UUID(int=2),
            locale=Locale.DE,
            call_origin=InputMode.TEST,
            experiment_name="exp",
        )
    )

    state = service.get_view_state(handle.session_id)

    assert events == [BackendEvent(kind="message", text="hello caller")]
    assert handle.user_id == UUID(int=1)
    assert handle.session_id == UUID(int=2)
    assert handle.locale is Locale.DE
    assert state is not None
    assert state.handle == handle
    assert state.experiment_name == "exp"
    assert state.save_path.exists()
    assert state.is_complete is False
    assert len(session_recorder.manifests) == 1
    assert len(session_recorder.events) == 1


@pytest.mark.asyncio
async def test_handle_input_marks_completion_and_end_session_cleans_up(
    monkeypatch, tmp_path
) -> None:
    """A completed session should stop advancing and be removable."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(
        start_events=[BackendEvent(kind="question", text="where?")],
        replies={
            "at home": [
                BackendEvent(
                    kind="completed",
                    text="The emergency call has been processed.",
                )
            ]
        },
    )
    session_recorder = FakeSessionRecorder()
    service = SessionService(
        policy_builder=build_fake_factory(policy),
        session_recorder=session_recorder,
    )

    handle, _ = await service.start_session(
        SessionParameters(
            frontend_name="tests",
            policy_name="fake",
            user_id=UUID(int=1),
            session_id=UUID(int=3),
            call_origin=InputMode.TEST,
        )
    )

    events = await service.handle_input(handle.session_id, "at home")
    repeated = await service.handle_input(handle.session_id, "ignored")
    ended = await service.end_session(handle.session_id)

    assert events == [
        BackendEvent(
            kind="completed",
            text="The emergency call has been processed.",
        )
    ]
    assert repeated == []
    assert service.get_view_state(handle.session_id) is None
    assert policy.close_calls == 1
    assert ended is True
    assert len(session_recorder.events) == 2


@pytest.mark.asyncio
async def test_submit_survey_writes_to_session_directory(monkeypatch, tmp_path) -> None:
    """Survey recording should target the active session directory."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(start_events=[])
    session_recorder = FakeSessionRecorder()
    service = SessionService(
        policy_builder=build_fake_factory(policy),
        session_recorder=session_recorder,
    )

    handle, _ = await service.start_session(
        SessionParameters(
            frontend_name="tests",
            policy_name="fake",
            user_id=UUID(int=1),
            session_id=UUID(int=4),
            call_origin=InputMode.TEST,
        )
    )

    file_path = await service.submit_survey(
        handle.session_id,
        responses=[{"id": 1, "value": 5}],
        metadata={"source": "pytest"},
    )

    content = file_path.read_text(encoding="utf-8")

    assert file_path.name == "survey.json"
    assert '"source": "pytest"' in content
    assert len(session_recorder.surveys) == 1


@pytest.mark.asyncio
async def test_service_raises_for_unknown_policy(monkeypatch, tmp_path) -> None:
    """Unknown policies should fail before a session is created."""
    monkeypatch.chdir(tmp_path)
    service = SessionService(
        policy_builder=build_policy,
        session_recorder=FakeSessionRecorder(),
    )

    with pytest.raises(UnsupportedPolicyError):
        await service.start_session(
            SessionParameters(
                frontend_name="tests",
                policy_name="missing",
                call_origin=InputMode.TEST,
            )
        )


@pytest.mark.asyncio
async def test_service_raises_for_unknown_session(monkeypatch, tmp_path) -> None:
    """Unknown session ids should fail with a dedicated lookup error."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(start_events=[])
    service = SessionService(
        policy_builder=build_fake_factory(policy),
        session_recorder=FakeSessionRecorder(),
    )

    with pytest.raises(SessionNotFoundError):
        await service.handle_input(UUID(int=999), "hello")
