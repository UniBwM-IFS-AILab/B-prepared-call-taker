"""Tests for the additive shared `SessionService`."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import pytest

from ems_prepared.model.context import InputMode, Locale, Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    ConversationPolicy,
    MessageRole,
    MessageType,
    SessionParameters,
    SessionResumeError,
)
from ems_prepared.model.session_service import SessionService
from tests.fakes import FakeSessionBackend


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
    """Create a policy factory bound to one fake policy instance."""

    async def factory(deps: Settings) -> ConversationPolicy:
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
        start_events=[BackendEvent(kind=BackendEventKind.MESSAGE, text="hello caller")]
    )
    session_backend = FakeSessionBackend()
    service = SessionService(
        policy_factories={"fake": build_fake_factory(policy)},
        session_backend=session_backend,
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
    history = service.get_history(handle.session_id)

    assert events == [
        BackendEvent(kind=BackendEventKind.MESSAGE, text="hello caller")
    ]
    assert handle.user_id == UUID(int=1)
    assert handle.session_id == UUID(int=2)
    assert handle.locale is Locale.DE
    assert state is not None
    assert state.handle == handle
    assert state.experiment_name == "exp"
    assert state.save_path.exists()
    assert state.is_complete is False
    assert history is not None
    assert history.messages == [
        ConversationMessage(
            id=history.messages[0].id,
            type=MessageType.MESSAGE,
            role=MessageRole.ASSISTANT,
            content="hello caller",
            timestamp=history.messages[0].timestamp,
        )
    ]
    assert len(session_backend.manifests) == 1
    assert len(session_backend.events) == 1


@pytest.mark.asyncio
async def test_handle_input_marks_completion_and_stops_future_turns(
    monkeypatch, tmp_path
) -> None:
    """A completed session should stop advancing and remain completed."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(
        start_events=[BackendEvent(kind=BackendEventKind.QUESTION, text="where?")],
        replies={
            "at home": [
                BackendEvent(
                    kind=BackendEventKind.COMPLETED,
                    text="The emergency call has been processed.",
                )
            ]
        },
    )
    session_backend = FakeSessionBackend()
    service = SessionService(
        policy_factories={"fake": build_fake_factory(policy)},
        session_backend=session_backend,
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
    state = service.get_view_state(handle.session_id)
    history = service.get_history(handle.session_id)

    assert events == [
        BackendEvent(
            kind=BackendEventKind.COMPLETED,
            text="The emergency call has been processed.",
        )
    ]
    assert repeated == []
    assert state is not None
    assert state.is_complete is True
    assert history is not None
    assert [entry.role for entry in history.messages] == [
        "assistant",
        "user",
        "assistant",
    ]
    assert [entry.content for entry in history.messages] == [
        "where?",
        "at home",
        "The emergency call has been processed.",
    ]
    assert policy.close_calls == 2
    assert len(session_backend.events) == 2


@pytest.mark.asyncio
async def test_submit_survey_writes_to_session_directory(monkeypatch, tmp_path) -> None:
    """Survey recording should target the active session directory."""
    monkeypatch.chdir(tmp_path)
    policy = FakeConversationPolicy(start_events=[])
    session_backend = FakeSessionBackend()
    service = SessionService(
        policy_factories={"fake": build_fake_factory(policy)},
        session_backend=session_backend,
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
    assert len(session_backend.surveys) == 1


@pytest.mark.asyncio
async def test_handle_input_rebuilds_deps_with_inferred_origin(
    monkeypatch,
    tmp_path,
) -> None:
    """Existing-session turns should rebuild deps with inferred origin."""
    monkeypatch.chdir(tmp_path)
    seen_call_origin: list[InputMode] = []
    policy = FakeConversationPolicy(
        start_events=[BackendEvent(kind=BackendEventKind.QUESTION, text="where?")]
    )
    session_backend = FakeSessionBackend()

    async def factory(deps: Settings) -> ConversationPolicy:
        seen_call_origin.append(deps.call_origin)
        return policy

    service = SessionService(
        policy_factories={"fake": factory},
        session_backend=session_backend,
    )

    handle, _ = await service.start_session(
        SessionParameters(
            frontend_name="fastapi_ws",
            policy_name="fake",
            user_id=UUID(int=1),
            session_id=UUID(int=5),
            call_origin=InputMode.API,
        )
    )

    _ = await service.handle_input(handle.session_id, "at home")

    assert seen_call_origin == [InputMode.API, InputMode.API]


@pytest.mark.asyncio
async def test_handle_input_propagates_resume_failures(monkeypatch, tmp_path) -> None:
    """Existing-session turns should fail loudly when policy restoration is impossible."""
    monkeypatch.chdir(tmp_path)
    session_backend = FakeSessionBackend()
    build_calls = 0

    async def factory(deps: Settings) -> ConversationPolicy:
        nonlocal build_calls
        build_calls += 1
        if build_calls > 1:
            raise SessionResumeError("missing runtime state")
        return FakeConversationPolicy(start_events=[])

    service = SessionService(
        policy_factories={"fake": factory},
        session_backend=session_backend,
    )

    handle, _ = await service.start_session(
        SessionParameters(
            frontend_name="fastapi",
            policy_name="fake",
            user_id=UUID(int=1),
            session_id=UUID(int=6),
            call_origin=InputMode.API,
        )
    )

    with pytest.raises(SessionResumeError):
        await service.handle_input(handle.session_id, "hello")

    assert build_calls == 2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_service_raises_for_unknown_policy(monkeypatch, tmp_path) -> None:
    """Unknown policies should fail before a session is created."""
    monkeypatch.chdir(tmp_path)
    service = SessionService(
        policy_factories={},
        session_backend=FakeSessionBackend(),
    )

    with pytest.raises(ValueError):
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
        policy_factories={"fake": build_fake_factory(policy)},
        session_backend=FakeSessionBackend(),
    )

    with pytest.raises(LookupError):
        await service.handle_input(UUID(int=999), "hello")
