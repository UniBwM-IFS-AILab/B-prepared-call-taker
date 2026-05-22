"""Contract-level tests for the shared backend abstractions."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import BackendEvent, BackendEventKind, SessionHandle
from tests.fakes import (
    FakeConversationPolicy,
    FakeSessionRecorder,
    assert_contract_runtime_types,
)


def test_backend_event_defaults_to_empty_payload() -> None:
    """`BackendEvent` should be easy to construct for message-only outputs."""
    event = BackendEvent(kind=BackendEventKind.MESSAGE, text="hello")

    assert event.payload == {}


def test_backend_event_rejects_unsupported_kind() -> None:
    """`BackendEvent` should reject event kinds outside the shared contract."""
    with pytest.raises(ValueError, match="Unsupported backend event kind"):
        _ = BackendEvent(kind="other")  # type: ignore[arg-type]


def test_runtime_checkable_protocols_accept_local_fakes() -> None:
    """The migration fakes should satisfy the declared contracts."""
    assert_contract_runtime_types()


@pytest.mark.asyncio
async def test_fake_policy_round_trip_uses_shared_contract_shapes() -> None:
    """A minimal fake engine should exercise the new backend-neutral types."""
    session = SessionHandle(
        user_id=UUID(int=1),
        session_id=UUID(int=2),
        locale=Locale.EN,
        frontend_name="tests",
        policy_name="fake",
        scenario_name="demo",
    )
    policy = FakeConversationPolicy()
    session_recorder = FakeSessionRecorder()

    start_events = await policy.start()
    reply_events = await policy.handle_input("where are you?")
    await policy.close()

    session_recorder.save_manifest(
        session=session,
        save_path=Path("."),
        metadata={"scenario": "demo"},
    )
    session_recorder.save_events(
        session=session,
        save_path=Path("."),
        events=[*start_events, *reply_events],
    )

    assert start_events == [BackendEvent(kind=BackendEventKind.MESSAGE, text="started")]
    assert reply_events == [
        BackendEvent(kind=BackendEventKind.QUESTION, text="where are you?")
    ]
    assert policy.closed == 1
    assert session_recorder.manifests[0][2]["scenario"] == "demo"
    assert len(session_recorder.events[0][2]) == 2
