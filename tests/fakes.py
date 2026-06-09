"""Minimal fakes used by backend-facing migration tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    ConversationPolicy,
    MessageFeedback,
    SessionBackend,
    SessionHandle,
    SessionRecord,
    SessionStatus,
)


class FakeConversationPolicy:
    """Small in-memory policy used to exercise the shared contracts."""

    name = "fake"

    def __init__(self) -> None:
        """Initialize runtime counters for assertions."""
        self.closed = 0

    async def start(self) -> list[BackendEvent]:
        """Emit the initial policy event."""
        return [BackendEvent(kind=BackendEventKind.MESSAGE, text="started")]

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Echo one caller message as a backend question event."""
        return [BackendEvent(kind=BackendEventKind.QUESTION, text=text)]

    async def close(self) -> None:
        """Record one close operation."""
        self.closed += 1


class FakeSessionBackend:
    """In-memory session backend fake with filesystem-compatible survey writes."""

    def __init__(self) -> None:
        """Initialize captured manifest/event/survey writes."""
        self.manifests: list[tuple[SessionHandle, Path, dict[str, Any]]] = []
        self.events: list[tuple[SessionHandle, Path, list[BackendEvent]]] = []
        self.surveys: list[tuple[SessionHandle, Path, list[dict[str, Any]]]] = []
        self.feedback: list[tuple[UUID, MessageFeedback]] = []
        self.sessions: dict[UUID, SessionRecord] = {}
        self.histories: dict[UUID, list[ConversationMessage]] = {}
        self.completions: list[
            tuple[SessionHandle, Path, Any, list[Any], dict[str, Any] | None]
        ] = []

    def create_session(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        experiment_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Capture session creation writes."""
        self.manifests.append((session, save_path, dict(metadata)))
        self.sessions[session.session_id] = SessionRecord(
            handle=session,
            experiment_name=experiment_name,
            save_path=save_path,
            status=SessionStatus.ACTIVE,
            created_at="2026-01-01T00:00:00",
        )
        self.histories.setdefault(session.session_id, [])

    def get_session(self, session_id: UUID) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def set_status(self, session_id: UUID, status: SessionStatus) -> bool:
        record = self.sessions.get(session_id)
        if record is None:
            return False
        self.sessions[session_id] = SessionRecord(
            handle=record.handle,
            experiment_name=record.experiment_name,
            save_path=record.save_path,
            status=status,
            created_at=record.created_at,
        )
        return True

    def save_events(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        events: Sequence[BackendEvent],
    ) -> None:
        """Capture event writes."""
        self.events.append((session, save_path, list(events)))

    def save_transcript_entries(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        messages: Sequence[ConversationMessage],
    ) -> None:
        """Capture visible history writes."""
        _ = save_path
        self.histories.setdefault(session.session_id, []).extend(list(messages))

    def load_history(self, session_id: UUID) -> list[ConversationMessage]:
        """Return visible history captured for one session."""
        return list(self.histories.get(session_id, []))

    def save_survey(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        responses: list[dict[str, Any]],
        feedback: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist survey payload as JSON and capture the write."""
        self.surveys.append((session, save_path, responses))
        output_path = save_path / "survey.json"
        payload: dict[str, Any] = {
            "responses": responses,
            "metadata": dict(metadata or {}),
        }
        if feedback is not None:
            payload["feedback"] = feedback
        output_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return output_path

    def save_completion_artifacts(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        state: Any,
        message_history: Sequence[Any],
        deps_payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Capture completion artifact writes."""
        self.completions.append(
            (
                session,
                save_path,
                state,
                list(message_history),
                dict(deps_payload) if deps_payload is not None else None,
            )
        )

    def save_feedback(
        self,
        *,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        self.feedback.append((session_id, feedback))

    def load_agent_snapshot(self, session_id: UUID) -> dict[str, Any] | None:
        _ = session_id
        return None

    def save_agent_snapshot(self, session_id: UUID, snapshot: Mapping[str, Any]) -> None:
        _ = (session_id, snapshot)

    # Backwards compatibility for old tests still using recorder naming.
    def save_manifest(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        self.create_session(
            session=session,
            save_path=save_path,
            experiment_name=str(metadata.get("experiment_name", "")),
            metadata=metadata,
        )


def assert_contract_runtime_types() -> None:
    """Ensure local fakes satisfy runtime-checkable contracts."""
    policy = FakeConversationPolicy()
    session_backend = FakeSessionBackend()

    assert isinstance(policy, ConversationPolicy)
    assert isinstance(session_backend, SessionBackend)


FakeSessionRecorder = FakeSessionBackend
