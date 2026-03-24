"""Minimal fakes used by backend-facing migration tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ems_prepared.model.contracts import (
    BackendEvent,
    ConversationPolicy,
    SessionHandle,
    SessionRecorder,
)


class FakeConversationPolicy:
    """Small in-memory policy used to exercise the shared contracts."""

    name = "fake"

    def __init__(self) -> None:
        """Initialize runtime counters for assertions."""
        self.closed = 0

    async def start(self) -> list[BackendEvent]:
        """Emit the initial policy event."""
        return [BackendEvent(kind="message", text="started")]

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Echo one caller message as a backend question event."""
        return [BackendEvent(kind="question", text=text)]

    async def close(self) -> None:
        """Record one close operation."""
        self.closed += 1


class FakeSessionRecorder:
    """In-memory session recorder fake with filesystem-compatible survey writes."""

    def __init__(self) -> None:
        """Initialize captured manifest/event/survey writes."""
        self.manifests: list[tuple[SessionHandle, Path, dict[str, Any]]] = []
        self.events: list[tuple[SessionHandle, Path, list[BackendEvent]]] = []
        self.surveys: list[tuple[SessionHandle, Path, list[dict[str, Any]]]] = []
        self.completions: list[
            tuple[SessionHandle, Path, Any, list[Any], dict[str, Any] | None]
        ] = []

    def save_manifest(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        """Capture manifest writes."""
        self.manifests.append((session, save_path, dict(metadata)))

    def save_events(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        events: Sequence[BackendEvent],
    ) -> None:
        """Capture event writes."""
        self.events.append((session, save_path, list(events)))

    def save_survey(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        responses: list[dict[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Persist survey payload as JSON and capture the write."""
        self.surveys.append((session, save_path, responses))
        output_path = save_path / "survey.json"
        payload = {
            "responses": responses,
            "metadata": dict(metadata or {}),
        }
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


def assert_contract_runtime_types() -> None:
    """Ensure local fakes satisfy runtime-checkable contracts."""
    policy = FakeConversationPolicy()
    session_recorder = FakeSessionRecorder()

    assert isinstance(policy, ConversationPolicy)
    assert isinstance(session_recorder, SessionRecorder)
