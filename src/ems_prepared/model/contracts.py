"""Shared backend contracts with a minimal abstraction surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, get_args, runtime_checkable
from uuid import UUID

from ems_prepared.model.context import InputMode, Locale, RequestInputCallable

PolicyName: TypeAlias = Literal["graph", "agent"]
BackendEventKind: TypeAlias = Literal["message", "question", "completed", "error"]


@dataclass(frozen=True, slots=True)
class BackendEvent:
    """Transport-neutral output emitted by the shared backend."""

    kind: BackendEventKind
    text: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject unsupported event kinds to keep adapter behavior stable."""
        if self.kind not in get_args(BackendEventKind):
            raise ValueError(f"Unsupported backend event kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class SessionState:
    """Lightweight in-memory session state for frontend adapters."""

    handle: SessionHandle
    experiment_name: str
    save_path: Path
    is_complete: bool


@dataclass(frozen=True, slots=True)
class SessionHandle:
    """Stable identifiers and routing metadata for a conversation session."""

    user_id: UUID
    session_id: UUID
    locale: Locale
    frontend_name: str
    policy_name: str | None = None
    scenario_name: str | None = None


@dataclass(frozen=True, slots=True)
class SessionParameters:
    """All inputs required to start one session."""

    frontend_name: str
    policy_name: str
    scenario_name: str | None = None
    user_id: UUID | None = None
    session_id: UUID | None = None
    locale: Locale = Locale.EN
    call_origin: InputMode = InputMode.API
    request_input: RequestInputCallable | None = None
    experiment_name: str | None = None


@runtime_checkable
class ConversationPolicy(Protocol):
    """Stateful policy runtime used by `SessionService`."""

    name: str

    async def start(self) -> list[BackendEvent]:
        """Start the policy and return initial events."""
        ...

    async def handle_input(self, text: str) -> list[BackendEvent]:
        """Handle one user message and return resulting events."""
        ...

    async def close(self) -> None:
        """Flush and release runtime resources."""
        ...


@runtime_checkable
class SessionRecorder(Protocol):
    """Session output recording contract for manifests, events, and surveys.

    Note: this is intentionally separate from policy-state persistence.
    Resumable policy persistence is implemented inside policy runtimes.
    """

    def save_manifest(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        """Write manifest metadata for one session."""
        ...

    def save_events(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        events: Sequence[BackendEvent],
    ) -> None:
        """Append backend events for one session."""
        ...

    def save_survey(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        responses: list[dict[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write one survey response payload and return the output file."""
        ...

    def save_completion_artifacts(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        state: Any,
        message_history: Sequence[Any],
        deps_payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Write final completion artifacts (state/history/deps payload)."""
        ...


@runtime_checkable
class SessionManager(Protocol):
    """Session lifecycle contract used by frontends."""

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        """Create a new session and return initial events."""
        ...

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        """Advance an existing session with one caller message."""
        ...

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        """Return in-memory state for an existing session."""
        ...

    async def end_session(self, session_id: UUID) -> bool:
        """Flush and remove an existing session."""
        ...

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write survey responses for one session."""
        ...

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        """Return lightweight in-memory state for frontend display."""
        ...
