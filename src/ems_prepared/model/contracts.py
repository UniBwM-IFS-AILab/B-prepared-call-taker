"""Shared backend contracts with a minimal abstraction surface."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from ems_prepared.model.context import (
    InputMode,
    Locale,
    RequestInputCallable,
    Settings,
)


class BackendEventKind(StrEnum):
    """Supported transport-neutral backend event kinds."""

    MESSAGE = "message"
    QUESTION = "question"
    COMPLETED = "completed"
    ERROR = "error"


class MessageRole(StrEnum):
    """Supported visible chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(StrEnum):
    """Supported visible chat message types."""

    MESSAGE = "message"
    QUESTION = "question"
    COMPLETION = "completion"
    ERROR = "error"


class SessionStatus(StrEnum):
    """Supported session lifecycle states."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ENDED = "ended"


_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


def _coerce_str_enum(
    enum_type: type[_StrEnumT],
    value: _StrEnumT,
    *,
    error_prefix: str,
) -> _StrEnumT:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{error_prefix}: {value}") from exc


class SessionResumeError(RuntimeError):
    """Raised when an existing session cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class BackendEvent:
    """Transport-neutral output emitted by the shared backend."""

    kind: BackendEventKind
    text: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject unsupported event kinds to keep adapter behavior stable."""
        object.__setattr__(
            self,
            "kind",
            _coerce_str_enum(
                BackendEventKind,
                self.kind,
                error_prefix="Unsupported backend event kind",
            ),
        )


@dataclass(frozen=True, slots=True)
class SessionState:
    """Lightweight view state for frontend adapters."""

    handle: SessionHandle
    experiment_name: str
    save_path: Path
    is_complete: bool

@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One user-visible chat message persisted for client retrieval."""

    id: str
    type: MessageType
    role: MessageRole
    content: str
    timestamp: str | None = None
    result: dict[str, Any] | None = None
    code: str | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "type",
            _coerce_str_enum(
                MessageType,
                self.type,
                error_prefix="Unsupported message type",
            ),
        )
        object.__setattr__(
            self,
            "role",
            _coerce_str_enum(
                MessageRole,
                self.role,
                error_prefix="Unsupported message role",
            ),
        )


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Persisted session metadata resolved from a storage backend."""

    handle: SessionHandle
    experiment_name: str
    save_path: Path
    status: SessionStatus

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _coerce_str_enum(
                SessionStatus,
                self.status,
                error_prefix="Unsupported session status",
            ),
        )


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """Visible chat history returned to HTTP and websocket clients."""

    handle: SessionHandle
    experiment_name: str
    status: SessionStatus
    messages: list[ConversationMessage]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _coerce_str_enum(
                SessionStatus,
                self.status,
                error_prefix="Unsupported session status",
            ),
        )


@dataclass(frozen=True, slots=True)
class MessageFeedback:
    """User feedback on one chat message."""

    role: MessageRole
    content: str
    value: str
    message_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role",
            _coerce_str_enum(
                MessageRole,
                self.role,
                error_prefix="Unsupported message role",
            ),
        )


# FIXME: why SessionHandle and SessionParameters?
# TODO: policy_name should be an enum of available ones.
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
class PolicyFactory(Protocol):
    """Factory callable that creates one policy runtime for a session deps object."""

    def __call__(self, deps: Settings, /) -> Awaitable[ConversationPolicy]:
        """Build one policy runtime for the given settings/deps."""
        ...


@runtime_checkable
class FrontendPlugin(Protocol):
    """Frontend plugin contract used by launcher composition."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register frontend-specific arguments onto one subparser."""
        ...

    def run(
        self,
        session_manager: "SessionManager",
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run one frontend and return a process exit code."""
        ...


@runtime_checkable
class SessionBackend(Protocol):
    """Storage-backed session lifecycle + logging contract."""

    def create_session(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        experiment_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Persist a newly started session."""
        ...

    def get_session(self, session_id: UUID) -> SessionRecord | None:
        """Resolve one session from storage."""
        ...

    def set_status(self, session_id: UUID, status: SessionStatus) -> bool:
        """Update lifecycle status for one session."""
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

    def save_transcript_entries(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        messages: Sequence[ConversationMessage],
    ) -> None:
        """Append user-visible chat messages for one session."""
        ...

    def load_history(self, session_id: UUID) -> list[ConversationMessage]:
        """Load ordered user-visible chat messages for one session."""
        ...

    def save_survey(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        responses: list[dict[str, Any]],
        feedback: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write one survey response payload and return the output location."""
        ...

    def save_feedback(
        self,
        *,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        """Persist one like/dislike-style feedback event."""
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
        """Write final state/history/deps artifacts."""
        ...

    def load_agent_snapshot(self, session_id: UUID) -> dict[str, Any] | None:
        """Load persisted agent runtime snapshot for a session."""
        ...

    def save_agent_snapshot(
        self, session_id: UUID, snapshot: Mapping[str, Any]
    ) -> None:
        """Persist agent runtime snapshot for a session."""
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

    async def end_session(self, session_id: UUID) -> bool:
        """Flush and remove an existing session."""
        ...

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, Any]],
        feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write survey responses for one session."""
        ...

    async def submit_feedback(
        self,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        """Persist one UI feedback event for a message."""
        ...

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        """Return lightweight view state for frontend display."""
        ...

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        """Return ordered visible history and session metadata."""
        ...
