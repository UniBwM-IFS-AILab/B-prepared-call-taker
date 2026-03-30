"""Shared backend entry point for session lifecycle management."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    ConversationPolicy,
    PolicyFactory,
    SessionHandle,
    SessionManager,
    SessionParameters,
    SessionRecorder,
    SessionState,
)
from ems_prepared.model.errors import SessionNotFoundError, UnsupportedPolicyError


@dataclass(slots=True)
class _SessionEntry:
    """In-memory runtime container owned by `SessionService`."""

    handle: SessionHandle
    deps: Settings
    policy: ConversationPolicy
    is_complete: bool = False


class SessionService(SessionManager):
    """Shared backend API for creating, driving, and ending sessions."""

    def __init__(
        self,
        *,
        policy_factories: Mapping[str, PolicyFactory],
        session_recorder: SessionRecorder,
    ) -> None:
        """Store configured policy factories and session output recorder."""
        self._policy_factories = dict(policy_factories)
        self._session_recorder = session_recorder
        self._sessions: dict[UUID, _SessionEntry] = {}

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        """Create and register a new in-memory session."""
        resolved_user_id = request.user_id or uuid4()
        resolved_session_id = request.session_id or uuid4()
        if resolved_session_id in self._sessions:
            raise ValueError(f"Session already exists: {resolved_session_id}")

        deps = Settings(
            name=f"{request.frontend_name}_{request.policy_name}",
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            experiment_name=request.experiment_name or "",
            scenario_name=request.scenario_name,
            policy_name=request.policy_name,
            locale=request.locale,
            call_origin=request.call_origin,
            request_input=request.request_input,
        )

        handle = SessionHandle(
            user_id=deps.user_id,
            session_id=deps.session_id,
            locale=deps.locale,
            frontend_name=request.frontend_name,
            policy_name=request.policy_name,
            scenario_name=request.scenario_name,
        )

        def _record_completion_artifacts(
            state: Any,
            message_history: list[Any],
        ) -> None:
            self._session_recorder.save_completion_artifacts(
                session=handle,
                save_path=deps.storage.save_path,
                state=state,
                message_history=message_history,
                deps_payload=deps.model_dump(
                    mode="json",
                    exclude={
                        "emit",
                        "request_input",
                        "record_completion_artifacts",
                    },
                ),
            )

        deps.record_completion_artifacts = _record_completion_artifacts
        policy = await self._build_policy(request.policy_name, deps)

        managed = _SessionEntry(handle=handle, deps=deps, policy=policy)
        self._sessions[handle.session_id] = managed

        self._session_recorder.save_manifest(
            session=handle,
            save_path=deps.storage.save_path,
            metadata={
                "experiment_name": deps.experiment_name,
                "save_path": str(deps.storage.save_path),
            },
        )

        try:
            events = await policy.start()
        except Exception:
            _ = self._sessions.pop(handle.session_id, None)
            raise

        self._session_recorder.save_events(
            session=handle,
            save_path=deps.storage.save_path,
            events=events,
        )
        managed.is_complete = self._has_completion(events)
        return handle, events

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        """Advance an existing session with one caller message."""
        session = self._require_session(session_id)
        if session.is_complete:
            return []

        events = await session.policy.handle_input(text)
        self._session_recorder.save_events(
            session=session.handle,
            save_path=session.deps.storage.save_path,
            events=events,
        )
        session.is_complete = self._has_completion(events)
        return events

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        """Return current in-memory session state."""
        return self.get_view_state(session_id)

    async def end_session(self, session_id: UUID) -> bool:
        """Flush and remove an existing session."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False

        await session.policy.close()
        return True

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, Any]],
        feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write survey responses into the current session directory."""
        session = self._require_session(session_id)
        return self._session_recorder.save_survey(
            session=session.handle,
            save_path=session.deps.storage.save_path,
            responses=responses,
            feedback=feedback,
            metadata=metadata,
        )

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        """Return a lightweight view of the current in-memory session."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        return SessionState(
            handle=session.handle,
            experiment_name=session.deps.experiment_name,
            save_path=session.deps.storage.save_path,
            is_complete=session.is_complete,
        )

    def _require_session(self, session_id: UUID) -> _SessionEntry:
        """Return the requested session or raise a lookup error."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Unknown session: {session_id}")
        return session

    async def _build_policy(
        self,
        policy_name: str,
        deps: Settings,
    ) -> ConversationPolicy:
        """Build one policy runtime from the configured factories."""
        factory = self._policy_factories.get(policy_name)
        if factory is None:
            raise UnsupportedPolicyError(f"Unsupported policy: {policy_name}")
        return await factory(deps)

    @staticmethod
    def _has_completion(events: Sequence[BackendEvent]) -> bool:
        """Check whether the event list marks a session complete."""
        return any(event.kind == "completed" for event in events)
