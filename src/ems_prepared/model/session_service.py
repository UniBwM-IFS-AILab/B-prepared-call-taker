"""Shared backend entry point for storage-backed session lifecycle management."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ems_prepared.model.context import InputMode, Settings
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    ConversationPolicy,
    MessageFeedback,
    MessageRole,
    MessageType,
    PolicyFactory,
    SessionBackend,
    SessionHandle,
    SessionHistory,
    SessionManager,
    SessionParameters,
    SessionRecord,
    SessionState,
    SessionStatus,
)


class SessionService(SessionManager):
    """Shared backend API for creating, driving, and ending sessions."""

    def __init__(
        self,
        *,
        policy_factories: Mapping[str, PolicyFactory],
        session_backend: SessionBackend,
    ) -> None:
        """Create service with pluggable policy factories and session backend."""
        self._policy_factories = dict(policy_factories)
        self._session_backend = session_backend

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        """Create one session, run initial policy start, and persist events."""
        resolved_user_id = request.user_id or uuid4()
        resolved_session_id = request.session_id or uuid4()

        existing = self._session_backend.get_session(resolved_session_id)
        if existing is not None:
            raise ValueError(f"Session already exists: {resolved_session_id}")

        deps = self._build_deps(
            frontend_name=request.frontend_name,
            policy_name=request.policy_name,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            experiment_name=request.experiment_name or "",
            scenario_name=request.scenario_name,
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

        self._session_backend.create_session(
            session=handle,
            save_path=deps.storage.save_path,
            experiment_name=deps.experiment_name,
            metadata={"save_path": str(deps.storage.save_path)},
        )

        policy = await self._build_policy(request.policy_name, deps)
        try:
            events = await policy.start()
        finally:
            await policy.close()

        self._session_backend.save_events(
            session=handle,
            save_path=deps.storage.save_path,
            events=events,
        )
        self._session_backend.save_transcript_entries(
            session=handle,
            save_path=deps.storage.save_path,
            messages=self._build_history_messages(events=events),
        )
        if self._has_completion(events):
            _ = self._session_backend.set_status(
                handle.session_id,
                SessionStatus.COMPLETED,
            )
        return handle, events

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        """Handle one user turn for an existing active session."""
        record = self._require_session(session_id)
        if record.status is SessionStatus.COMPLETED:
            return []

        policy_name = record.handle.policy_name
        if policy_name is None:
            raise ValueError(f"Missing policy_name for session: {session_id}")

        deps = self._build_deps_from_record(record)
        policy = await self._build_policy(policy_name, deps)
        try:
            events = await policy.handle_input(text)
        finally:
            await policy.close()

        self._session_backend.save_events(
            session=record.handle,
            save_path=record.save_path,
            events=events,
        )
        self._session_backend.save_transcript_entries(
            session=record.handle,
            save_path=record.save_path,
            messages=self._build_history_messages(events=events, user_text=text),
        )
        if self._has_completion(events):
            _ = self._session_backend.set_status(
                session_id,
                SessionStatus.COMPLETED,
            )
        return events

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        """Compatibility alias that returns lightweight state without probing runtime."""
        return self.get_view_state(session_id)

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, Any]],
        feedback: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist survey payload for one session."""
        record = self._require_session(session_id)
        return self._session_backend.save_survey(
            session=record.handle,
            save_path=record.save_path,
            responses=responses,
            feedback=feedback,
            metadata=metadata,
        )

    async def submit_feedback(
        self,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        """Persist one message feedback event for a session."""
        _ = self._require_session(session_id)
        self._session_backend.save_feedback(session_id=session_id, feedback=feedback)

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        """Return one UI-facing session snapshot from persistent storage."""
        record = self._session_backend.get_session(session_id)
        if record is None:
            return None

        return SessionState(
            handle=record.handle,
            experiment_name=record.experiment_name,
            save_path=record.save_path,
            is_complete=record.status is SessionStatus.COMPLETED,
        )

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        """Return persisted user-visible chat history for one session."""
        record = self._session_backend.get_session(session_id)
        if record is None:
            return None

        return SessionHistory(
            handle=record.handle,
            experiment_name=record.experiment_name,
            status=record.status,
            created_at=record.created_at,
            messages=self._session_backend.load_history(session_id),
        )

    def _require_session(self, session_id: UUID) -> SessionRecord:
        record = self._session_backend.get_session(session_id)
        if record is None:
            raise LookupError(f"Unknown session: {session_id}")
        return record

    async def _build_policy(
        self,
        policy_name: str,
        deps: Settings,
    ) -> ConversationPolicy:
        factory = self._policy_factories.get(policy_name)
        if factory is None:
            raise ValueError(f"Unsupported policy: {policy_name}")
        return await factory(deps)

    def _build_deps_from_record(self, record: SessionRecord) -> Settings:
        return self._build_deps(
            frontend_name=record.handle.frontend_name,
            policy_name=record.handle.policy_name,
            user_id=record.handle.user_id,
            session_id=record.handle.session_id,
            experiment_name=record.experiment_name,
            scenario_name=record.handle.scenario_name,
            locale=record.handle.locale,
            call_origin=None,
            request_input=None,
        )

    def _build_deps(
        self,
        *,
        frontend_name: str,
        policy_name: str | None,
        user_id: UUID,
        session_id: UUID,
        experiment_name: str,
        scenario_name: str | None,
        locale,
        call_origin,
        request_input,
    ) -> Settings:
        resolved_call_origin = call_origin or self._infer_call_origin(frontend_name)
        kwargs: dict[str, Any] = {
            "name": f"{frontend_name}_{policy_name or 'unknown'}",
            "user_id": user_id,
            "session_id": session_id,
            "experiment_name": experiment_name,
            "scenario_name": scenario_name,
            "policy_name": policy_name,
            "locale": locale,
            "call_origin": resolved_call_origin,
        }
        if request_input is not None:
            kwargs["request_input"] = request_input

        deps = Settings(**kwargs)

        def _record_completion_artifacts(
            state: Any,
            message_history: list[Any],
        ) -> None:
            self._session_backend.save_completion_artifacts(
                session=SessionHandle(
                    user_id=deps.user_id,
                    session_id=deps.session_id,
                    locale=deps.locale,
                    frontend_name=frontend_name,
                    policy_name=policy_name,
                    scenario_name=scenario_name,
                ),
                save_path=deps.storage.save_path,
                state=state,
                message_history=message_history,
                deps_payload=deps.model_dump(
                    mode="json",
                    exclude={
                        "emit",
                        "request_input",
                        "record_completion_artifacts",
                        "load_agent_snapshot",
                        "save_agent_snapshot",
                    },
                ),
            )

        deps.record_completion_artifacts = _record_completion_artifacts
        deps.load_agent_snapshot = lambda: self._session_backend.load_agent_snapshot(
            deps.session_id
        )
        deps.save_agent_snapshot = lambda snapshot: self._session_backend.save_agent_snapshot(
            deps.session_id,
            snapshot,
        )
        return deps

    @staticmethod
    def _infer_call_origin(frontend_name: str) -> InputMode:
        if frontend_name == "cli":
            return InputMode.CLI
        if frontend_name == "tests":
            return InputMode.TEST
        return InputMode.API

    @staticmethod
    def _has_completion(events: Sequence[BackendEvent]) -> bool:
        return any(event.kind is BackendEventKind.COMPLETED for event in events)

    @staticmethod
    def _build_history_messages(
        *,
        events: Sequence[BackendEvent],
        user_text: str | None = None,
    ) -> list[ConversationMessage]:
        messages: list[ConversationMessage] = []
        if user_text is not None and user_text.strip():
            messages.append(
                ConversationMessage(
                    id=f"msg_{uuid4().hex}",
                    type=MessageType.MESSAGE,
                    role=MessageRole.USER,
                    content=user_text,
                    timestamp=datetime.now().isoformat(),
                )
            )

        for event in events:
            if not event.text:
                continue
            message_type = (
                MessageType.COMPLETION
                if event.kind is BackendEventKind.COMPLETED
                else MessageType(event.kind)
            )
            role = (
                MessageRole.SYSTEM
                if event.kind is BackendEventKind.ERROR
                else MessageRole.ASSISTANT
            )
            result = (
                dict(event.payload)
                if event.kind is BackendEventKind.COMPLETED and event.payload
                else None
            )
            details = (
                dict(event.payload)
                if event.kind is BackendEventKind.ERROR and event.payload
                else None
            )
            code = None
            if event.kind is BackendEventKind.ERROR:
                code = str(event.payload.get("code", "internal_error"))
            messages.append(
                ConversationMessage(
                    id=f"msg_{uuid4().hex}",
                    type=message_type,
                    role=role,
                    content=event.text,
                    timestamp=datetime.now().isoformat(),
                    result=result,
                    code=code,
                    details=details,
                )
            )
        return messages
