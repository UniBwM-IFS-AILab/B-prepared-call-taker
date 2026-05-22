"""FastAPI adapter acting as a thin view over the session manager contract."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Annotated, Any, Protocol, cast, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request, WebSocket
from pydantic import TypeAdapter, ValidationError
from starlette.websockets import WebSocketDisconnect

from ems_prepared.adapters.fastapi.schemas import (
    FeedbackRequest,
    GetHistoryCommand,
    HistoryResultFrame,
    MessageResponse,
    SendMessageCommand,
    SessionDefaultsResponse,
    SessionHistoryResponse,
    SessionStartOptions,
    SessionStartedFrame,
    SessionStateResponse,
    StartSessionCommand,
    StartSessionResponse,
    SubmitSurveyRequest,
    TurnResponse,
    TurnResultFrame,
    WebSocketClientCommand,
    WebSocketErrorFrame,
    WebSocketRouteInfo,
    WebSocketRouteParameter,
    WebSocketSchemaResponse,
)
from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    ConversationMessage,
    FrontendPlugin,
    MessageType,
    MessageFeedback,
    SessionHandle,
    SessionHistory,
    SessionManager,
    SessionParameters,
    SessionResumeError,
    SessionState,
    SessionStatus,
)


@runtime_checkable
class _FastAPIRuntimeState(Protocol):
    """Typed subset of app.state required by this adapter."""

    session_manager: SessionManager
    default_policy: str
    default_locale: Locale
    app_default_experiment_name: str | None


@runtime_checkable
class _SupportsFastAPIState(Protocol):
    """Objects that expose a state object with FastAPI runtime settings."""

    state: _FastAPIRuntimeState


_WS_CLIENT_COMMAND_ADAPTER = TypeAdapter(WebSocketClientCommand)
_WS_SERVER_FRAME_ADAPTER = TypeAdapter(
    SessionStartedFrame | TurnResultFrame | HistoryResultFrame | WebSocketErrorFrame
)
_SESSION_HANDLE_ADAPTER = TypeAdapter(SessionHandle)


def _to_state_response(state: SessionState) -> SessionStateResponse:
    """Map one domain in-memory state snapshot to an API response model."""
    return SessionStateResponse(
        session=state.handle,
        experiment_name=state.experiment_name,
        save_path=str(state.save_path),
        is_complete=state.is_complete,
    )


def _to_message_response(message: ConversationMessage) -> MessageResponse:
    """Map one domain history message to a typed public response model."""
    return MessageResponse(
        id=message.id,
        type=message.type,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp,
        result=cast(dict[str, object] | None, message.result)
        if message.type is MessageType.COMPLETION
        else None,
        code=(
            message.code or "internal_error"
            if message.type is MessageType.ERROR
            else None
        ),
        details=cast(dict[str, object] | None, message.details)
        if message.type is MessageType.ERROR
        else None,
    )


def _to_history_response(history: SessionHistory) -> SessionHistoryResponse:
    """Map one domain history view to an API response model."""
    return SessionHistoryResponse(
        session=history.handle,
        experiment_name=history.experiment_name,
        status=history.status,
        messages=[_to_message_response(message) for message in history.messages],
    )


def _messages_for_visible_events(
    history: SessionHistory | None,
    events: list[BackendEvent],
) -> list[MessageResponse]:
    """Return the persisted visible messages corresponding to one policy turn."""
    if history is None:
        return []
    visible_event_count = sum(1 for event in events if event.text)
    if visible_event_count <= 0:
        return []
    return [
        _to_message_response(message)
        for message in history.messages[-visible_event_count:]
    ]


def _build_ws_schema_response() -> WebSocketSchemaResponse:
    """Build a machine-readable websocket contract helper payload."""
    return WebSocketSchemaResponse(
        routes=[
            WebSocketRouteInfo(
                path="/ws",
                description=(
                    "Open a websocket transport for explicit start_session, "
                    "send_message, and get_history commands. Opening the socket "
                    "alone has no session side effects. Commands are processed "
                    "sequentially: send one command, wait for one response, then "
                    "send the next command."
                ),
                parameters=[],
            )
        ],
        client_message_schema=_WS_CLIENT_COMMAND_ADAPTER.json_schema(),
        server_message_schema=_WS_SERVER_FRAME_ADAPTER.json_schema(),
        shared_schemas={
            "MessageResponse": MessageResponse.model_json_schema(),
            "SessionHandle": _SESSION_HANDLE_ADAPTER.json_schema(),
            "SessionStartOptions": SessionStartOptions.model_json_schema(),
            "StartSessionResponse": StartSessionResponse.model_json_schema(),
            "TurnResponse": TurnResponse.model_json_schema(),
            "SessionHistoryResponse": SessionHistoryResponse.model_json_schema(),
            "SessionStateResponse": SessionStateResponse.model_json_schema(),
        },
    )


def _resolve_runtime(app_like: Any) -> _FastAPIRuntimeState:
    """Resolve typed runtime state from a FastAPI app-like object."""
    if not isinstance(app_like, _SupportsFastAPIState):
        raise RuntimeError("FastAPI app state is missing required runtime settings.")
    runtime = app_like.state
    if not isinstance(runtime.default_policy, str) or not runtime.default_policy:
        raise RuntimeError("FastAPI default_policy must be a non-empty string.")
    if not isinstance(runtime.default_locale, Locale):
        raise RuntimeError("FastAPI default_locale must be a Locale value.")
    if runtime.app_default_experiment_name is not None and not isinstance(
        runtime.app_default_experiment_name, str
    ):
        raise RuntimeError(
            "FastAPI app_default_experiment_name must be a string or None."
        )
    if not isinstance(runtime.session_manager, SessionManager):
        raise RuntimeError(
            "Configured session manager does not satisfy SessionManager protocol."
        )
    return runtime


def get_session_manager(request: Request) -> SessionManager:
    """FastAPI dependency that resolves the configured session manager."""
    return _resolve_runtime(request.app).session_manager


SessionManagerDep = Annotated[SessionManager, Depends(get_session_manager)]


router = APIRouter()


@router.post("/session")
async def start_session(
    request: Request,
    session_manager: SessionManagerDep,
    payload: SessionStartOptions | None = None,
) -> StartSessionResponse:
    """Create a new session and return initial assistant messages."""
    effective_payload = payload or SessionStartOptions()
    runtime = _resolve_runtime(request.app)
    effective_policy = effective_payload.policy_name or runtime.default_policy
    effective_locale = effective_payload.locale or runtime.default_locale
    effective_experiment_name = (
        effective_payload.experiment_name or runtime.app_default_experiment_name
    )
    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="fastapi",
                policy_name=effective_policy,
                scenario_name=effective_payload.scenario_name,
                user_id=effective_payload.user_id,
                session_id=effective_payload.session_id,
                locale=effective_locale,
                call_origin=InputMode.API,
                experiment_name=effective_experiment_name,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history = session_manager.get_history(handle.session_id)
    status = history.status if history is not None else SessionStatus.ACTIVE
    return StartSessionResponse(
        session=handle,
        status=status,
        messages=_messages_for_visible_events(history, events),
    )


@router.get("/session/defaults")
async def get_session_defaults(request: Request) -> SessionDefaultsResponse:
    """Return the effective default values used for new sessions."""
    runtime = _resolve_runtime(request.app)
    return SessionDefaultsResponse(
        policy_name=runtime.default_policy,
        locale=runtime.default_locale,
        experiment_name=runtime.app_default_experiment_name,
    )


@router.post("/session/{session_id}/messages")
async def send_message(
    session_id: UUID,
    content: Annotated[str, Body(embed=True, min_length=1)],
    session_manager: SessionManagerDep,
) -> TurnResponse:
    """Send one message and return the resulting assistant messages."""
    try:
        events = await session_manager.handle_input(session_id, content)
    except SessionResumeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    history = session_manager.get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return TurnResponse(
        session_id=session_id,
        status=history.status,
        messages=_messages_for_visible_events(history, events),
    )


@router.delete("/session/{session_id}")
async def end_session(
    session_id: UUID,
    session_manager: SessionManagerDep,
) -> bool:
    """End one session and release policy resources."""
    return await session_manager.end_session(session_id)


@router.get("/session/{session_id}/history")
async def get_history(
    session_id: UUID,
    session_manager: SessionManagerDep,
) -> SessionHistoryResponse:
    """Return one session's visible chat history."""
    history = session_manager.get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return _to_history_response(history)


@router.post("/session/{session_id}/survey")
async def submit_survey(
    session_id: UUID,
    payload: SubmitSurveyRequest,
    session_manager: SessionManagerDep,
) -> str:
    """Write survey responses for one active session."""
    try:
        file_path = await session_manager.submit_survey(
            session_id,
            responses=payload.responses,
            feedback=payload.feedback,
            metadata=payload.metadata,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return str(file_path)


@router.post("/session/{session_id}/feedback")
async def submit_feedback(
    session_id: UUID,
    payload: FeedbackRequest,
    session_manager: SessionManagerDep,
) -> bool:
    """Persist one like/dislike feedback event."""
    try:
        await session_manager.submit_feedback(
            session_id,
            MessageFeedback(
                role=payload.role,
                content=payload.content,
                value=payload.value,
                message_index=payload.message_index,
                metadata=dict(payload.metadata or {}),
            ),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return True


@router.get("/session/{session_id}/state")
async def get_state(
    session_id: UUID,
    session_manager: SessionManagerDep,
) -> SessionStateResponse:
    """Return one session's current in-memory state snapshot."""
    state = session_manager.get_view_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return _to_state_response(state)


@router.get("/ws/schema")
async def get_websocket_schema() -> WebSocketSchemaResponse:
    """Return a machine-readable description of the websocket contract."""
    return _build_ws_schema_response()


async def _send_ws_frame(
    websocket: WebSocket,
    frame: SessionStartedFrame | TurnResultFrame | HistoryResultFrame | WebSocketErrorFrame,
) -> None:
    """Send one typed websocket frame as JSON."""
    await websocket.send_json(_WS_SERVER_FRAME_ADAPTER.dump_python(frame, mode="json"))


async def _send_ws_error(
    websocket: WebSocket,
    *,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
    close_code: int | None = None,
) -> None:
    """Send one structured websocket error frame and optionally close."""
    await _send_ws_frame(
        websocket,
        WebSocketErrorFrame(
            type="error",
            code=code,
            message=message,
            details=details,
        ),
    )
    if close_code is not None:
        await websocket.close(code=close_code)


async def _receive_ws_client_command(websocket: WebSocket) -> WebSocketClientCommand:
    """Receive and validate one websocket command frame."""
    payload = await websocket.receive_json()
    return _WS_CLIENT_COMMAND_ADAPTER.validate_python(payload)


async def _dispatch_start_session(
    websocket: WebSocket,
    *,
    command: StartSessionCommand,
    runtime: _FastAPIRuntimeState,
) -> None:
    effective_policy = command.policy_name or runtime.default_policy
    effective_locale = command.locale or runtime.default_locale
    effective_experiment_name = (
        command.experiment_name or runtime.app_default_experiment_name
    )

    try:
        handle, events = await runtime.session_manager.start_session(
            SessionParameters(
                frontend_name="fastapi_ws",
                policy_name=effective_policy,
                scenario_name=command.scenario_name,
                user_id=command.user_id,
                session_id=command.session_id,
                locale=effective_locale,
                call_origin=InputMode.API,
                experiment_name=effective_experiment_name,
            )
        )
    except ValueError as exc:
        await _send_ws_error(
            websocket,
            code="invalid_request",
            message=str(exc),
        )
        return

    history = runtime.session_manager.get_history(handle.session_id)
    status = history.status if history is not None else SessionStatus.ACTIVE
    await _send_ws_frame(
        websocket,
        SessionStartedFrame(
            type="session_started",
            session=handle,
            status=status,
            messages=_messages_for_visible_events(history, events),
        ),
    )


async def _dispatch_send_message(
    websocket: WebSocket,
    *,
    command: SendMessageCommand,
    runtime: _FastAPIRuntimeState,
) -> None:
    session_id = command.session_id
    try:
        events = await runtime.session_manager.handle_input(
            session_id, command.content
        )
    except SessionResumeError as exc:
        await _send_ws_error(
            websocket,
            code="session_resume_failed",
            message=str(exc),
        )
        return
    except LookupError as exc:
        await _send_ws_error(
            websocket,
            code="unknown_session",
            message=str(exc),
        )
        return
    except ValueError as exc:
        await _send_ws_error(
            websocket,
            code="invalid_state",
            message=str(exc),
        )
        return

    history = runtime.session_manager.get_history(session_id)
    if history is None:
        await _send_ws_error(
            websocket,
            code="unknown_session",
            message=f"Unknown session: {session_id}",
        )
        return

    await _send_ws_frame(
        websocket,
        TurnResultFrame(
            type="turn_result",
            session_id=session_id,
            status=history.status,
            messages=_messages_for_visible_events(history, events),
        ),
    )


async def _dispatch_get_history(
    websocket: WebSocket,
    *,
    command: GetHistoryCommand,
    runtime: _FastAPIRuntimeState,
) -> None:
    history = runtime.session_manager.get_history(command.session_id)
    if history is None:
        await _send_ws_error(
            websocket,
            code="unknown_session",
            message=f"Unknown session: {command.session_id}",
        )
        return

    await _send_ws_frame(
        websocket,
        HistoryResultFrame(
            type="history_result",
            session=history.handle,
            experiment_name=history.experiment_name,
            status=history.status,
            messages=[_to_message_response(message) for message in history.messages],
        ),
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run one websocket command transport backed by shared session orchestration."""
    await websocket.accept()
    runtime = _resolve_runtime(websocket.app)

    try:
        while True:
            try:
                command = await _receive_ws_client_command(websocket)
            except ValidationError as exc:
                await _send_ws_error(
                    websocket,
                    code="invalid_message",
                    message=str(exc),
                    close_code=1003,
                )
                return
            except ValueError as exc:
                await _send_ws_error(
                    websocket,
                    code="invalid_json",
                    message=str(exc),
                    close_code=1003,
                )
                return

            if isinstance(command, StartSessionCommand):
                await _dispatch_start_session(
                    websocket,
                    command=command,
                    runtime=runtime,
                )
                continue

            if isinstance(command, SendMessageCommand):
                await _dispatch_send_message(
                    websocket,
                    command=command,
                    runtime=runtime,
                )
                continue
            if isinstance(command, GetHistoryCommand):
                await _dispatch_get_history(
                    websocket,
                    command=command,
                    runtime=runtime,
                )
                continue
    except WebSocketDisconnect:
        pass


def create_app(
    session_manager: SessionManager,
    *,
    default_policy: str = "graph",
    default_locale: Locale = Locale.EN,
    app_default_experiment_name: str | None = None,
) -> FastAPI:
    """Create FastAPI app and wire the shared session manager."""
    app = FastAPI()
    app.state.session_manager = session_manager
    app.state.default_policy = default_policy
    app.state.default_locale = default_locale
    app.state.app_default_experiment_name = app_default_experiment_name
    app.include_router(router)
    return app


class FastAPIFrontend(FrontendPlugin):
    """Frontend plugin that serves the FastAPI adapter via uvicorn."""

    def register_arguments(self, subparser: ArgumentParser, /) -> None:
        """Register FastAPI frontend specific arguments."""
        _ = subparser.add_argument("-H", "--host", default="127.0.0.1")
        _ = subparser.add_argument("--port", type=int, default=8000)

    def run(
        self,
        session_manager: SessionManager,
        parsed_args: Namespace,
        /,
    ) -> int | None:
        """Run the FastAPI app using parser-composed launcher arguments."""
        app = create_app(
            session_manager,
            default_policy=parsed_args.policy,
            default_locale=Locale(parsed_args.locale),
            app_default_experiment_name=parsed_args.experiment_name,
        )
        import uvicorn

        uvicorn.run(app, host=parsed_args.host, port=parsed_args.port, reload=False)
        return 0
