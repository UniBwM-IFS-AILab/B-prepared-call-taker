"""FastAPI adapter acting as a thin view over the session manager contract."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Annotated, Any, Protocol, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    FrontendPlugin,
    SessionHandle,
    SessionManager,
    SessionParameters,
    SessionState,
)
from ems_prepared.model.errors import SessionNotFoundError, UnsupportedPolicyError


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


class StartSessionPayload(BaseModel):
    """Request payload for starting a new session."""

    policy_name: str | None = None
    scenario_name: str | None = None
    user_id: UUID | None = None
    session_id: UUID | None = None
    locale: Locale | None = None
    experiment_name: str | None = None


class SendMessageRequest(BaseModel):
    """Request payload for one user message."""

    text: str = Field(min_length=1)


class SubmitSurveyRequest(BaseModel):
    """Request payload for one survey submission."""

    responses: list[dict[str, object]]
    feedback: str | None = None
    metadata: dict[str, object] | None = None


class BackendEventResponse(BaseModel):
    """JSON-serializable representation of a backend event."""

    kind: str
    text: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class SessionHandleResponse(BaseModel):
    """JSON-serializable session identity payload."""

    user_id: UUID
    session_id: UUID
    locale: Locale
    frontend_name: str
    policy_name: str | None = None
    scenario_name: str | None = None


class StartSessionResponse(BaseModel):
    """Response payload for a session start request."""

    session: SessionHandleResponse
    events: list[BackendEventResponse]


class EventsResponse(BaseModel):
    """Response payload for message handling."""

    events: list[BackendEventResponse]


class EndSessionResponse(BaseModel):
    """Response payload for session termination."""

    ended: bool


class SubmitSurveyResponse(BaseModel):
    """Response payload for survey output location."""

    path: str


class SessionStateResponse(BaseModel):
    """Response payload for one in-memory state snapshot."""

    session: SessionHandleResponse
    experiment_name: str
    save_path: str
    is_complete: bool


def _to_event_response(event: BackendEvent) -> BackendEventResponse:
    """Map one domain backend event to an API response model."""
    return BackendEventResponse(
        kind=event.kind,
        text=event.text,
        payload=event.payload,
    )


def _to_handle_response(handle: SessionHandle) -> SessionHandleResponse:
    """Map one domain session handle to an API response model."""
    return SessionHandleResponse(
        user_id=handle.user_id,
        session_id=handle.session_id,
        locale=handle.locale,
        frontend_name=handle.frontend_name,
        policy_name=handle.policy_name,
        scenario_name=handle.scenario_name,
    )


def _to_state_response(state: SessionState) -> SessionStateResponse:
    """Map one domain in-memory state snapshot to an API response model."""
    return SessionStateResponse(
        session=_to_handle_response(state.handle),
        experiment_name=state.experiment_name,
        save_path=str(state.save_path),
        is_complete=state.is_complete,
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
    payload: StartSessionPayload,
    request: Request,
    session_manager: SessionManagerDep,
) -> StartSessionResponse:
    """Create a new session and return initial backend events."""
    runtime = _resolve_runtime(request.app)
    default_policy = runtime.default_policy
    default_locale = runtime.default_locale
    app_default_experiment_name = runtime.app_default_experiment_name
    effective_policy = payload.policy_name or default_policy
    effective_locale = payload.locale or default_locale
    effective_experiment_name = payload.experiment_name or app_default_experiment_name
    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="fastapi",
                policy_name=effective_policy,
                scenario_name=payload.scenario_name,
                user_id=payload.user_id,
                session_id=payload.session_id,
                locale=effective_locale,
                call_origin=InputMode.API,
                experiment_name=effective_experiment_name,
            )
        )
    except UnsupportedPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StartSessionResponse(
        session=_to_handle_response(handle),
        events=[_to_event_response(event) for event in events],
    )


@router.post("/session/{session_id}/messages")
async def send_message(
    session_id: UUID,
    payload: SendMessageRequest,
    session_manager: SessionManagerDep,
) -> EventsResponse:
    """Send one message and return backend events."""
    try:
        events = await session_manager.handle_input(session_id, payload.text)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return EventsResponse(events=[_to_event_response(event) for event in events])


@router.delete("/session/{session_id}")
async def end_session(
    session_id: UUID,
    session_manager: SessionManagerDep,
) -> EndSessionResponse:
    """End one session and release policy resources."""
    ended = await session_manager.end_session(session_id)
    return EndSessionResponse(ended=ended)


@router.post("/session/{session_id}/survey")
async def submit_survey(
    session_id: UUID,
    payload: SubmitSurveyRequest,
    session_manager: SessionManagerDep,
) -> SubmitSurveyResponse:
    """Write survey responses for one active session."""
    try:
        file_path = await session_manager.submit_survey(
            session_id,
            responses=payload.responses,
            feedback=payload.feedback,
            metadata=payload.metadata,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SubmitSurveyResponse(path=str(file_path))


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


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Run one websocket chat session backed by shared session orchestration."""
    await websocket.accept()
    started_session_id: UUID | None = None
    runtime = _resolve_runtime(websocket.app)
    session_manager = runtime.session_manager
    default_policy = runtime.default_policy
    default_locale = runtime.default_locale
    app_default_experiment_name = runtime.app_default_experiment_name

    policy_name = websocket.query_params.get("policy") or default_policy
    scenario_name = websocket.query_params.get("scenario")
    locale_raw = websocket.query_params.get("locale") or default_locale.value
    experiment_name = (
        websocket.query_params.get("experiment") or app_default_experiment_name
    )
    try:
        locale = Locale(locale_raw)
    except ValueError:
        await websocket.send_json(
            {"kind": "error", "text": f"Unsupported locale: {locale_raw}"}
        )
        await websocket.close(code=1003)
        return

    try:

        async def websocket_request_input(_deps: object, prompt: str) -> str:
            await websocket.send_text(prompt)
            return await websocket.receive_text()

        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="fastapi_ws",
                policy_name=policy_name,
                scenario_name=scenario_name,
                locale=locale,
                call_origin=InputMode.API,
                request_input=websocket_request_input,
                experiment_name=experiment_name,
            )
        )
        started_session_id = handle.session_id
        for event in events:
            await websocket.send_json(_to_event_response(event).model_dump(mode="json"))

        while True:
            user_text = await websocket.receive_text()
            events = await session_manager.handle_input(started_session_id, user_text)
            for event in events:
                await websocket.send_json(
                    _to_event_response(event).model_dump(mode="json")
                )
    except UnsupportedPolicyError:
        await websocket.send_json(
            {"kind": "error", "text": f"Unsupported policy: {policy_name}"}
        )
    except SessionNotFoundError as exc:
        await websocket.send_json({"kind": "error", "text": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        if started_session_id is not None:
            _ = await session_manager.end_session(started_session_id)


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
