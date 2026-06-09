"""FastAPI adapter exposing the minimal mock-compatible HTTP contract."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
import time
from typing import Annotated, Any, Protocol, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ems_prepared.adapters.fastapi.schemas import (
    ApiMessage,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorDetails,
    ErrorResponse,
    FeedbackAcceptedResponse,
    FeedbackRequest,
    SessionDefaultsResponse,
    SessionMessageRequest,
    SessionMessagesResponse,
    SessionResource,
    SubmitSurveyRequest,
    SurveyAcceptedResponse,
)
from ems_prepared.model.context import InputMode, Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    ConversationMessage,
    FrontendPlugin,
    MessageFeedback,
    MessageRole,
    MessageType,
    SessionHistory,
    SessionManager,
    SessionParameters,
    SessionResumeError,
)

_INITIAL_GREETING = {
    Locale.EN: "Emergency line connected.",
    Locale.DE: "Notruf verbunden.",
}


class APIError(Exception):
    """Structured HTTP error used by the FastAPI adapter."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: str,
        param: str | None,
        code: str,
    ) -> None:
        self.status_code = status_code
        self.error = ErrorResponse(
            error=ErrorDetails(
                message=message,
                type=error_type,
                param=param,
                code=code,
            )
        )
        super().__init__(message)


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


def _now_timestamp() -> int:
    return int(time.time())


def _to_timestamp(value: str | None) -> int:
    if value is None:
        return _now_timestamp()
    return int(datetime.fromisoformat(value).timestamp())


def _public_session_id(session_id: UUID) -> str:
    return str(session_id)


def _decode_public_session_id(public_session_id: str) -> UUID | None:
    try:
        return UUID(public_session_id)
    except ValueError:
        return None


def _unknown_session_error(public_session_id: str) -> APIError:
    return APIError(
        status_code=404,
        message=f"Unknown session: {public_session_id}",
        error_type="not_found_error",
        param="session_id",
        code="unknown_session",
    )


def _to_api_message(message: ConversationMessage) -> ApiMessage:
    return ApiMessage(
        role=message.role,
        type=message.type,
        content=message.content,
        result=message.result if message.type is MessageType.COMPLETION else None,
    )


def _to_api_messages(messages: list[ConversationMessage]) -> list[ApiMessage]:
    return [_to_api_message(message) for message in messages]


def _messages_for_visible_events(
    history: SessionHistory | None,
    events: list[BackendEvent],
) -> list[ApiMessage]:
    if history is None:
        return []
    visible_event_count = sum(1 for event in events if event.text)
    if visible_event_count <= 0:
        return []
    return _to_api_messages(history.messages[-visible_event_count:])


def _ensure_initial_messages(
    messages: list[ApiMessage],
    *,
    locale: Locale,
) -> list[ApiMessage]:
    if not messages:
        return messages
    if (
        len(messages) >= 2
        and messages[0].role is MessageRole.ASSISTANT
        and messages[0].type is MessageType.MESSAGE
        and messages[1].role is MessageRole.ASSISTANT
        and messages[1].type is MessageType.QUESTION
    ):
        return messages
    if messages[0].role is MessageRole.ASSISTANT and messages[0].type is MessageType.QUESTION:
        return [
            ApiMessage(
                role=MessageRole.ASSISTANT,
                type=MessageType.MESSAGE,
                content=_INITIAL_GREETING[locale],
            ),
            *messages,
        ]
    return messages


def _session_resource_from_history(history: SessionHistory) -> SessionResource:
    return SessionResource(
        object="session",
        id=_public_session_id(history.handle.session_id),
        created_at=_to_timestamp(history.created_at),
        status=history.status,
        locale=history.handle.locale,
        policy_name=history.handle.policy_name,
        scenario_name=history.handle.scenario_name,
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


@router.get("/")
async def get_root() -> dict[str, object]:
    """Return a small discovery payload for the mock-compatible API."""
    return {
        "object": "api",
        "paths": [
            "/sessions",
            "/sessions/defaults",
            "/sessions/{session_id}",
            "/sessions/{session_id}/messages",
            "/sessions/{session_id}/history",
            "/sessions/{session_id}/survey",
            "/sessions/{session_id}/feedback",
        ],
    }


@router.get("/sessions/defaults", response_model=SessionDefaultsResponse)
async def get_session_defaults(request: Request) -> SessionDefaultsResponse:
    """Return the effective default values used for new sessions."""
    runtime = _resolve_runtime(request.app)
    return SessionDefaultsResponse(
        object="session.defaults",
        policy_name=runtime.default_policy,
        locale=runtime.default_locale,
        experiment_name=runtime.app_default_experiment_name,
    )


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_session(
    request: Request,
    payload: CreateSessionRequest,
    session_manager: SessionManagerDep,
) -> CreateSessionResponse:
    """Create a new session and return the initial assistant output."""
    runtime = _resolve_runtime(request.app)
    effective_policy = payload.policy_name or runtime.default_policy
    effective_locale = payload.locale or runtime.default_locale
    effective_experiment_name = (
        payload.experiment_name or runtime.app_default_experiment_name
    )
    try:
        handle, events = await session_manager.start_session(
            SessionParameters(
                frontend_name="fastapi",
                policy_name=effective_policy,
                scenario_name=payload.scenario_name,
                locale=effective_locale,
                call_origin=InputMode.API,
                experiment_name=effective_experiment_name,
            )
        )
    except ValueError as exc:
        raise APIError(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            param="policy_name",
            code="validation_error",
        ) from exc

    history = session_manager.get_history(handle.session_id)
    if history is None:
        raise _unknown_session_error(_public_session_id(handle.session_id))

    initial_messages = _ensure_initial_messages(
        _messages_for_visible_events(history, events),
        locale=history.handle.locale,
    )
    resource = _session_resource_from_history(history)
    return CreateSessionResponse(
        **resource.model_dump(mode="python"),
        messages=initial_messages,
    )


def _require_history(
    session_manager: SessionManager,
    public_session_id: str,
) -> SessionHistory:
    session_id = _decode_public_session_id(public_session_id)
    if session_id is None:
        raise _unknown_session_error(public_session_id)
    history = session_manager.get_history(session_id)
    if history is None:
        raise _unknown_session_error(public_session_id)
    return history


@router.get("/sessions/{session_id}", response_model=SessionResource)
async def get_session(
    session_id: str,
    session_manager: SessionManagerDep,
) -> SessionResource:
    """Return one flattened session resource."""
    history = _require_history(session_manager, session_id)
    return _session_resource_from_history(history)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
)
async def send_message(
    session_id: str,
    payload: SessionMessageRequest,
    session_manager: SessionManagerDep,
) -> SessionMessagesResponse:
    """Send one caller message and return assistant output only."""
    decoded_session_id = _decode_public_session_id(session_id)
    if decoded_session_id is None:
        raise _unknown_session_error(session_id)

    try:
        events = await session_manager.handle_input(decoded_session_id, payload.content)
    except SessionResumeError as exc:
        raise APIError(
            status_code=409,
            message=str(exc),
            error_type="invalid_request_error",
            param="session_id",
            code="session_resume_failed",
        ) from exc
    except LookupError as exc:
        raise _unknown_session_error(session_id) from exc
    except ValueError as exc:
        raise APIError(
            status_code=409,
            message=str(exc),
            error_type="invalid_request_error",
            param="session_id",
            code="invalid_state",
        ) from exc

    history = session_manager.get_history(decoded_session_id)
    if history is None:
        raise _unknown_session_error(session_id)
    return SessionMessagesResponse(
        object="session.messages",
        created_at=_now_timestamp(),
        status=history.status,
        messages=_messages_for_visible_events(history, events),
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionMessagesResponse,
)
async def get_history(
    session_id: str,
    session_manager: SessionManagerDep,
) -> SessionMessagesResponse:
    """Return visible session history in the shared messages envelope."""
    history = _require_history(session_manager, session_id)
    return SessionMessagesResponse(
        object="session.messages",
        created_at=_now_timestamp(),
        status=history.status,
        messages=_ensure_initial_messages(
            _to_api_messages(history.messages),
            locale=history.handle.locale,
        ),
    )


@router.post(
    "/sessions/{session_id}/survey",
    response_model=SurveyAcceptedResponse,
)
async def submit_survey(
    session_id: str,
    payload: SubmitSurveyRequest,
    session_manager: SessionManagerDep,
) -> SurveyAcceptedResponse:
    """Persist survey responses for one session."""
    decoded_session_id = _decode_public_session_id(session_id)
    if decoded_session_id is None:
        raise _unknown_session_error(session_id)
    try:
        await session_manager.submit_survey(
            decoded_session_id,
            responses=payload.responses,
            feedback=payload.feedback,
            metadata=payload.metadata,
        )
    except LookupError as exc:
        raise _unknown_session_error(session_id) from exc
    return SurveyAcceptedResponse(
        object="session.survey",
        created_at=_now_timestamp(),
        status="accepted",
    )


@router.post(
    "/sessions/{session_id}/feedback",
    response_model=FeedbackAcceptedResponse,
)
async def submit_feedback(
    session_id: str,
    payload: FeedbackRequest,
    session_manager: SessionManagerDep,
) -> FeedbackAcceptedResponse:
    """Persist one feedback event for a session message."""
    decoded_session_id = _decode_public_session_id(session_id)
    if decoded_session_id is None:
        raise _unknown_session_error(session_id)
    try:
        await session_manager.submit_feedback(
            decoded_session_id,
            MessageFeedback(
                role=payload.role,
                content=payload.content,
                value=payload.value,
                message_index=payload.message_index,
                metadata=dict(payload.metadata or {}),
            ),
        )
    except LookupError as exc:
        raise _unknown_session_error(session_id) from exc
    return FeedbackAcceptedResponse(
        object="session.feedback",
        created_at=_now_timestamp(),
        status="accepted",
    )


def _error_json_response(
    *,
    status_code: int,
    payload: ErrorResponse,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


def _validation_param(exc: RequestValidationError) -> str:
    for error in exc.errors():
        for item in reversed(error.get("loc", ())):
            if item != "body":
                return str(item)
    return "body"


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

    @app.exception_handler(APIError)
    async def handle_api_error(
        request: Request,
        exc: APIError,
    ) -> JSONResponse:
        _ = request
        return _error_json_response(status_code=exc.status_code, payload=exc.error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        _ = request
        return _error_json_response(
            status_code=400,
            payload=ErrorResponse(
                error=ErrorDetails(
                    message="Request validation failed.",
                    type="invalid_request_error",
                    param=_validation_param(exc),
                    code="validation_error",
                )
            ),
        )

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
