"""FastAPI adapter tests for the minimal HTTP contract."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from ems_prepared.adapters.fastapi.app import FastAPIFrontend, create_app
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    MessageFeedback,
    MessageRole,
    MessageType,
    SessionHandle,
    SessionHistory,
    SessionParameters,
    SessionResumeError,
    SessionState,
    SessionStatus,
)

_SESSION_UUID = UUID(int=2)
_PUBLIC_SESSION_ID = str(_SESSION_UUID)
_CREATED_AT = "2026-05-13T12:00:00"
_CREATED_AT_TS = int(datetime.fromisoformat(_CREATED_AT).timestamp())


class FakeSessionManager:
    """Small in-memory service fake for HTTP adapter tests."""

    def __init__(self, tmp_path: Path) -> None:
        self.base_path = tmp_path / "logs" / "exp"
        self.started_requests: list[SessionParameters] = []
        self.handle = SessionHandle(
            user_id=UUID(int=1),
            session_id=_SESSION_UUID,
            locale=Locale.EN,
            frontend_name="fastapi",
            policy_name="graph",
            scenario_name="scenario_01",
        )
        self.state = SessionState(
            handle=self.handle,
            experiment_name="exp",
            save_path=self.base_path,
            is_complete=False,
        )
        self.history_status = SessionStatus.ACTIVE
        self.received: list[str] = []
        self.feedback_calls: list[tuple[UUID, MessageFeedback]] = []
        self.resume_error_on_input = False
        self.created_at = _CREATED_AT
        self.history_messages = [
            ConversationMessage(
                id="msg_start_1",
                type=MessageType.QUESTION,
                role=MessageRole.ASSISTANT,
                content="Where are you?",
                timestamp=_CREATED_AT,
            )
        ]

    def _update_handle(
        self,
        *,
        frontend_name: str,
        locale: Locale,
        policy_name: str | None,
        scenario_name: str | None,
    ) -> None:
        self.handle = SessionHandle(
            user_id=self.handle.user_id,
            session_id=self.handle.session_id,
            locale=locale,
            frontend_name=frontend_name,
            policy_name=policy_name,
            scenario_name=scenario_name,
        )
        self.state = SessionState(
            handle=self.handle,
            experiment_name=self.state.experiment_name,
            save_path=self.state.save_path,
            is_complete=self.state.is_complete,
        )

    async def start_session(
        self,
        request: SessionParameters,
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        self._update_handle(
            frontend_name=request.frontend_name,
            locale=request.locale,
            policy_name=request.policy_name,
            scenario_name=request.scenario_name,
        )
        self.started_requests.append(request)
        self.history_status = SessionStatus.ACTIVE
        self.state = SessionState(
            handle=self.handle,
            experiment_name="exp",
            save_path=self.base_path,
            is_complete=False,
        )
        self.history_messages = [
            ConversationMessage(
                id="msg_start_1",
                type=MessageType.QUESTION,
                role=MessageRole.ASSISTANT,
                content="Where are you?",
                timestamp=_CREATED_AT,
            )
        ]
        return self.handle, [
            BackendEvent(
                kind=BackendEventKind.QUESTION,
                text="Where are you?",
                payload={"step": 1},
            )
        ]

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        if self.resume_error_on_input:
            raise SessionResumeError(f"Missing runtime state for session: {session_id}")
        if session_id != self.handle.session_id:
            raise LookupError(f"Unknown session: {session_id}")
        self.received.append(text)
        self.history_messages.extend(
            [
                ConversationMessage(
                    id="msg_user_1",
                    type=MessageType.MESSAGE,
                    role=MessageRole.USER,
                    content=text,
                    timestamp="2026-05-13T12:00:01",
                ),
                ConversationMessage(
                    id="msg_reply_1",
                    type=MessageType.MESSAGE,
                    role=MessageRole.ASSISTANT,
                    content="Acknowledged",
                    timestamp="2026-05-13T12:00:02",
                ),
            ]
        )
        return [BackendEvent(kind=BackendEventKind.MESSAGE, text="Acknowledged")]

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        if session_id == self.handle.session_id:
            return self.state
        return None

    def get_history(self, session_id: UUID) -> SessionHistory | None:
        if session_id != self.handle.session_id:
            return None
        return SessionHistory(
            handle=self.handle,
            experiment_name="exp",
            status=self.history_status,
            created_at=self.created_at,
            messages=list(self.history_messages),
        )

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        if session_id != self.handle.session_id:
            raise LookupError(f"Unknown session: {session_id}")
        assert responses
        _ = (feedback, metadata)
        return self.base_path / "survey.json"

    async def submit_feedback(
        self,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        if session_id != self.handle.session_id:
            raise LookupError(f"Unknown session: {session_id}")
        self.feedback_calls.append((session_id, feedback))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    return TestClient(create_app(fake_manager))


def test_root_discovery_response_shape(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "api"
    assert "/sessions" in payload["paths"]


def test_start_session_response_shape(client: TestClient) -> None:
    response = client.post(
        "/sessions",
        json={
            "policy_name": "graph",
            "scenario_name": "scenario_01",
            "locale": "english",
            "experiment_name": "exp",
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "object": "session",
        "id": _PUBLIC_SESSION_ID,
        "created_at": _CREATED_AT_TS,
        "status": "active",
        "locale": "english",
        "policy_name": "graph",
        "scenario_name": "scenario_01",
        "messages": [
            {
                "role": "assistant",
                "type": "message",
                "content": "Emergency line connected.",
            },
            {
                "role": "assistant",
                "type": "question",
                "content": "Where are you?",
            },
        ],
    }


def test_start_session_uses_defaults_when_fields_are_omitted(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    response = client.post("/sessions", json={})

    assert response.status_code == 201
    request = fake_manager.started_requests[-1]
    assert request.policy_name == "agent"
    assert request.locale == Locale.DE
    assert request.experiment_name == "exp_default"


def test_get_session_defaults_returns_app_defaults(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    response = client.get("/sessions/defaults")

    assert response.status_code == 200
    assert response.json() == {
        "object": "session.defaults",
        "policy_name": "agent",
        "locale": "german",
        "experiment_name": "exp_default",
    }


def test_get_session_response_shape(client: TestClient) -> None:
    response = client.get(f"/sessions/{_PUBLIC_SESSION_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "object": "session",
        "id": _PUBLIC_SESSION_ID,
        "created_at": _CREATED_AT_TS,
        "status": "active",
        "locale": "english",
        "policy_name": "graph",
        "scenario_name": "scenario_01",
    }


def test_send_message_response_shape(client: TestClient) -> None:
    message_response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/messages",
        json={"content": "At Main Street"},
    )

    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["object"] == "session.messages"
    assert isinstance(payload["created_at"], int)
    assert payload["status"] == "active"
    assert payload["messages"] == [
        {
            "role": "assistant",
            "type": "message",
            "content": "Acknowledged",
        }
    ]


def test_send_message_returns_resume_error_shape(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    fake_manager.resume_error_on_input = True
    client = TestClient(create_app(fake_manager))

    response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/messages",
        json={"content": "At Main Street"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "message": f"Missing runtime state for session: {_SESSION_UUID}",
            "type": "invalid_request_error",
            "param": "session_id",
            "code": "session_resume_failed",
        }
    }


def test_send_message_returns_unknown_session_shape(client: TestClient) -> None:
    response = client.post(
        "/sessions/sess_missing/messages",
        json={"content": "hello"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Unknown session: sess_missing",
            "type": "not_found_error",
            "param": "session_id",
            "code": "unknown_session",
        }
    }


def test_send_message_validation_error_shape(client: TestClient) -> None:
    response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/messages",
        json={},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Request validation failed.",
            "type": "invalid_request_error",
            "param": "content",
            "code": "validation_error",
        }
    }


def test_get_history_response_shape(client: TestClient) -> None:
    response = client.get(f"/sessions/{_PUBLIC_SESSION_ID}/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "session.messages"
    assert isinstance(payload["created_at"], int)
    assert payload["status"] == "active"
    assert payload["messages"] == [
        {
            "role": "assistant",
            "type": "message",
            "content": "Emergency line connected.",
        },
        {
            "role": "assistant",
            "type": "question",
            "content": "Where are you?",
        },
    ]


def test_get_history_includes_user_and_assistant_turns(client: TestClient) -> None:
    response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/messages",
        json={"content": "At Main Street"},
    )
    assert response.status_code == 200

    history_response = client.get(f"/sessions/{_PUBLIC_SESSION_ID}/history")

    assert history_response.status_code == 200
    assert history_response.json()["messages"] == [
        {
            "role": "assistant",
            "type": "message",
            "content": "Emergency line connected.",
        },
        {
            "role": "assistant",
            "type": "question",
            "content": "Where are you?",
        },
        {
            "role": "user",
            "type": "message",
            "content": "At Main Street",
        },
        {
            "role": "assistant",
            "type": "message",
            "content": "Acknowledged",
        },
    ]


def test_submit_survey_response_shape(client: TestClient) -> None:
    response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/survey",
        json={
            "responses": [
                {"label": "q1", "category": "quality", "question": "Q1", "score": 5}
            ],
            "feedback": "helpful",
            "metadata": {"note": "ok"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "session.survey"
    assert isinstance(payload["created_at"], int)
    assert payload["status"] == "accepted"


def test_submit_feedback_response_shape(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    client = TestClient(create_app(fake_manager))

    response = client.post(
        f"/sessions/{_PUBLIC_SESSION_ID}/feedback",
        json={
            "role": "assistant",
            "content": "Acknowledged",
            "value": "like",
            "message_index": 1,
            "metadata": {"source": "pytest"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "session.feedback"
    assert isinstance(payload["created_at"], int)
    assert payload["status"] == "accepted"
    submitted_session_id, submitted_feedback = fake_manager.feedback_calls[-1]
    assert submitted_session_id == _SESSION_UUID
    assert submitted_feedback.role is MessageRole.ASSISTANT
    assert submitted_feedback.value == "like"


def test_openapi_exposes_new_response_models_only(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    paths = schema["paths"]

    assert "/" in paths
    assert "/sessions" in paths
    assert "/sessions/defaults" in paths
    assert "/sessions/{session_id}" in paths
    assert "/sessions/{session_id}/messages" in paths
    assert "/sessions/{session_id}/history" in paths
    assert "/sessions/{session_id}/survey" in paths
    assert "/sessions/{session_id}/feedback" in paths
    assert "/session" not in paths
    assert "/ws/schema" not in paths

    start_response_schema = paths["/sessions"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["schema"]
    defaults_response_schema = paths["/sessions/defaults"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    session_response_schema = paths["/sessions/{session_id}"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    history_response_schema = paths["/sessions/{session_id}/history"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    message_response_schema = paths["/sessions/{session_id}/messages"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    survey_response_schema = paths["/sessions/{session_id}/survey"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    feedback_response_schema = paths["/sessions/{session_id}/feedback"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert start_response_schema == {
        "$ref": "#/components/schemas/CreateSessionResponse"
    }
    assert defaults_response_schema == {
        "$ref": "#/components/schemas/SessionDefaultsResponse"
    }
    assert session_response_schema == {
        "$ref": "#/components/schemas/SessionResource"
    }
    assert history_response_schema == {
        "$ref": "#/components/schemas/SessionMessagesResponse"
    }
    assert message_response_schema == {
        "$ref": "#/components/schemas/SessionMessagesResponse"
    }
    assert survey_response_schema == {
        "$ref": "#/components/schemas/SurveyAcceptedResponse"
    }
    assert feedback_response_schema == {
        "$ref": "#/components/schemas/FeedbackAcceptedResponse"
    }
    assert "SessionHandle" not in components
    assert "SessionStateResponse" not in components
    assert "WebSocketSchemaResponse" not in components


def test_frontend_plugin_registers_fastapi_specific_args() -> None:
    plugin = FastAPIFrontend()
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("-p", "--policy", default="graph")
    _ = parser.add_argument(
        "-l",
        "--locale",
        choices=[Locale.EN.value, Locale.DE.value],
        default=Locale.EN.value,
    )
    _ = parser.add_argument(
        "-e",
        "--experiment",
        "--experiment-name",
        dest="experiment_name",
        default=None,
    )
    plugin.register_arguments(parser)
    args = parser.parse_args(
        [
            "-p",
            "agent",
            "-l",
            "german",
            "--experiment",
            "exp_shared",
            "--port",
            "9001",
        ]
    )
    assert args.policy == "agent"
    assert args.locale == "german"
    assert args.experiment_name == "exp_shared"
    assert args.port == 9001
