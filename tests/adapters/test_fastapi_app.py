"""FastAPI adapter tests for request/response boundary behavior."""

from __future__ import annotations

import argparse
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


class FakeSessionManager:
    """Small in-memory service fake for HTTP adapter tests."""

    def __init__(self, tmp_path: Path) -> None:
        self.base_path = tmp_path / "logs" / "exp"
        self.started_requests: list[SessionParameters] = []
        self.handle = SessionHandle(
            user_id=UUID(int=1),
            session_id=UUID(int=2),
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
        self.history_messages = [
            ConversationMessage(
                id="msg_start_1",
                type=MessageType.QUESTION,
                role=MessageRole.ASSISTANT,
                content="Where are you?",
                timestamp="2026-05-13T12:00:00",
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
                timestamp="2026-05-13T12:00:00",
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
        assert session_id == self.handle.session_id
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

    async def end_session(self, session_id: UUID) -> bool:
        if session_id == self.handle.session_id:
            self.history_status = SessionStatus.ENDED
        return session_id == self.handle.session_id

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
            messages=list(self.history_messages),
        )

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        assert session_id == self.handle.session_id
        assert responses
        _ = (feedback, metadata)
        return self.base_path / "survey.json"

    async def submit_feedback(
        self,
        session_id: UUID,
        feedback: MessageFeedback,
    ) -> None:
        self.feedback_calls.append((session_id, feedback))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    return TestClient(create_app(fake_manager))


def test_start_session_response_shape(client: TestClient) -> None:
    response = client.post(
        "/session",
        json={
            "policy_name": "graph",
            "scenario_name": "scenario_01",
            "locale": "english",
            "experiment_name": "exp",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["user_id"] == str(UUID(int=1))
    assert payload["session"]["session_id"] == str(UUID(int=2))
    assert payload["status"] == "active"
    assert payload["messages"] == [
        {
            "id": "msg_start_1",
            "type": "question",
            "role": "assistant",
            "content": "Where are you?",
            "timestamp": "2026-05-13T12:00:00",
        }
    ]


def test_start_session_accepts_omitted_body_and_uses_defaults(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    response = client.post("/session")

    assert response.status_code == 200
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

    response = client.get("/session/defaults")

    assert response.status_code == 200
    assert response.json() == {
        "policy_name": "agent",
        "locale": "german",
        "experiment_name": "exp_default",
    }


def test_send_message_response_shape(client: TestClient) -> None:
    session_id = UUID(int=2)

    message_response = client.post(
        f"/session/{session_id}/messages",
        json={"content": "At Main Street"},
    )
    assert message_response.status_code == 200
    assert message_response.json() == {
        "session_id": str(session_id),
        "status": "active",
        "messages": [
            {
                "id": "msg_reply_1",
                "type": "message",
                "role": "assistant",
                "content": "Acknowledged",
                "timestamp": "2026-05-13T12:00:02",
            }
        ],
    }


def test_send_message_returns_resume_conflict(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    fake_manager.resume_error_on_input = True
    client = TestClient(create_app(fake_manager))

    response = client.post(
        f"/session/{UUID(int=2)}/messages",
        json={"content": "At Main Street"},
    )

    assert response.status_code == 409
    assert "Missing runtime state" in response.json()["detail"]


def test_end_session_response_shape(client: TestClient) -> None:
    session_id = UUID(int=2)
    response = client.delete(f"/session/{session_id}")
    assert response.status_code == 200
    assert response.json() is True


def test_submit_survey_response_shape(client: TestClient) -> None:
    session_id = UUID(int=2)
    response = client.post(
        f"/session/{session_id}/survey",
        json={
            "responses": [
                {"label": "q1", "category": "quality", "question": "Q1", "score": 5}
            ],
            "feedback": "helpful",
            "metadata": {"note": "ok"},
        },
    )
    assert response.status_code == 200
    assert response.json().endswith("survey.json")


def test_get_state_response_shape(client: TestClient) -> None:
    session_id = UUID(int=2)
    response = client.get(f"/session/{session_id}/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["session_id"] == str(session_id)
    assert payload["experiment_name"] == "exp"
    assert payload["save_path"].endswith("logs/exp")
    assert payload["is_complete"] is False


def test_get_history_response_shape(client: TestClient) -> None:
    session_id = UUID(int=2)
    response = client.get(f"/session/{session_id}/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["session_id"] == str(session_id)
    assert payload["status"] == "active"
    assert payload["messages"] == [
        {
            "id": "msg_start_1",
            "type": "question",
            "role": "assistant",
            "content": "Where are you?",
            "timestamp": "2026-05-13T12:00:00",
        }
    ]


def test_submit_feedback_response_shape(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    client = TestClient(create_app(fake_manager))
    session_id = UUID(int=2)

    response = client.post(
        f"/session/{session_id}/feedback",
        json={
            "role": "assistant",
            "content": "Acknowledged",
            "value": "like",
            "message_index": 1,
            "metadata": {"source": "pytest"},
        },
    )

    assert response.status_code == 200
    assert response.json() is True
    submitted_session_id, submitted_feedback = fake_manager.feedback_calls[-1]
    assert submitted_session_id == session_id
    assert submitted_feedback.role is MessageRole.ASSISTANT
    assert submitted_feedback.value == "like"


def test_websocket_open_has_no_side_effects(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    client = TestClient(create_app(fake_manager))

    with client.websocket_connect("/ws"):
        pass

    assert fake_manager.started_requests == []


def test_websocket_start_send_and_get_history(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        first = websocket.receive_json()
        assert first == {
            "type": "session_started",
            "session": {
                "user_id": str(UUID(int=1)),
                "session_id": str(UUID(int=2)),
                "locale": "english",
                "frontend_name": "fastapi_ws",
                "policy_name": "graph",
                "scenario_name": None,
            },
            "status": "active",
            "messages": [
                {
                    "id": "msg_start_1",
                    "type": "question",
                    "role": "assistant",
                    "content": "Where are you?",
                    "timestamp": "2026-05-13T12:00:00",
                }
            ],
        }

        websocket.send_json(
            {
                "type": "send_message",
                "session_id": str(UUID(int=2)),
                "content": "At Main Street",
            }
        )
        second = websocket.receive_json()
        assert second == {
            "type": "turn_result",
            "session_id": str(UUID(int=2)),
            "status": "active",
            "messages": [
                {
                    "id": "msg_reply_1",
                    "type": "message",
                    "role": "assistant",
                    "content": "Acknowledged",
                    "timestamp": "2026-05-13T12:00:02",
                }
            ],
        }

        websocket.send_json(
            {
                "type": "get_history",
                "session_id": str(UUID(int=2)),
            }
        )
        third = websocket.receive_json()
        assert third == {
            "type": "history_result",
            "session": {
                "user_id": str(UUID(int=1)),
                "session_id": str(UUID(int=2)),
                "locale": "english",
                "frontend_name": "fastapi_ws",
                "policy_name": "graph",
                "scenario_name": None,
            },
            "experiment_name": "exp",
            "status": "active",
            "messages": [
                {
                    "id": "msg_start_1",
                    "type": "question",
                    "role": "assistant",
                    "content": "Where are you?",
                    "timestamp": "2026-05-13T12:00:00",
                },
                {
                    "id": "msg_user_1",
                    "type": "message",
                    "role": "user",
                    "content": "At Main Street",
                    "timestamp": "2026-05-13T12:00:01",
                },
                {
                    "id": "msg_reply_1",
                    "type": "message",
                    "role": "assistant",
                    "content": "Acknowledged",
                    "timestamp": "2026-05-13T12:00:02",
                },
            ],
        }


def test_websocket_uses_defaults_on_start_command(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session"})
        response = websocket.receive_json()

    assert response["session"]["locale"] == "german"
    assert response["session"]["policy_name"] == "agent"
    request = fake_manager.started_requests[-1]
    assert request.frontend_name == "fastapi_ws"
    assert request.policy_name == "agent"
    assert request.locale == Locale.DE
    assert request.experiment_name == "exp_default"


def test_websocket_returns_resume_error_on_send(tmp_path: Path) -> None:
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    fake_manager.resume_error_on_input = True
    client = TestClient(create_app(fake_manager))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "send_message",
                "session_id": str(UUID(int=2)),
                "content": "At Main Street",
            }
        )
        error_frame = websocket.receive_json()

    assert error_frame["type"] == "error"
    assert error_frame["code"] == "session_resume_failed"
    assert "request_id" not in error_frame
    assert "details" not in error_frame


def test_websocket_rejects_invalid_payload(client: TestClient) -> None:
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "send_message"})
        error_frame = websocket.receive_json()
        assert error_frame["type"] == "error"
        assert error_frame["code"] == "invalid_message"
        assert "request_id" not in error_frame


def test_get_websocket_schema_response_shape(client: TestClient) -> None:
    response = client.get("/ws/schema")

    assert response.status_code == 200
    payload = response.json()
    assert [route["path"] for route in payload["routes"]] == ["/ws"]
    assert "oneOf" in payload["client_message_schema"]
    assert "anyOf" in payload["server_message_schema"] or "oneOf" in payload[
        "server_message_schema"
    ]
    assert "request_id" not in str(payload["client_message_schema"])
    assert "request_id" not in str(payload["server_message_schema"])
    assert "SessionHistoryResponse" in payload["shared_schemas"]
    assert "SessionHandle" in payload["shared_schemas"]
    assert "SessionHandleResponse" not in payload["shared_schemas"]


def test_openapi_exposes_response_models(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    start_response_schema = schema["paths"]["/session"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    defaults_response_schema = schema["paths"]["/session/defaults"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    history_response_schema = schema["paths"]["/session/{session_id}/history"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    ws_schema_response_schema = schema["paths"]["/ws/schema"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    message_response_schema = schema["paths"]["/session/{session_id}/messages"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    delete_response_schema = schema["paths"]["/session/{session_id}"]["delete"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    survey_response_schema = schema["paths"]["/session/{session_id}/survey"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    feedback_response_schema = schema["paths"]["/session/{session_id}/feedback"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert start_response_schema == {
        "$ref": "#/components/schemas/StartSessionResponse"
    }
    assert defaults_response_schema == {
        "$ref": "#/components/schemas/SessionDefaultsResponse"
    }
    assert history_response_schema == {
        "$ref": "#/components/schemas/SessionHistoryResponse"
    }
    assert ws_schema_response_schema == {
        "$ref": "#/components/schemas/WebSocketSchemaResponse"
    }
    assert message_response_schema == {"$ref": "#/components/schemas/TurnResponse"}
    assert delete_response_schema == {"type": "boolean", "title": "Response End Session Session  Session Id  Delete"}
    assert survey_response_schema == {"type": "string", "title": "Response Submit Survey Session  Session Id  Survey Post"}
    assert feedback_response_schema == {"type": "boolean", "title": "Response Submit Feedback Session  Session Id  Feedback Post"}
    assert "SessionHandle" in components
    assert "SessionHandleResponse" not in components
    assert components["StartSessionResponse"]["properties"]["session"] == {
        "$ref": "#/components/schemas/SessionHandle"
    }
    assert components["SessionHistoryResponse"]["properties"]["session"] == {
        "$ref": "#/components/schemas/SessionHandle"
    }
    assert components["SessionStateResponse"]["properties"]["session"] == {
        "$ref": "#/components/schemas/SessionHandle"
    }


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
