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
    SessionHandle,
    SessionParameters,
    SessionState,
)


class FakeSessionManager:
    """Small in-memory service fake for HTTP adapter tests."""

    def __init__(self, tmp_path: Path) -> None:
        """Prebuild deterministic domain objects used by API responses."""
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
        self.received: list[str] = []

    async def start_session(
        self, request: SessionParameters
    ) -> tuple[SessionHandle, list[BackendEvent]]:
        """Return one deterministic session and initial question event."""
        assert request.frontend_name in {"fastapi", "fastapi_ws"}
        self.started_requests.append(request)
        return self.handle, [
            BackendEvent(kind="question", text="Where are you?", payload={"step": 1})
        ]

    async def handle_input(self, session_id: UUID, text: str) -> list[BackendEvent]:
        """Record input and return one deterministic message event."""
        assert session_id == self.handle.session_id
        self.received.append(text)
        return [BackendEvent(kind="message", text="Acknowledged", payload={})]

    async def end_session(self, session_id: UUID) -> bool:
        """Return True only for the known session id."""
        return session_id == self.handle.session_id

    async def resume_session(self, session_id: UUID) -> SessionState | None:
        """Return deterministic in-memory state for the known session id."""
        return self.get_view_state(session_id)

    def get_view_state(self, session_id: UUID) -> SessionState | None:
        """Return deterministic in-memory state for the known session id."""
        if session_id == self.handle.session_id:
            return self.state
        return None

    async def submit_survey(
        self,
        session_id: UUID,
        responses: list[dict[str, object]],
        feedback: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        """Return a deterministic survey output path."""
        assert session_id == self.handle.session_id
        assert responses
        _ = (feedback, metadata)
        return self.base_path / "survey.json"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """FastAPI test client with the backend service replaced by a fake."""
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    return TestClient(create_app(fake_manager))


def test_start_session_response_shape(client: TestClient) -> None:
    """`POST /session` should return the typed response payload shape."""
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
    assert payload["session"]["locale"] == "english"
    assert payload["events"] == [
        {"kind": "question", "text": "Where are you?", "payload": {"step": 1}}
    ]


def test_send_message_response_shape(client: TestClient) -> None:
    """`POST /messages` should return the typed events payload."""
    session_id = UUID(int=2)

    message_response = client.post(
        f"/session/{session_id}/messages",
        json={"text": "At Main Street"},
    )
    assert message_response.status_code == 200
    assert message_response.json() == {
        "events": [{"kind": "message", "text": "Acknowledged", "payload": {}}]
    }


def test_end_session_response_shape(client: TestClient) -> None:
    """`DELETE /session/{id}` should return a typed termination payload."""
    session_id = UUID(int=2)

    response = client.delete(f"/session/{session_id}")
    assert response.status_code == 200
    assert response.json() == {"ended": True}


def test_submit_survey_response_shape(client: TestClient) -> None:
    """`POST /survey` should return a typed output-path payload."""
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
    assert response.json()["path"].endswith("survey.json")


def test_get_state_response_shape(client: TestClient) -> None:
    """`GET /state` should return a typed in-memory state payload."""
    session_id = UUID(int=2)
    response = client.get(f"/session/{session_id}/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["session_id"] == str(session_id)
    assert payload["experiment_name"] == "exp"
    assert payload["save_path"].endswith("logs/exp")
    assert payload["is_complete"] is False


def test_websocket_round_trip(client: TestClient) -> None:
    """`/ws` should stream events from start + one user message."""
    with client.websocket_connect("/ws?policy=graph&locale=english") as websocket:
        first = websocket.receive_json()
        assert first == {
            "kind": "question",
            "text": "Where are you?",
            "payload": {"step": 1},
        }

        websocket.send_text("At Main Street")
        second = websocket.receive_json()
        assert second == {"kind": "message", "text": "Acknowledged", "payload": {}}


def test_start_session_uses_app_defaults_when_policy_and_locale_missing(
    tmp_path: Path,
) -> None:
    """POST /session should apply app-level defaults when payload omits values."""
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    response = client.post("/session", json={"scenario_name": "scenario_01"})

    assert response.status_code == 200
    assert fake_manager.started_requests
    request = fake_manager.started_requests[-1]
    assert request.policy_name == "agent"
    assert request.locale == Locale.DE
    assert request.experiment_name == "exp_default"


def test_websocket_uses_app_defaults_when_query_missing(tmp_path: Path) -> None:
    """WS start should apply app defaults when query params are absent."""
    fake_manager = FakeSessionManager(tmp_path=tmp_path)
    app = create_app(
        fake_manager,
        default_policy="agent",
        default_locale=Locale.DE,
        app_default_experiment_name="exp_default",
    )
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first = websocket.receive_json()
        assert first == {
            "kind": "question",
            "text": "Where are you?",
            "payload": {"step": 1},
        }
        websocket.send_text("At Main Street")
        _ = websocket.receive_json()

    assert fake_manager.started_requests
    request = fake_manager.started_requests[-1]
    assert request.frontend_name == "fastapi_ws"
    assert request.policy_name == "agent"
    assert request.locale == Locale.DE
    assert request.experiment_name == "exp_default"


def test_openapi_exposes_response_models(client: TestClient) -> None:
    """OpenAPI should reference concrete response models for key endpoints."""
    schema = client.get("/openapi.json").json()

    start_response_schema = schema["paths"]["/session"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    message_response_schema = schema["paths"]["/session/{session_id}/messages"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    end_response_schema = schema["paths"]["/session/{session_id}"]["delete"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    survey_response_schema = schema["paths"]["/session/{session_id}/survey"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    state_response_schema = schema["paths"]["/session/{session_id}/state"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert start_response_schema == {
        "$ref": "#/components/schemas/StartSessionResponse"
    }
    assert message_response_schema == {"$ref": "#/components/schemas/EventsResponse"}
    assert end_response_schema == {"$ref": "#/components/schemas/EndSessionResponse"}
    assert survey_response_schema == {
        "$ref": "#/components/schemas/SubmitSurveyResponse"
    }
    assert state_response_schema == {
        "$ref": "#/components/schemas/SessionStateResponse"
    }


def test_frontend_plugin_registers_fastapi_specific_args() -> None:
    """FastAPI plugin parser should include frontend options with shared args."""
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
