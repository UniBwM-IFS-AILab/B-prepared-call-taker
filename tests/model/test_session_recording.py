"""Tests for file-backed session recording behavior."""

from __future__ import annotations

import json
from uuid import UUID

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import Locale
from ems_prepared.model.contracts import (
    BackendEvent,
    BackendEventKind,
    ConversationMessage,
    MessageRole,
    MessageType,
    SessionHandle,
)
from ems_prepared.model.session_backend_file import FileSessionBackend


def _build_session_handle() -> SessionHandle:
    return SessionHandle(
        user_id=UUID(int=1),
        session_id=UUID(int=2),
        locale=Locale.EN,
        frontend_name="tests",
        policy_name="graph",
        scenario_name="scenario_01",
    )


def test_save_manifest_writes_expected_file(tmp_path) -> None:
    """Manifest write should create `manifest.json` with session metadata."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)

    recorder.save_manifest(
        session=_build_session_handle(),
        save_path=save_path,
        metadata={"experiment_name": "exp"},
    )

    payload = json.loads((save_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["session"]["session_id"] == str(UUID(int=2))
    assert payload["experiment_name"] == "exp"
    assert "timestamp" in payload


def test_save_events_appends_jsonl_lines(tmp_path) -> None:
    """Event writes should append newline-delimited JSON records."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)
    session = _build_session_handle()

    recorder.save_events(
        session=session,
        save_path=save_path,
        events=[
            BackendEvent(kind=BackendEventKind.MESSAGE, text="hello"),
            BackendEvent(
                kind=BackendEventKind.QUESTION,
                text="where?",
                payload={"step": 1},
            ),
        ],
    )
    recorder.save_events(
        session=session,
        save_path=save_path,
        events=[BackendEvent(kind=BackendEventKind.COMPLETED, text="done")],
    )

    lines = (save_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 3
    assert records[0]["kind"] == "message"
    assert records[1]["payload"] == {"step": 1}
    assert records[2]["kind"] == "completed"
    assert all(record["session_id"] == str(session.session_id) for record in records)


def test_save_survey_writes_expected_payload(tmp_path) -> None:
    """Survey write should return path to created survey file."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)

    output_path = recorder.save_survey(
        session=_build_session_handle(),
        save_path=save_path,
        responses=[{"question": "Q1", "score": 5}],
        feedback="ok",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "survey.json"
    assert payload["responses"] == [{"question": "Q1", "score": 5}]
    assert payload["feedback"] == "ok"
    assert payload["metadata"]["session_id"] == str(UUID(int=2))


def test_save_and_load_history_messages_round_trip(tmp_path) -> None:
    """History writes should be readable while the session is still active."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)
    session = _build_session_handle()
    recorder.save_manifest(
        session=session,
        save_path=save_path,
        metadata={"experiment_name": "exp"},
    )

    recorder.save_transcript_entries(
        session=session,
        save_path=save_path,
        messages=[
            ConversationMessage(
                id="msg_assistant",
                type=MessageType.QUESTION,
                role=MessageRole.ASSISTANT,
                content="Where are you?",
                timestamp="2026-01-01T12:00:00",
            ),
            ConversationMessage(
                id="msg_user",
                type=MessageType.MESSAGE,
                role=MessageRole.USER,
                content="At Main Street",
                timestamp="2026-01-01T12:00:01",
            ),
        ],
    )

    history = recorder.load_history(session.session_id)

    assert [entry.role for entry in history] == ["assistant", "user"]
    assert [entry.content for entry in history] == [
        "Where are you?",
        "At Main Street",
    ]
    assert history[0].type == "question"
    assert history[0].timestamp == "2026-01-01T12:00:00"


def test_save_events_noop_for_empty_list(tmp_path) -> None:
    """Empty event writes should not create `events.jsonl`."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)

    recorder.save_events(
        session=_build_session_handle(),
        save_path=save_path,
        events=[],
    )

    assert not (save_path / "events.jsonl").exists()


def test_save_completion_artifacts_writes_expected_files(tmp_path) -> None:
    """Completion artifact writes should emit state, history, and deps files."""
    recorder = FileSessionBackend()
    save_path = tmp_path / "session"
    save_path.mkdir(parents=True, exist_ok=True)

    recorder.save_completion_artifacts(
        session=_build_session_handle(),
        save_path=save_path,
        state=EmergencyCall(heavy_injury=True),
        message_history=[{"speaker": "operator", "text": "Where are you?"}],
        deps_payload={"policy_name": "graph"},
    )

    final_state = json.loads(
        (save_path / "final_state.json").read_text(encoding="utf-8")
    )
    state_schema = json.loads(
        (save_path / "state_schema.json").read_text(encoding="utf-8")
    )
    message_history = json.loads(
        (save_path / "message_history.json").read_text(encoding="utf-8")
    )
    deps_payload = json.loads((save_path / "deps.json").read_text(encoding="utf-8"))

    assert final_state["rd1"] is True
    assert "properties" in state_schema
    assert message_history == [{"speaker": "operator", "text": "Where are you?"}]
    assert deps_payload["policy_name"] == "graph"
