"""File-backed session output recording implementation.

Recorder ownership:
- Canonical cross-policy session outputs are defined here and written here.
- Policy-specific artifacts (for example graph debug exports) stay in policy code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ems_prepared.model.contracts import BackendEvent, SessionHandle, SessionRecorder

CANONICAL_SESSION_OUTPUT_FILES: Final[tuple[str, ...]] = (
    "manifest.json",
    "events.jsonl",
    "survey.json",
    "deps.json",
    "final_state.json",
    "state_schema.json",
    "message_history.json",
)
(
    MANIFEST_FILE,
    EVENTS_FILE,
    SURVEY_FILE,
    DEPS_FILE,
    FINAL_STATE_FILE,
    STATE_SCHEMA_FILE,
    MESSAGE_HISTORY_FILE,
) = CANONICAL_SESSION_OUTPUT_FILES


class FileSessionRecorder(SessionRecorder):
    """Default filesystem-backed implementation of `SessionRecorder`."""

    def save_manifest(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        """Write one `manifest.json` file for the session."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session": {
                "user_id": str(session.user_id),
                "session_id": str(session.session_id),
                "locale": session.locale.value,
                "frontend_name": session.frontend_name,
                "policy_name": session.policy_name,
                "scenario_name": session.scenario_name,
            },
        }
        payload.update(dict(metadata))
        (save_path / MANIFEST_FILE).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def save_events(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        events: Sequence[BackendEvent],
    ) -> None:
        """Append backend events to `events.jsonl` for one session."""
        if not events:
            return

        events_file = save_path / EVENTS_FILE
        with events_file.open(mode="a", encoding="utf-8") as f:
            for event in events:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "session_id": str(session.session_id),
                            "kind": event.kind,
                            "text": event.text,
                            "payload": event.payload,
                        },
                        default=str,
                    )
                    + "\n"
                )

    def save_survey(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        responses: list[dict[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write one survey payload to `survey.json`."""
        survey_payload = {
            "timestamp": datetime.now().isoformat(),
            "responses": responses,
            "metadata": {
                "user_id": str(session.user_id),
                "session_id": str(session.session_id),
                "policy_name": session.policy_name,
                "scenario_name": session.scenario_name,
                **(dict(metadata) if metadata is not None else {}),
            },
        }
        output_path = save_path / SURVEY_FILE
        output_path.write_text(
            json.dumps(survey_payload, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path

    def save_completion_artifacts(
        self,
        *,
        session: SessionHandle,
        save_path: Path,
        state: Any,
        message_history: Sequence[Any],
        deps_payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Write final state/deps/message-history artifacts for one completion."""
        if deps_payload is not None:
            (save_path / DEPS_FILE).write_text(
                json.dumps(dict(deps_payload), indent=2, default=str),
                encoding="utf-8",
            )

        self._save_state_json(state=state, save_path=save_path)
        self._save_message_history_json(
            message_history=message_history,
            save_path=save_path,
        )

    @staticmethod
    def _save_state_json(*, state: Any, save_path: Path) -> None:
        """Write `final_state.json` and optional `state_schema.json`."""
        if isinstance(state, BaseModel):
            (save_path / FINAL_STATE_FILE).write_text(
                state.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (save_path / STATE_SCHEMA_FILE).write_text(
                json.dumps(state.model_json_schema(), indent=2),
                encoding="utf-8",
            )
            return

        (save_path / FINAL_STATE_FILE).write_text(
            json.dumps(to_jsonable_python(state), indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _save_message_history_json(
        *,
        message_history: Sequence[Any],
        save_path: Path,
    ) -> None:
        """Write `message_history.json`."""
        (save_path / MESSAGE_HISTORY_FILE).write_text(
            json.dumps(to_jsonable_python(message_history), default=str),
            encoding="utf-8",
        )
