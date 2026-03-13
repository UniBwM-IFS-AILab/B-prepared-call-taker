"""Shared helpers used by both policy implementations."""

import json
from pathlib import Path
from typing import Any

from pydantic.main import BaseModel
from pydantic_core import to_jsonable_python

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.util.custom_deepmerge import ignore_empty_merger
from ems_prepared.util.settings import Settings


def merge_call_state(
    current_state: EmergencyCall,
    new_state: EmergencyCall,
    deps: Settings,
    *,
    state_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge `new_state` into `current_state` and log the merge event."""
    deps.logger.debug(f"Old State:\t{current_state.model_dump(exclude_none=True)}")
    deps.logger.debug(f"New State:\t{new_state.model_dump(exclude_none=True)}")

    _ = ignore_empty_merger.merge(current_state.__dict__, new_state.__dict__)

    merged_state_data = current_state.model_dump(exclude_none=True)
    deps.logger.debug(f"Merged State:\t{merged_state_data}")
    deps.state_logger.info(
        {"state": state_payload if state_payload is not None else merged_state_data},
        extra={"event": "state_merged"},
    )
    return merged_state_data


def save_state_json(
    state: BaseModel,
    save_path: Path,
) -> None:
    """Save final state and schema to JSON files."""
    json_content = state.model_dump_json(indent=2)
    _ = (save_path / "final_state.json").write_text(data=json_content, encoding="utf-8")

    schema_content = state.model_json_schema()
    formatted_schema_content = json.dumps(schema_content, indent=2)
    _ = (save_path / "state_schema.json").write_text(
        formatted_schema_content, encoding="utf-8"
    )


def save_run_artifacts(
    state: BaseModel,
    message_history: list[Any],
    deps: Settings,
) -> None:
    """Persist deps, final state and message history for a completed run."""
    (deps.save_path / "deps.json").write_text(
        data=deps.model_dump_json(indent=2), encoding="utf-8"
    )
    save_state_json(state, deps.save_path)
    save_message_history_json(message_history, deps.save_path)


def save_message_history_json(
    message_history: list[Any],
    save_path: Path,
) -> None:
    """Save message history to JSON file."""
    message_history_file_path = save_path / "message_history.json"
    message_history_json = to_jsonable_python(message_history)
    _ = message_history_file_path.write_text(
        json.dumps(message_history_json), encoding="utf-8"
    )
