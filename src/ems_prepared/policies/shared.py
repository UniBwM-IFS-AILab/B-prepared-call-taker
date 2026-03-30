"""Shared helpers used by policy implementations and runtime adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.model.context import Settings
from ems_prepared.util.custom_deepmerge import ignore_empty_merger


def merge_call_state(
    current_state: EmergencyCall,
    new_state: EmergencyCall,
    deps: Settings,
    *,
    state_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge `new_state` into `current_state` and log the merge event."""
    deps.telemetry.logger.debug(
        f"Old State:\t{current_state.model_dump(exclude_none=True)}"
    )
    deps.telemetry.logger.debug(
        f"New State:\t{new_state.model_dump(exclude_none=True)}"
    )

    _ = ignore_empty_merger.merge(current_state.__dict__, new_state.__dict__)

    merged_state_data = current_state.model_dump(exclude_none=True)
    deps.telemetry.logger.debug(f"Merged State:\t{merged_state_data}")
    deps.telemetry.state_logger.info(
        {"state": state_payload if state_payload is not None else merged_state_data},
        extra={"event": "state_merged"},
    )
    return merged_state_data


def record_completion_artifacts(
    deps: Settings,
    state: Any,
    message_history: list[Any],
) -> None:
    """Persist completion artifacts through the session-scoped recorder callback."""
    if deps.record_completion_artifacts is None:
        deps.telemetry.logger.warning(
            "Completion artifact recorder is not configured; skipping final artifacts."
        )
        return
    deps.record_completion_artifacts(state, message_history)


def to_event_payload(value: Any) -> dict[str, Any]:
    """Convert runtime values to JSON-friendly payload data."""
    if isinstance(value, BaseModel):
        return {"state": value.model_dump(exclude_none=True)}
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}
