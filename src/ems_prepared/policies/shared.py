"""Shared helpers used by both policy implementations."""

from typing import Any

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
