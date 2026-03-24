"""Shared helpers for policy runtime adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ems_prepared.model.context import Settings
from ems_prepared.model.contracts import ConversationPolicy
from ems_prepared.model.errors import UnsupportedPolicyError


async def build_policy(policy_name: str, deps: Settings) -> ConversationPolicy:
    """Build one concrete conversation-policy runtime by name.

    Imports are intentionally local so selecting one policy does not import
    the full dependency tree of all runtimes.
    """
    if policy_name == "graph":
        from ems_prepared.policies.pydantic_graph.runtime import build_graph_policy

        return await build_graph_policy(deps)
    if policy_name == "agent":
        from ems_prepared.policies.llm_only.runtime import build_agent_policy

        return await build_agent_policy(deps)
    raise UnsupportedPolicyError(f"Unsupported policy: {policy_name}")


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
