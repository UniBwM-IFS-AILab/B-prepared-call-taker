"""Provide a cached agent for determining emergency types.

This module exposes get_emergency_type_agent which returns a cached fallback
agent configured to determine whether an emergency is MEDICAL or FIRE.
"""

from functools import lru_cache
from typing import Literal

from ems_prepared.agents.state_fill_agent import state_fill_prompt
from ems_prepared.dialogue_state.type_defs import EmergencyType
from ems_prepared.util.models import build_fallback_agent


@lru_cache(maxsize=1)
def get_emergency_type_agent():
    """Return a cached fallback agent configured to determine the emergency type.

    The returned agent is built via build_fallback_agent. We cast the argument
    dicts to the expected type to satisfy the static type checker while keeping
    the runtime values expressive.
    """
    return build_fallback_agent(
        output_type=[Literal[EmergencyType.MEDICAL, EmergencyType.FIRE]],
        instructions=(state_fill_prompt.full_prompt),
    )
