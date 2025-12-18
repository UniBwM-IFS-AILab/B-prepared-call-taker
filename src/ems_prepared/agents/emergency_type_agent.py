from typing import Literal

from ems_prepared.agents.state_fill_agent import state_fill_prompt
from ems_prepared.dialogue_state.type_defs import EmergencyType
from ems_prepared.util.models import build_fallback_agent

emergency_type_agent = build_fallback_agent(
    output_type=[Literal[EmergencyType.MEDICAL, EmergencyType.FIRE]],
    instructions=(state_fill_prompt.full_prompt),
)
