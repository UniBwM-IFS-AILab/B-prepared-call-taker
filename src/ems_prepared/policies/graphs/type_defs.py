from typing import TypeAlias

from pydantic_graph.nodes import BaseNode

from ems_prepared.util.settings import Settings
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall

EmergencyNode: TypeAlias = BaseNode[EmergencyCall, Settings, EmergencyCall]
