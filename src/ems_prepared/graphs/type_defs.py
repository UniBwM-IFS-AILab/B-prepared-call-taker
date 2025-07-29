from typing import TypeAlias

from pydantic_graph.nodes import BaseNode

from ems_prepared.settings import Settings
from ems_prepared.state_model.emergency_call_state import EmergencyCall

EmergencyNode: TypeAlias = BaseNode[EmergencyCall, Settings, EmergencyCall]
