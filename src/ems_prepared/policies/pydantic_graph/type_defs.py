from typing import TypeAlias

from pydantic_graph import BaseNode

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.model.context import Settings

EmergencyNode: TypeAlias = BaseNode[GraphState, Settings, EmergencyCall]
