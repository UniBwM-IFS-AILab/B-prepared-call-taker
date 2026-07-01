from typing import TypeAlias

from pydantic_graph.nodes import BaseNode

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.meta_state import GraphState
from ems_prepared.util.settings import Settings

EmergencyNode: TypeAlias = BaseNode[GraphState, Settings, EmergencyCall]
