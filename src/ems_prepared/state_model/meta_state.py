from collections import deque

from pydantic import BaseModel, computed_field

from ems_prepared.state_model.emergency_call_state import EmergencyCall
from ems_prepared.state_model.type_defs import EmergencyType


class MetaState(BaseModel):
    pass

    subgraphs: deque[EmergencyType] = deque[EmergencyType](set[EmergencyType](), 3)

    symptoms = EmergencyCall()

    @computed_field
    @property
    def current_subgraph(self):
        return self.subgraphs[0]


    initialization = ...
    fire = ...
    other = ...

    # optionally include additional instructions
    questions: list[str | tuple[str, str]]
