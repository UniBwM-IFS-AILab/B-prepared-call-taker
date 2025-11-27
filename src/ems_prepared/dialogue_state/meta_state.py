from pydantic import BaseModel

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.question_state import QuestionCatalog


class GraphState(BaseModel):
    """Base class for graph state, combining question catalog and medical emergency details."""

    questions: QuestionCatalog = QuestionCatalog()
    call_state: EmergencyCall = EmergencyCall()


# class MetaState(BaseModel):
#     pass

#     subgraphs: deque[EmergencyType] = deque[EmergencyType](set[EmergencyType](), 3)

#     symptoms: EmergencyCall = EmergencyCall()

#     @computed_field
#     @property
#     def current_subgraph(self):
#         return self.subgraphs[0]

#     initialization = ...
#     fire = ...
#     other = ...

#     # optionally include additional instructions
#     questions: list[str | tuple[str, str]]
