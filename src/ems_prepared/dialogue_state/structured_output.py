from __future__ import annotations

from typing import Annotated, TypeVar

from pydantic.fields import Field
from pydantic.functional_validators import AfterValidator, model_validator
from pydantic.main import BaseModel
from pydantic.types import StringConstraints

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall

ModelT = TypeVar("ModelT", bound=BaseModel)


def ensure_non_empty_model(value: ModelT) -> ModelT:
    # `value` is already a validated model instance here (AfterValidator)
    if not value.model_dump(exclude_none=True):
        # No non-None fields at all → treat as empty
        raise ValueError("Model must not be empty (all fields None/default).")
    return value


NonEmptyEmergencyCall = Annotated[EmergencyCall, AfterValidator(ensure_non_empty_model)]

# NonEmptyStr: annotated type for non-empty question strings used by agents
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# TODO: consolidate enough_information_gathered somehow, maye merge DialogueOutput and GraphState?
class DialogueOutput(BaseModel):
    # state: EmergencyCall  # This seems easier for the llm to understand
    state: EmergencyCall | None = None  # This seems easier for the llm to understand
    # state: NonEmptyEmergencyCall | None = None
    next_question: NonEmptyStr | None = None
    enough_information_gathered: Annotated[
        bool,
        Field(
            description=(
                "Set this to true only when you believe you have collected enough "
                "information from the user to make a final decision about how to handle the case. "
            ),
        ),
    ] = False

    # @model_validator(mode="after")
    # def require_exactly_one(self) -> DialogueOutput:
    #     # "Non-empty" state = has at least one non-None field
    #     has_state = (
    #         self.state is not None
    #     )  # and bool( self.state.model_dump(exclude_none=True) )
    #     has_question = (
    #         self.next_question is not None
    #     )  # already guaranteed non-empty if not None

    #     # XOR: exactly one of them must be True
    #     if has_state == has_question:
    #         # has_state == has_question means either:
    #         # - both False (no state, no question)
    #         # - both True (both state and question)
    #         from devtools import debug

    #         if self.state:
    #             debug(self.state.model_dump(exclude_none=True))
    #             debug(self.next_question)
    #         raise ValueError(
    #             "Exactly one of `state` (non-empty) or `next_question` must be set."
    #         )

    #     return self

    @model_validator(mode="after")
    def require_question_when_not_enough_info(self) -> DialogueOutput:
        if self.enough_information_gathered is False and not self.next_question:
            raise ValueError(
                "next_question must be provided when enough_information_gathered is False"
            )
        return self

    @model_validator(mode="after")
    def enforce_invariants(self) -> DialogueOutput:
        if (
            (self.enough_information_gathered is True)
            and isinstance(self.state, EmergencyCall)
            and (not self.state.rd1)
        ):
            # Option B: silently normalize (instead of raising)
            self.enough_information_gathered = False

        return self
