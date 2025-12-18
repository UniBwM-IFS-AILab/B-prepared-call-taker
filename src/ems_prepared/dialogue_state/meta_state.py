from __future__ import annotations

import logging

from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.functional_validators import model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai.messages import ModelMessage
from typing_extensions import Annotated

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.question_state import QuestionCatalog

logger = logging.getLogger(__name__)


class GraphState(BaseModel):
    """Base class for graph state, combining question catalog and medical emergency details."""

    questions: QuestionCatalog = QuestionCatalog()
    call_state: EmergencyCall = EmergencyCall()
    message_history: list[ModelMessage] = Field(default_factory=list)

    enough_information_gathered: SkipJsonSchema[
        Annotated[
            bool | None,
            Field(
                description=(
                    "Set this to true only when you believe you have collected enough "
                    "information from the user to make a final decision about how to handle the case. "
                ),
            ),
        ]
    ] = None

    @model_validator(mode="after")
    def enforce_invariants(self) -> GraphState:
        if (self.enough_information_gathered is not None) and (not self.call_state.rd1):
            # Option A: hard fail
            # raise ValueError("ready_to_finalize can only be True when RD1 is True.")

            # Option B: silently normalize (instead of raising)
            logger.debug(
                "Resetting enough_information_gathered to None because rd1 is not True. "
                "rd1=%s, enough_information_gathered=%s",
                self.call_state.rd1,
                self.enough_information_gathered,
            )
            self.enough_information_gathered = None

        return self
