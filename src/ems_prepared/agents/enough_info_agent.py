from functools import lru_cache
from typing import Annotated

from pydantic import BaseModel
from pydantic.fields import Field
from pydantic.functional_validators import model_validator

from ems_prepared.agents.history_processors import remove_before_extracion_processor
from ems_prepared.agents.reusable_prompts import (
    BASE_SYSTEM_PROMPT,
    extend_system_prompt,
)
from ems_prepared.dialogue_state.structured_output import NonEmptyStr
from ems_prepared.util.models import build_fallback_agent

enough_info_prompt = extend_system_prompt(
    BASE_SYSTEM_PROMPT,
    task=(
        "Decide if enough information has been gathered based on the current state to make a final decision. "
        "Decide if enough information has been gathered "
        "If not, provide a contextual and concise follow-up question."
        "Try to ensure that you undertand the situation well enough to make a decision."
    ),
    rules=(
        "Do not return a new question when enough information has been gathered and further questions are not needed. "
        "Evaluate completeness of metadata fields like caller name and location. Consult the conversation history to avoid repeating questions. "
        "The contextual question should only ask for fields from state if they are potentially related to already set state. "
    ),
    decisions=(
        "When the current state is sufficient for decision-making, do not provide a next question. "
        "This is a time-critical dialogue. Minimize the number of questions asked while ensuring safety and completeness."
    ),
)


class EnoughInfoGathered(BaseModel):
    """
    Agent output: indicates whether enough information has been gathered
    and provides a `next_question` when more info is needed.
    """

    enough_information_gathered: bool


class EnoughInfoOutput(BaseModel):
    """
    Agent output: indicates whether enough information has been gathered
    and provides a `next_question` when more info is needed.
    """

    enough_information_gathered: Annotated[
        bool | None,
        Field(
            description=(
                "Set this to true only when you believe you have collected enough "
                "information from the user to make a final decision about how to handle the case. "
            ),
        ),
    ] = None
    next_question: NonEmptyStr | None

    @model_validator(mode="after")
    def require_question_when_not_enough(self) -> "EnoughInfoOutput":
        if self.enough_information_gathered is False and not self.next_question:
            raise ValueError(
                "next_question must be provided when enough_information_gathered is False"
            )
        return self


@lru_cache(maxsize=1)
def get_enough_info_agent():
    return build_fallback_agent(
        output_type=EnoughInfoOutput,  # [EnoughInfoGathered, NonEmptyStr],
        instructions=(enough_info_prompt.full_prompt),
        history_processors=[remove_before_extracion_processor],
    )
