"""Emergency call state model module.

This module provides the EmergencyCall class for modeling emergency call data
including caller information, location, emergency type, and medical details.
"""

from collections import deque
from typing import Iterator, TypeVar

from anthropic._models import BaseModel
from cryptography.utils import cached_property
from pydantic import Field
from pydantic.fields import computed_field

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from ems_prepared.dialogue_state.type_defs import EmergencyType, KnownString, Unknown
from ems_prepared.locale.iterator import load_questions

T = TypeVar("T")
_SENTINEL = object()


class QuestionCatalog(BaseModel):
    locale: str = "en"
    phase: str = "Intro"
    _current_iterator: Iterator = iter([])

    # @computed_field(return_type=deque[str], repr=False)
    @cached_property
    def questions(
        self,
    ) -> dict[str, dict[str, list[str]]]:
        questions: dict = load_questions()
        return questions

    def fifo(
        self,
    ):
        dq = deque(self.questions.get(self.locale, {}).get(self.phase, []))
        return iter(
            lambda: dq.popleft() if dq else _SENTINEL, _SENTINEL
        )  # preserves order

    def lifo(self):
        lq = self.questions.get(self.locale, {}).get(self.phase, [])
        return iter(lambda: lq.pop() if lq else _SENTINEL, _SENTINEL)  # fastest


class EmergencyCall(MedicalEmergency, QuestionCatalog):
    """Model representing an emergency call with essential details."""

    caller_name: KnownString = Field(
        default=None,
        title="Caller Name",
        description="Name of the person making the emergency call.",
    )  # type: ignore
    caller_phone: KnownString = Field(
        default=None, title="Caller Phone", description="Phone number of the caller."
    )  # type: ignore
    location: KnownString = Field(
        default=None,
        title="Distinctive Location",
        alias="Address",
        description=(
            "A textual location description that allows for exact pinpointing where the emergency is occurring."
            "It must be precise enough so that is distinct within the area of the department that takes the call."
        ),
    )  # type: ignore
    emergency_type: EmergencyType | None = Field(
        default=None,
        examples=[
            EmergencyType.FIRE,
            EmergencyType.MEDICAL,
            EmergencyType.FIRE_MEDICAL,
            EmergencyType.NON_EMERGENCY,
            Unknown,
        ],
        title="Emergency Type",
        description="Type of emergency (e.g., medical, fire, non-emergency).",
    )  # type: ignore
    situation_description: KnownString = Field(
        default=None,
        alias="What",
        title="Situation Description",
        description="Description of what just happened. The reason for calling the emergency line.",
    )  # type: ignore
    # patient_symptoms: MedicalEmergency = Field(
    #     # default=MedicalEmergency(),
    #     default_factory=MedicalEmergency,
    #     examples=[],  # TODO: fill with full and partial examples
    #     title="Medical Emergency",
    #     description="Details of the medical emergency if applicable.",
    # )
