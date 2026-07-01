"""Emergency call state model module.

This module provides the EmergencyCall class for modeling emergency call data
including caller information, location, emergency type, and medical details.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from ems_prepared.dialogue_state.type_defs import EmergencyType, KnownString

T = TypeVar("T")
_SENTINEL = object()
logger = logging.getLogger(__name__)


class EmergencyCall(MedicalEmergency):
    """Model representing an emergency call with essential details."""

    caller_name: KnownString = Field(
        default=None,
        title="Caller Name",
        description="Name of the person making the emergency call.",
    )
    # caller_phone: KnownString = Field(
    #     default=None, title="Caller Phone", description="Phone number of the caller."
    # )
    emergency_location: KnownString = Field(
        default=None,
        title="Distinctive Location",
        description=(
            "A location description that allows for exact pinpointing where the emergency is occurring."
            "It must be precise enough so that is distinct within the area of the department that takes the call."
        ),
    )
    emergency_type: SkipJsonSchema[EmergencyType | None] = Field(
        default=None,
        examples=[
            EmergencyType.FIRE,
            EmergencyType.MEDICAL,
            EmergencyType.INTRO,
            # EmergencyType.FIRE_MEDICAL,
            # EmergencyType.NON_EMERGENCY,
        ],
        title="Emergency Type",
        description="Type of emergency (e.g., medical, fire, non-emergency).",
        # exclude=True,
    )

    situation_description: KnownString = Field(
        default=None,
        title="Situation Description",
        description=(
            "Description of what just happened. The reason for calling the emergency line. \n"
            "This should only be filled once."
        ),
    )

    # function property to check if all rd1 and r2 symptoms are False (not None / Unknown)
    @property
    def no_outcomes(self) -> bool:
        """Check if all RD1 and RD2 symptoms are explicitly False.

        Returns:
            True if all RD1 and RD2 symptoms are False, False otherwise.
        """
        outcome_vars = [
            self.rd1,
            self.rd2,
            self.cpr_needed,
            # self.time_critical,
        ]

        return all(outcome is False for outcome in outcome_vars)


def response_cleanup(input: EmergencyCall | str) -> EmergencyCall | str:
    """Apply various fixes to strings returned by LLMs."""
    if isinstance(input, str):
        # Case: LLM returns markdown codeblock instead of strucured data / code
        if input.startswith("```") and input.endswith("```"):
            print("deteced Markdown codeblock in Agent response")

            input = input.removeprefix("```")
            input = input.removesuffix("```")

            if input.startswith("json"):
                input = input.removeprefix("json")

        # input = input[input.find("\n") + 1 :  input.rfind("\n")]

        # try to produce String at the end of methods
        try:
            return EmergencyCall.model_validate_json(input)
        except Exception as _:
            return input

    return input
