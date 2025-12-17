"""Emergency call state model module.

This module provides the EmergencyCall class for modeling emergency call data
including caller information, location, emergency type, and medical details.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import Field
from pydantic.functional_validators import model_validator

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from ems_prepared.dialogue_state.type_defs import EmergencyType, KnownString, Unknown

T = TypeVar("T")
_SENTINEL = object()
logger = logging.getLogger(__name__)


# @dataclass
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
    emergency_type: EmergencyType | None = Field(
        default=None,
        examples=[
            EmergencyType.FIRE,
            EmergencyType.MEDICAL,
            # EmergencyType.FIRE_MEDICAL,
            # EmergencyType.NON_EMERGENCY,
        ],
        title="Emergency Type",
        description="Type of emergency (e.g., medical, fire, non-emergency).",
    )
    situation_description: KnownString = Field(
        default=None,
        title="Situation Description",
        description="Description of what just happened. The reason for calling the emergency line.",
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

    enough_information_gathered: Annotated[
        bool | None,
        Field(
            description=(
                "Set this to true only when you believe you have collected enough "
                "information from the user to make a final decision about how to handle the case. "
            )
        ),
    ] = None

    @model_validator(mode="after")
    def enforce_invariants(self) -> EmergencyCall:
        if self.enough_information_gathered and not self.rd1:
            # Option A: hard fail
            # raise ValueError("ready_to_finalize can only be True when RD1 is True.")

            # Option B: silently normalize (instead of raising)
            self.enough_information_gathered = True
            # object.__setattr__(self, "information_gathering_complete", False)
        return self
