"""Emergency call state model module.

This module provides the EmergencyCall class for modeling emergency call data
including caller information, location, emergency type, and medical details.
"""

from dataclasses import dataclass
from typing import TypeVar

from pydantic import Field

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from ems_prepared.dialogue_state.type_defs import EmergencyType, KnownString, Unknown

T = TypeVar("T")
_SENTINEL = object()


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
