"""Emergency call state model module.

This module provides the EmergencyCall class for modeling emergency call data
including caller information, location, emergency type, and medical details.
"""

from pydantic import Field

from ems_prepared.state_model.medical_symptoms_state import MedicalEmergency
from ems_prepared.state_model.type_defs import EmergencyType, KnownString, Unknown


class EmergencyCall(MedicalEmergency):
    """Model representing an emergency call with essential details."""

    caller_name: KnownString = Field(
        default=Unknown,
        title="Caller Name",
        description="Name of the person making the emergency call.",
    )  # type: ignore
    caller_phone: KnownString = Field(
        default=Unknown, title="Caller Phone", description="Phone number of the caller."
    )  # type: ignore
    location: KnownString = Field(
        default=Unknown,
        title="Distinctive Location",
        alias="Address",
        description=(
            "A textual location description that allows for exact pinpointing where the emergency is occurring."
            "It must be precise enough so that is distinct within the area of the department that takes the call."
        ),
    )  # type: ignore
    emergency_type: EmergencyType | Unknown = Field(
        default=Unknown,
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
        default=Unknown,
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
