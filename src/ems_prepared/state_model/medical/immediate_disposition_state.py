from pydantic import BaseModel, Field

from ems_prepared.state_model.medical.base_models import UrgencySymptom
from ems_prepared.state_model.type_defs import KnownBoolean, Unknown


class ImmediateDisposition(UrgencySymptom):
    """Model representing a patient's immediate disposition with various key questions."""

    cyanosis: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Cyanosis",
        description="Indicates if the patient is currently experiencing cyanosis (bluish discoloration of the skin).",
    )
    suffocation: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Suffocation",
        description="Indicates if the patient is currently experiencing suffocation (inability to breathe).",
    )
    severe_accident: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Severe Accident",
        description="Indicates if the patient is currently involved in a severe accident.",
    )
    severe_injury: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Severe Injury",
        description="Indicates if the patient is currently experiencing a severe injury.",
    )
