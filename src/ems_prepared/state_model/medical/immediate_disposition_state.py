from pydantic import BaseModel, Field
from pydantic.fields import computed_field

from ems_prepared.state_model.medical.base_models import KeyQuestionSymptom
from ems_prepared.state_model.type_defs import KnownBoolean, Unknown, tristate


class ImmediateDisposition(KeyQuestionSymptom):
    """Model representing a patient's immediate disposition with various key questions."""

    cyanosis: KnownBoolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cyanosis",
        description="Indicates if the patient is currently experiencing cyanosis (bluish discoloration of the skin).",
    )
    suffocation: KnownBoolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Suffocation",
        description="Indicates if the patient is currently experiencing suffocation (inability to breathe).",
    )
    severe_accident: KnownBoolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Accident",
        description="Indicates if the patient is currently involved in a severe accident.",
    )
    severe_injury: KnownBoolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Injury",
        description="Indicates if the patient is currently experiencing a severe injury.",
    )

    @computed_field(
        title="Urgency Needed",
        description="Indicates if urgency is needed based on the symptoms.",
    )
    @property
    def urgency_needed(self) -> KnownBoolean:
        """Returns True if urgency is needed based on the symptoms."""
        symptoms: list = [
            self.cyanosis,
            self.suffocation,
            self.severe_accident,
            self.severe_injury,
        ]
        return tristate(symptoms)
