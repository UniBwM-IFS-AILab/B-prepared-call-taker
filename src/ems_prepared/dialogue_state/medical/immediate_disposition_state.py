from pydantic import BaseModel, Field
from pydantic.fields import computed_field

from ems_prepared.dialogue_state.medical.base_models import (
    KeyQuestionSymptom,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import KnownBoolean, Unknown, tristate

type TimeCritical_Boolean = KnownBoolean


class ImmediateDisposition(KeyQuestionSymptom):
    """Model representing a patient's immediate disposition with various key questions."""

    cyanosis: TimeCritical_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cyanosis",
        description="Indicates if the patient is currently experiencing cyanosis (bluish discoloration of the skin).",
    )
    suffocation: TimeCritical_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Suffocation",
        description="Indicates if the patient is currently experiencing suffocation (inability to breathe).",
    )
    severe_accident: TimeCritical_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Accident",
        description="Indicates if the patient is currently involved in a severe accident.",
    )
    severe_injury: TimeCritical_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Injury",
        description="Indicates if the patient is currently experiencing a severe injury.",
    )

    @property
    def immediate_disposition_symptoms(self) -> set[TimeCritical_Boolean]:
        """Returns the list of symptoms for immediate disposition."""
        return {
            # this will compute a set of respective values, meaning len will be 3 at most
            getattr(self, name)
            for name, field in type(self).model_fields.items()
            if field.annotation is TimeCritical_Boolean
        }

    @computed_field(
        title="Urgency Needed",
        description="Indicates if urgency is needed based on the symptoms.",
    )
    @property
    def time_critical(self) -> RD2_Boolean:
        """Returns True if urgency is needed based on the symptoms.

        LT-Drs. 17/11351:
        „Um Menschenleben zu retten, ist (…) unverzügliche Erste Hilfe vor allem beim Herz-Kreislaufstillstand, beim Verschlucken von Fremd-körpern, bei Verbrennungen oder bei schweren Blutungen sinnvoll und notwendig.“
        """

        return tristate(self.immediate_disposition_symptoms)
