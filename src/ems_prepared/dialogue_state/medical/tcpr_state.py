from pydantic import BaseModel, Field
from pydantic.fields import computed_field
from pydantic.json_schema import SkipJsonSchema

from ems_prepared.dialogue_state.medical.base_models import (
    CPR_Boolean,
    KeyQuestionSymptom,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import KnownBoolean, Unknown, tristate


# TODO: maybe merge this with Breathing
class TeleCpr(BaseModel):
    """Model representing a patient's CPR status with various key questions."""

    cardiac_arrest: CPR_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cardiac Arrest",
        description="Indicates if the patient is currently experiencing cardiac arrest.",
    )

    ems_arrived: KnownBoolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Emergency Medical Services (EMS) Arrived",
        description="Indicates if an someone from EMS, such as an Ambulance or a doctor arrived at the location of the emergency.",
        exclude=True,
        repr=False,
    )

    @property
    def cpr_symptoms(self) -> set[RD2_Boolean]:
        """Returns the list of symptoms for rd1."""
        return {
            # field
            getattr(self, name)
            for name, field in type(self).model_fields.items()
            if field.annotation is CPR_Boolean
        }

    # TODO: need to implement conciousness (patient needs to be unconcious and also have agonal or no breathing)
    @computed_field(
        title="CPR Needed",
        description="Indicates if CPR is needed based on the symptoms.",
    )
    @property
    def cpr_needed(self) -> RD2_Boolean:
        """Returns True if CPR is needed based on the symptoms."""
        return tristate(self.cpr_symptoms)
