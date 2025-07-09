from pydantic import Field

from ems_prepared.state_model.medical.base_models import CPRSymptom
from ems_prepared.state_model.type_defs import KnownBoolean, Unknown


class TeleCpr(CPRSymptom):
    """Model representing a patient's CPR status with various key questions."""

    cardiac_arrest: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Cardiac Arrest",
        description="Indicates if the patient is currently experiencing cardiac arrest.",
    )
    agonal_breathing: KnownBoolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Agonal Breathing",
        description="Indicates if the patient is currently experiencing agonal breathing (gasping or irregular breathing patterns).",
    )
