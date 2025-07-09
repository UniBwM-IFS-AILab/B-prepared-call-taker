"""Models for representing a patient's breathing status and related clinical indicators.

This module defines:
- KnownBooleanLeaf: A Pydantic model for key breathing-related clinical questions.
"""

from pydantic import Field

from ems_prepared.state_model.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.state_model.type_defs import Unknown


class Consciousness(KeyQuestionSymptom):
    """Model representing a patient's consciousness status with various key questions."""

    now_unresponsive: RD1_Boolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Normal Responsiveness",
        description="Indicates if the patient is currently unresponsive.",
    )

    unconscious: RD2_Boolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Unconscious",
        description="Indicates if the patient is currently unconscious.",
    )
    rapidly_progressing_unconsciousness: RD2_Boolean = Field(
        default=Unknown,
        examples=[True, False, Unknown],
        title="Rapidly Progressing Unconsciousness",
        description="Indicates if the patient is currently experiencing rapidly progressing unconsciousness.",
    )
