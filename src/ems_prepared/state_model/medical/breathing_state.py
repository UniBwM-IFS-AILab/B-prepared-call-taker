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


class Breathing(KeyQuestionSymptom):
    """Model representing a patient's breathing status with various key questions."""

    now_insufficient_air: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Irregular Breathing",
        description="Indicates if the patient is currently breathing irregularly.",
    )

    rapidly_progressive_respiratory_distress: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Rapidly Progressive Respiratory Distress",
        description="Indicates if the patient is currently experiencing rapidly progressive respiratory distress.",
    )
    severe_dyspnoea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Dyspnoea",
        description="Indicates if the patient is currently experiencing severe dyspnoea (difficulty breathing).",
    )
    cyanosis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cyanosis",
        description="Indicates if the patient is currently experiencing cyanosis (bluish discoloration of the skin).",
    )
    stridor: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Stridor",
        description="Indicates if the patient is currently experiencing stridor (a high-pitched wheezing sound caused by disrupted airflow).",
    )
    abnormal_breathing_sounds: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Abnormal Breath Sounds",
        description="Indicates if the patient is currently experiencing abnormal breath sounds.",
    )
    aspiration: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Aspiration",
        description="Indicates if the patient is currently experiencing aspiration (inhalation of foreign objects into the airway).",
    )
    abnormal_chest_movement: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Abnormal Chest Movement",
        description="Indicates if the patient is currently experiencing abnormal chest movement during breathing.",
    )
    bradypnea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Bradypnea",
        description="Indicates if the patient is currently experiencing bradypnea (abnormally slow breathing).",
    )
    tachypnea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Tachypnea",
        description="Indicates if the patient is currently experiencing tachypnea (abnormally rapid breathing).",
    )
    apnea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Apnea",
        description="Indicates if the patient is currently experiencing apnea (temporary cessation of breathing).",
    )
