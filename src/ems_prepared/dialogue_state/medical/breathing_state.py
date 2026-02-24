"""Models for representing a patient's breathing status and related clinical indicators.

This module defines:
- KnownBooleanLeaf: A Pydantic model for key breathing-related clinical questions.
"""

from pydantic import Field

from ems_prepared.dialogue_state.medical.base_models import (
    CPR_Boolean,
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import Unknown


class Breathing(KeyQuestionSymptom):
    """Model representing a patient's breathing status with various key questions."""

    now_insufficient_air: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Irregular Breathing",
        description="Indicates if the patient is currently breathing irregularly. [RD1]",
    )

    rapidly_progressive_respiratory_distress: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Rapidly Progressive Respiratory Distress",
        description="Indicates if the patient is currently experiencing rapidly progressive respiratory distress. [RD2]",
    )
    severe_dyspnoea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Dyspnoea",
        description="Indicates if the patient is currently experiencing severe dyspnoea (difficulty breathing). [RD2]",
    )
    cyanosis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cyanosis",
        description="Indicates if the patient is currently experiencing cyanosis (bluish discoloration of the skin). [RD2]",
    )
    stridor: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Stridor",
        description="Indicates if the patient is currently experiencing stridor (a high-pitched wheezing sound caused by disrupted airflow). [RD2]",
    )
    abnormal_breathing_sounds: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Abnormal Breath Sounds",
        description="Indicates if the patient is currently experiencing abnormal breath sounds. [RD2]",
    )
    aspiration: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Aspiration",
        description="Indicates if the patient is currently experiencing aspiration (inhalation of foreign objects into the airway). [RD2]",
    )
    abnormal_chest_movement: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Abnormal Chest Movement",
        description="Indicates if the patient is currently experiencing abnormal chest movement during breathing. [RD2]",
    )
    bradypnea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Bradypnea",
        description="Indicates if the patient is currently experiencing bradypnea (abnormally slow breathing). [RD2]",
    )
    tachypnea: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Tachypnea",
        description="Indicates if the patient is currently experiencing tachypnea (abnormally rapid breathing). [RD2]",
    )

    apnea: CPR_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Apnea",
        description="Indicates if the patient is currently experiencing apnea (temporary cessation of breathing). [RD2][CPR]",
    )
    agonal_respiration: CPR_Boolean = Field(  # German: Schnappatmung
        default=None,
        examples=[True, False, Unknown],
        title="Agonal Breathing",
        description="Indicates if the patient is currently experiencing agonal breathing (gasping or irregular breathing patterns). [RD2][CPR]",
    )
