from pydantic import Field

from ems_prepared.state_model.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.state_model.type_defs import Unknown


class Circulation(KeyQuestionSymptom):
    """Model representing a patient's circulation status with various key questions."""

    chest_discomfort: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Chest Discomfort",
        description="Indicates if the patient is currently experiencing chest discomfort.",
    )
    acute_circulatory_problems: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Circulatory Problems",
        description="Indicates if the patient is currently experiencing acute circulatory problems.",
    )

    acute_chest_pain: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Chest Pain",
        description="Indicates if the patient is currently experiencing acute chest pain.",
    )
    cold_sweat: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Cold Sweat",
        description="Indicates if the patient is currently experiencing cold sweat.",
    )
    pallor: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Pallor",
        description="Indicates if the patient is currently experiencing pallor (pale skin).",
    )
    hypertensive_crisis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hypertensive Crisis",
        description="Indicates if the patient is currently experiencing a hypertensive crisis (severely high blood pressure).",
    )
    hypotensive_collapse: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hypotensive Collapse",
        description="Indicates if the patient is currently experiencing a hypotensive collapse (severely low blood pressure).",
    )

    # TODO: this require one of the sub-symptoms below to be true
    # headache / chest pain / stomach pain / palpations / difficutly breathing
    tachycardia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Tachycardia",
        description="Indicates if the patient is currently experiencing tachycardia (abnormally fast heart rate).",
    )
    bradycardia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Bradycardia",
        description="Indicates if the patient is currently experiencing bradycardia (abnormally slow heart rate).",
    )
    arrhythmia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Arrhythmia",
        description="Indicates if the patient is currently experiencing arrhythmia (irregular heartbeat).",
    )
    pacemaker_malfunction: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Pacemaker Malfunction",
        description="Indicates if the patient is currently experiencing a malfunction of a pacemaker (device that regulates heart rhythm).",
    )

    # TODO: sub-symptoms / indications:  "skin rash", "circulatory issues", "difficulty breathing",
    allergic_reaction: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Allergic Reaction",
        description="Indicates if the patient is currently experiencing an allergic reaction (immune response to a substance).",
    )
    anaphylaxis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Anaphylaxis",
        description="Indicates if the patient is currently experiencing anaphylaxis (severe allergic reaction).",
    )
