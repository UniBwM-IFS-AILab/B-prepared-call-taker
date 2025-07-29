from pydantic import Field

from ems_prepared.state_model.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.state_model.type_defs import Unknown


class Neurological(KeyQuestionSymptom):
    """Model representing a patient's neurological status with various key questions."""

    acute_motor_deficit: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Motor Deficit",
        description="Indicates if the patient is currently experiencing an acute motor deficit (loss of movement or weakness in a limb).",
    )
    acute_speech_disorder: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Speech Disorder",
        description="Indicates if the patient is currently experiencing an acute speech disorder (difficulty speaking or understanding speech).",
    )
    acute_speech_comprehension_disorder: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Speech Comprehension Disorder",
        description="Indicates if the patient is currently experiencing an acute speech comprehension disorder (difficulty understanding speech).",
    )
    acute_visual_disturbance: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Visual Disturbance",
        description="Indicates if the patient is currently experiencing an acute visual disturbance (sudden changes in vision).",
    )
    acute_headache: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Headache",
        description="Indicates if the patient is currently experiencing an acute headache (sudden and severe headache).",
    )
    acute_hemisensory_loss: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Hemisensory Loss",
        description="Indicates if the patient is currently experiencing acute hemisensory loss (loss of sensation on one side of the body).",
    )
    acute_vertigo_with_falling_risk: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Vertigo with Fall",
        description="Indicates if the patient is currently experiencing acute vertigo with a fall (sudden dizziness leading to a fall).",
    )

    # TODO: encode requiement that consciousness is not normal
    neurological_deficits: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Neurological Deficits",
        description="Indicates if the patient is currently experiencing neurological deficits (loss of function in the nervous system).",
    )
    ongoing_seizure: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Ongoing Seizures",
        description="Indicates if the patient is currently experiencing ongoing seizures (recurrent episodes of abnormal electrical activity in the brain).",
    )
    status_epilepticus: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Status Epilepticus",
        description="Indicates if the patient is currently experiencing status epilepticus (a prolonged seizure lasting more than 5 minutes or multiple seizures without recovery in between).",
    )
