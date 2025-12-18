from pydantic import Field

from ems_prepared.dialogue_state.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import Unknown


class Neurological(KeyQuestionSymptom):
    """Model representing a patient's neurological status with various key questions."""

    acute_paralysis: RD1_Boolean = Field(  # 4.1
        default=None,
        examples=[True, False, Unknown],
        title="Acute Motor Deficit",
        description="Indicates if the patient is currently experiencing an acute motor deficit (loss of movement or weakness in a limb).",
    )
    acute_speech_disorder: RD1_Boolean = Field(  # 4.2
        default=None,
        examples=[True, False, Unknown],
        title="Acute Speech Disorder",
        description="Indicates if the patient is currently experiencing an acute speech disorder (difficulty speaking or understanding speech).",
    )
    acute_speech_comprehension_disorder: RD1_Boolean = Field(  # 4.2
        default=None,
        examples=[True, False, Unknown],
        title="Acute Speech Comprehension Disorder",
        description="Indicates if the patient is currently experiencing an acute speech comprehension disorder (difficulty understanding speech).",
    )
    acute_visual_disturbance: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Visual Disturbance",
        description="Indicates if the patient is currently experiencing an acute visual disturbance (sudden changes in vision). E.g. Blindness, double vision, visual field defects.",
    )
    acute_severe_headache: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Headache",
        description="Indicates if the patient is currently experiencing an acute strong headache (unprecedented and severe headache).",
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
    seizure: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Seizure",
        description="Indicates if the patient is currently experiencing a seizure (sudden, uncontrolled electrical disturbance in the brain).",
    )

    # TODO: encode requiement that consciousness is not normal
    neurological_deficits: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Neurological Deficits",
        description="Indicates if the patient is currently experiencing neurological deficits (loss of function in the nervous system). Should only be true if consciousness is not normal.",
    )
    prolonged_deadly_seizure: RD2_Boolean = Field(
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
