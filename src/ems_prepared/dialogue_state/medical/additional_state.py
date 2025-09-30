from pydantic import Field

from ems_prepared.dialogue_state.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import Unknown


class Injury(KeyQuestionSymptom):
    """Model representing a patient's injury status with various key questions."""

    heavy_injury: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Heavy Injury",
        description="Indicates if the patient is currently experiencing a heavy injury (severe physical trauma).",
    )

    vital_threat_injury: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Vital Threat Injury",
        description="Indicates if the injury is life threatening.",
    )


class Bleeding(KeyQuestionSymptom):
    """Model representing a patient's bleeding status with various key questions."""

    heavy_non_traumatic_bleeding: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Non-Traumatic Bleeding",
        description="Indicates if the patient is currently experiencing non-traumatic bleeding (bleeding not caused by an injury).",
    )

    gastrointestinal_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Gastrointestinal Bleeding",
        description="Indicates if the patient is currently experiencing gastrointestinal bleeding (bleeding from the digestive tract).",
    )
    gynecological_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Gynecological Bleeding",
        description="Indicates if the patient is currently experiencing gynecological bleeding (bleeding from the female reproductive tract).",
    )
    # TODO: requirements: "Consciousness", "Breathing", "Circulatory", "severe pain",
    epistaxis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Epistaxis",
        description="Indicates if the patient is currently experiencing epistaxis (nosebleed).",
    )


class Poisoning(KeyQuestionSymptom):
    """Model representing a patient's poisoning status with various key questions."""

    intoxication: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Poisoning",
        description="Indicates if the patient is currently experiencing acute poisoning (sudden exposure to a toxic substance).",
    )

    life_threatening_intoxication: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Life-Threatening Intoxication",
        description="Indicates if the patient is currently experiencing life-threatening intoxication (severe poisoning that poses a risk to life).",
    )


class ImminentChildbirth(KeyQuestionSymptom):
    """Model representing a patient's imminent childbirth status with various key questions."""

    imminent_childbirth: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Imminent Childbirth",
        description="Indicates if the patient is currently experiencing imminent childbirth (the baby is about to be born).",
    )

    pregnancy_contraction_frequency_less_than_3min: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Pregnancy Contraction Frequency < 3 min",
        description="Indicates if the patient is currently experiencing contractions every less than 3 minutes.",
    )
    ongoing_delivery: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Ongoing Delivery",
        description="Indicates if the patient is currently in the process of delivering the baby.",
    )
    vaginal_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Vaginal Bleeding",
        description="Indicates if the patient is currently experiencing vaginal bleeding during pregnancy or childbirth.",
    )


class Pain(KeyQuestionSymptom):
    """Model representing a patient's pain status with various key questions."""

    severe_pain: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Pain",
        description="Indicates if the patient is currently experiencing severe pain.",
    )

    unbearable_pain: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Unbearable Pain",
        description="Indicates if the patient is currently experiencing unbearable pain (pain that cannot be tolerated).",
    )
    thunderclap_headache: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Thunderclap Headache",
        description="Indicates if the patient is currently experiencing a thunderclap headache (a sudden and severe headache that comes on quickly).",
    )
    colicky_pain: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Colicky Pain",
        description="Indicates if the patient is currently experiencing colicky pain (intermittent and severe abdominal pain).",
    )


class Psychiatric(KeyQuestionSymptom):
    """Model representing a patient's psychiatric status with various key questions."""

    psychiatric_behaviour: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Psychiatric Disorder",
        description="Indicates if the patient is currently experiencing an acute psychiatric disorder (sudden onset of mental health issues).",
    )

    acute_self_harm_risk: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Self-Harm Risk",
        description="Indicates if the patient is currently at risk of self-harm (intentionally causing harm to oneself).",
    )
    acute_violence_risk: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Violence Risk",
        description="Indicates if the patient is currently at risk of violence (potential to harm others).",
    )
    suicidal_tendency: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Suicidal Tendency",
        description="Indicates if the patient is currently exhibiting suicidal tendencies (thoughts or behaviors indicating a desire to harm oneself).",
    )


class Metabolic(KeyQuestionSymptom):
    """Model representing a patient's metabolic status with various key questions."""

    known_metabolic_disorder: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Metabolic Disorder",
        description="Indicates if the patient is currently experiencing a metabolic disorder (a condition that affects the body's metabolism).",
    )
    diabetic: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Diabetic",
        description="Indicates if the patient is currently experiencing diabetes (a metabolic disorder characterized by high blood sugar levels).",
    )

    metabolic_decompensation: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Metabolic Decompensation",
        description="Indicates if the patient is currently experiencing metabolic decompensation (a worsening of a metabolic disorder).",
    )
    hyperthermia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hyperthermia",
        description="Indicates if the patient is currently experiencing hyperthermia (abnormally high body temperature).",
    )
    hypothermia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hypothermia",
        description="Indicates if the patient is currently experiencing hypothermia (abnormally low body temperature).",
    )


class AdditionalQuestions(
    Metabolic, Psychiatric, Pain, ImminentChildbirth, Poisoning, Bleeding, Injury
):
    """Model representing additional key questions for a patient's condition."""
