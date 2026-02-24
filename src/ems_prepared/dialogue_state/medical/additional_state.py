from pydantic import Field

from ems_prepared.dialogue_state.medical.base_models import (
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
)
from ems_prepared.dialogue_state.type_defs import Unknown

# TODO: There is no concrete ordering for these questions, we should pick appropriate ones based on the current context / dialogue history


class Injury(KeyQuestionSymptom):
    """Model representing a patient's injury status with various key questions."""

    heavy_injury: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Heavy Injury",
        description="Indicates if the patient is currently experiencing a heavy injury (severe physical trauma). [RD1]",
    )

    vital_threat_injury: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Vital Threat Injury",
        description="""Indicates if the injury is life threatening. Examples of vital threats / additions:
        - Impaired consciousness
        - impaired breathing
        - circulatory problems
        - severe bleeding or severe pain for specific causes:
            - fall from a height > 3m
            - stuck or crushed
            - high-speed trauma (e.g., car accident)
            - penetrating injury (head, thorax, abdomen)
        [RD2]""",
    )


class Bleeding(KeyQuestionSymptom):
    """Model representing a patient's bleeding status with various key questions."""

    heavy_non_traumatic_bleeding: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Non-Traumatic Bleeding",
        description="Indicates if the patient is currently experiencing non-traumatic bleeding (bleeding not caused by an injury). [RD1]",
    )

    gastrointestinal_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Gastrointestinal Bleeding",
        description="Indicates if the patient is currently experiencing gastrointestinal bleeding (bleeding from the digestive tract). [RD2]",
    )
    gynecological_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Gynecological Bleeding",
        description="Indicates if the patient is currently experiencing gynecological bleeding (bleeding from the female reproductive tract). [RD2]",
    )
    epistaxis: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Epistaxis",
        description="Indicates if the patient is currently experiencing epistaxis (nosebleed). [RD2]",
    )


class Poisoning(KeyQuestionSymptom):
    """Model representing a patient's poisoning status with various key questions."""

    intoxication: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Poisoning",
        description="Indicates if the patient is currently experiencing acute poisoning (sudden exposure to a toxic substance). [RD1]",
    )

    life_threatening_intoxication: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Life-Threatening Intoxication",
        description="Indicates if the patient is currently experiencing life-threatening intoxication (severe poisoning that poses a risk to life). [RD2]",
    )


class ImminentChildbirth(KeyQuestionSymptom):
    """Model representing a patient's imminent childbirth status with various key questions."""

    imminent_childbirth: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Imminent Childbirth",
        description="Indicates if the patient is currently experiencing imminent childbirth (the baby is about to be born). [RD1]",
    )

    frequent_pregnancy_contraction: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Pregnancy Contraction Frequency < 3 min",
        description="Indicates if the patient is currently experiencing contractions every less than 3 minutes. [RD2]",
    )
    ongoing_delivery: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Ongoing Delivery",
        description="Indicates if the patient is currently in the process of delivering the baby. [RD2]",
    )
    finisehd_delivery: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Finished Delivery",
        description="Indicates if the patient has just finished delivering the baby. [RD2]",
    )
    vaginal_bleeding: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Vaginal Bleeding",
        description="Indicates if the patient is currently experiencing vaginal bleeding during pregnancy or childbirth. [RD2]",
    )


class Pain(KeyQuestionSymptom):  # 5.5
    """Model representing a patient's pain status with various key questions."""

    severe_pain: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Severe Pain",
        description="Indicates if the patient is currently experiencing severe pain. [RD1]",
    )

    unbearable_pain: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Unbearable Pain",
        description="Indicates if the patient is currently experiencing unbearable pain (pain that cannot be tolerated). [RD2]",
    )
    thunderclap_headache: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Thunderclap Headache",
        description="Indicates if the patient is currently experiencing a thunderclap headache (a sudden and severe headache that comes on quickly). [RD2]",
    )
    colicky_pain: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Colicky Pain",
        description="Indicates if the patient is currently experiencing colicky pain (intermittent and severe abdominal pain). [RD2]",
    )


class Psychiatric(KeyQuestionSymptom):
    """Model representing a patient's psychiatric status with various key questions."""

    acute_mental_health_issue: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Issues with Mental Health",
        description="Indicates if the patient is currently experiencing an acute psychiatric disorder (sudden onset of mental health issues). [RD1]",
    )

    acute_self_harm_risk: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Acute Self-Harm Risk",
        description="Indicates if the patient is currently at risk of self-harm (intentionally causing harm to oneself). [RD2]",
    )
    acute_violence_risk: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Violence Risk",
        description="Indicates if the patient is currently at risk of violence (potential to harm others). [RD2]",
    )
    suicidal_tendency: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Suicidal Tendency",
        description="Indicates if the patient is currently exhibiting suicidal tendencies (thoughts or behaviors indicating a desire to harm oneself). [RD2]",
    )


class Metabolic(KeyQuestionSymptom):
    """Model representing a patient's metabolic status with various key questions."""

    known_metabolic_disorder: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Metabolic Disorder",
        description="Indicates if the patient is currently experiencing a metabolic disorder (a condition that affects the body's metabolism). [RD1]",
    )
    diabetic: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Diabetic",
        description="Indicates if the patient is currently experiencing diabetes (a metabolic disorder characterized by high blood sugar levels). [RD1]",
    )
    noticable_coldness: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Noticable Coldness",
        description="Indicates if the patient is currently experiencing noticeable coldness (a sensation of being unusually cold) [RD1]",
    )
    noticable_heat: RD1_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Noticable Heat",
        description="Indicates if the patient is currently experiencing noticeable heat (a sensation of being hot). [RD1]",
    )

    metabolic_decompensation: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Metabolic Decompensation",
        description="Indicates if the patient is currently experiencing metabolic decompensation (a worsening of a metabolic disorder). [RD2]",
    )
    hyperthermia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hyperthermia",
        description="Indicates if the patient is currently experiencing hyperthermia (abnormally high body temperature) with potentially vital threat. [RD2]",
    )
    hypothermia: RD2_Boolean = Field(
        default=None,
        examples=[True, False, Unknown],
        title="Hypothermia",
        description="Indicates if the patient is currently experiencing hypothermia (abnormally low body temperature) with potentially vital threat. [RD2]",
    )


class AdditionalQuestions(
    Metabolic, Psychiatric, Pain, ImminentChildbirth, Poisoning, Bleeding, Injury
):
    """Model representing additional key questions for a patient's condition."""

    # TODO: What to do if all is false?
    # TODO: probably hand over to real human dispatcher in a case such as this
    # Entscheidung nach individueller situationsbedingter Einschätzung des Disponenten: Überprüfung anderer rettungsdienstlicher Indikationen (z.B. RD 2 Sonderlage, RD 1, Krankentransport, RD 3 ff., MANV) oder Überprüfung der Abgabe an andere Vermittlungszentralen (z.B. Ärztlicher ereitschaftsdienst)""
