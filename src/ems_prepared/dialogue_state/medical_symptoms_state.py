"""Defines the MedicalEmergency model for representing patient state in medical emergencies.

This module provides:
- MedicalEmergency: A model combining symptoms and states from various emergency call protocol components.
"""  # noqa: E501

from enum import StrEnum
from typing import Literal

from numpy import ndarray
from pydantic import Field
from pydantic.fields import computed_field
from pydantic.functional_validators import model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic.types import T
from rapidfuzz import utils
from rapidfuzz.fuzz import token_sort_ratio
from rapidfuzz.process import cdist

from ems_prepared.dialogue_state.medical.additional_state import AdditionalQuestions
from ems_prepared.dialogue_state.medical.base_models import (
    CPR_Boolean,
    KeyQuestionSymptom,
    RD1_Boolean,
    RD2_Boolean,
    Urgency_Boolean,
)
from ems_prepared.dialogue_state.medical.breathing_state import Breathing
from ems_prepared.dialogue_state.medical.circulation_state import Circulation
from ems_prepared.dialogue_state.medical.conscious_state import Consciousness
from ems_prepared.dialogue_state.medical.immediate_disposition_state import (
    ImmediateDisposition,
    TimeCritical_Boolean,
)
from ems_prepared.dialogue_state.medical.neurological_state import Neurological
from ems_prepared.dialogue_state.medical.tcpr_state import TeleCpr
from ems_prepared.dialogue_state.type_defs import (
    KnownBoolean,
    KnownString,
    Unknown,
    tristate,
)


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Extras(KeyQuestionSymptom):
    """Model for any additional fields that might be added to MedicalEmergency in the future."""

    patient_age: int | Unknown = Field(
        default=None,
        title="Patient Age",
        description="The age of the patient.",
    )
    patient_gender: Gender = Field(
        default=Gender.UNKNOWN,
        title="Patient Gender",
        description="The gender of the patient.",
    )
    covid_vaccination_status: KnownString = Field(
        default=None,
        title="COVID-19 Vaccination Status",
        description=(
            "The COVID-19 vaccination status of the patient (e.g., unvaccinated, partially vaccinated, fully vaccinated, boosted)."
        ),
    )


class MedicalEmergency(
    ImmediateDisposition,
    TeleCpr,
    Breathing,
    Consciousness,
    Circulation,
    Neurological,
    AdditionalQuestions,
    Extras,
):
    """Model representing patient state for a medical emergency.

    This model combines the symptoms from the emergency call protocoll to provide a
    comprehensive view of the patient's medical emergency status.
    """

    def get_outcome(self) -> Literal["cpr", "rd1", "rd2"] | None:
        if self.cpr_needed:
            return "cpr"
        elif self.time_critical:
            return "rd2"
        elif self.rd2:
            return "rd2"
        elif self.rd1:
            return "rd1"
        else:
            return None

    # TODO: move RD1 and RD2 detection to KeyQuestionSymptom and calculate RD1/RD2 by looping over members of type KeyQuestionSymptom
    @property
    def rd1_symptoms(self) -> set[KnownBoolean]:
        """Returns the list of symptoms for rd1."""
        return {
            getattr(self, name)
            for name, field_info in type(self).model_fields.items()
            if field_info.annotation is RD1_Boolean
        }

    @property
    def rd2_symptoms(self) -> set[KnownBoolean]:
        """Returns the set of fields of type KnownBoolean for rd2."""
        return {
            getattr(self, name)
            for name, field_info in type(self).model_fields.items()
            if field_info.annotation in [RD2_Boolean, CPR_Boolean, TimeCritical_Boolean]
        }

    # computed fields are not shown to pydantic_ai agents when enforcing strutured output using output_type
    @computed_field
    @property
    def rd1(self) -> SkipJsonSchema[KnownBoolean]:
        """Returns the count of symptoms in rd1_symptoms."""
        return tristate(
            self.rd1_symptoms.union(
                self.rd2_symptoms,
                self.time_critical_symptoms,
                self.cpr_symptoms,
            )
        )

    @computed_field
    @property
    def rd2(self) -> SkipJsonSchema[KnownBoolean]:
        """Returns the count of symptoms in rd2_symptoms."""
        return tristate(self.rd2_symptoms)

    @model_validator(mode="before")
    @classmethod
    def check_similar_field_names(cls, data: T) -> T:
        """Check for very similar field names in the model and perform validation.

        Parameters
        ----------
        data : T
            The input data to validate.

        Returns
        -------
        T
            The validated data, unchanged.

        Notes
        -----
        Uses rapidfuzz to compare field names and identify those that are very similar,
        which may indicate a typo or duplication. Also useful to avoid too similar field
        names which might lead to confusion in the model.

        """
        field_names: list[str] = list(cls.model_fields.keys())
        threshold: float = 90.0  # Adjust as needed for "very similar"

        if len(field_names) < 2:
            return data

        # Compute the similarity matrix
        sim_matrix: ndarray = cdist(
            field_names,
            field_names,
            scorer=token_sort_ratio,
            processor=utils.default_process,  # type: ignore
            score_cutoff=threshold,
        )  # ty: ignore[no-matching-overload]

        # Identify pairs with similarity above the cutoff
        for first_index in range(len(field_names)):
            for second_index in range(first_index + 1, len(field_names)):
                score = sim_matrix[first_index][second_index]
                if score >= threshold:
                    raise ValueError(
                        f'"{field_names[first_index]}" and "{field_names[second_index]}" are too similar (score: {score:.2f})'  # noqa: E501
                    )
        return data
