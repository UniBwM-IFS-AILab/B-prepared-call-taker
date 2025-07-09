"""Defines the MedicalEmergency model for representing patient state in medical emergencies.

This module provides:
- MedicalEmergency: A model combining symptoms and states from various emergency call protocol components.
"""  # noqa: E501

from deepdiff.helper import UnkownValueCode
from numpy import ndarray
from pydantic import Field
from pydantic.fields import computed_field
from pydantic.functional_validators import model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic.types import T
from rapidfuzz import utils
from rapidfuzz.fuzz import token_sort_ratio
from rapidfuzz.process import cdist

from ems_prepared.state_model.medical.additional_state import AdditionalQuestions
from ems_prepared.state_model.medical.base_models import (
    CPR_Boolean,
    RD1_Boolean,
    RD2_Boolean,
    Urgency_Boolean,
)
from ems_prepared.state_model.medical.breathing_state import Breathing
from ems_prepared.state_model.medical.circulation_state import Circulation
from ems_prepared.state_model.medical.conscious_state import Consciousness
from ems_prepared.state_model.medical.immediate_disposition_state import (
    ImmediateDisposition,
)
from ems_prepared.state_model.medical.neurological_state import Neurological
from ems_prepared.state_model.medical.tcpr_state import TeleCpr
from ems_prepared.state_model.type_defs import KnownBoolean, Unknown


class MedicalEmergency(
    ImmediateDisposition,
    TeleCpr,
    Breathing,
    Consciousness,
    Circulation,
    Neurological,
    AdditionalQuestions,
):
    """Model representing patient state for a medical emergency.

    This model combines the symptoms from the emergency call protocoll to provide a
    comprehensive view of the patient's medical emergency status.
    """

    # @model_validator(mode="before") #TODO: maybe reactivate if needed
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
        for i in range(len(field_names)):
            for j in range(i + 1, len(field_names)):
                score = sim_matrix[i][j]
                if score >= threshold:
                    raise ValueError(
                        f'"{field_names[i]}" and "{field_names[j]}" are too similar (score: {score:.2f})'  # noqa: E501
                    )
        return data

    # TODO: move RD1 and RD2 detection to KeyQuestionSymptom and calculate RD1/RD2 by looping over members of type KeyQuestionSymptom
    @property
    def rd1_symptoms(self) -> set[KnownBoolean]:
        """Returns the list of symptoms for rd1."""
        return {
            getattr(self, name)
            for name, field in type(self).model_fields.items()
            if field.annotation is RD1_Boolean
        }

    @property
    def rd2_symptoms(self) -> set[KnownBoolean]:
        """Returns the set of fields of type KnownBoolean for rd2."""
        return {
            getattr(self, name)
            for name, field in type(self).model_fields.items()
            if field.annotation in [RD2_Boolean, CPR_Boolean, Urgency_Boolean]
        }

    # computed fields are not shown to pydantic_ai agents when enforcing strutured output using output_type
    @computed_field
    @property
    def rd1(self) -> SkipJsonSchema[KnownBoolean]:
        """Returns the count of symptoms in rd1_symptoms."""
        return any(self.rd1_symptoms) or None

    # NOTE: this is a hack, better alternatives would be very welcome
    # hidden_rd2: KnownBoolean = Field(Unknown, exclude=True, repr=False)

    @computed_field(description="")
    @property
    def rd2(self) -> SkipJsonSchema[KnownBoolean]:
        """Returns the count of symptoms in rd2_symptoms."""
        # if self.hidden_rd2 is False:
        #     return False

        return any(self.rd2_symptoms) or None
