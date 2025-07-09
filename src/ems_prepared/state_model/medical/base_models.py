"""Defines the base classes for modelling the State necessary for emrgency calls.

The state can be modelled as a tree structure, where each node can represent a set of symptoms.
We can fold over the tree to determine the outcome of the call by the symptoms.
This can be done in a divide and conquer fashion, as each node does not need to know about their parent.

This module provides:
- KeyQuestionSymptom: a Pydantic model for lists of KnownBoolean values and an optional question.
"""
# - KnownBooleanParent: a Pydantic model that contains a list of KnownBooleanLeaf instances and can be chained recursively.

from functools import reduce

from pydantic import BaseModel
from pydantic.fields import computed_field

from ems_prepared.state_model.type_defs import KnownBoolean, Unknown

# todo maybe move these elswhere
type RD1_Boolean = KnownBoolean
type RD2_Boolean = KnownBoolean
type CPR_Boolean = KnownBoolean
type Urgency_Boolean = KnownBoolean


class KeyQuestionSymptom(BaseModel):
    """A model that extends KnownBoolean to include additional attributes."""

    # @property
    # def rd1_symptoms(self) -> set[KnownBoolean]:
    #     """Returns the list of symptoms for rd2."""
    #     return {
    #         getattr(self, name)
    #         for name, field in type(self).model_fields.items()
    #         if field.annotation is RD1_Boolean
    #     }

    # @property
    # def rd2_symptoms(self) -> set[KnownBoolean]:
    #     """Returns the set of fields of type KnownBoolean for rd2."""
    #     return {
    #         getattr(self, name)
    #         for name, field in type(self).model_fields.items()
    #         if field.annotation is RD2_Boolean
    #     }

    # @computed_field
    # @property
    # def rd1(self) -> KnownBoolean:
    #     """Returns the count of symptoms in rd1_symptoms."""
    #     return any(self.rd1_symptoms) or None

    # @computed_field
    # @property
    # def rd2(self) -> KnownBoolean:
    #     """Returns the count of symptoms in rd2_symptoms."""
    #     return any(self.rd2_symptoms) or None


# class KnownBooleanParent(BaseModel):
#     """A model that contains a list of KnownBooleanModel instances."""

#     symptom_groups: list[KeyQuestionSymptom] = []

#     @computed_field  # will be included in the model's schema
#     @property
#     def rd1(self) -> KnownBoolean:
#         """Returns the count of symptoms in all rd1_symptoms across all models."""
#         return any(model.rd1 for model in self.symptom_groups)

#     @computed_field
#     @property
#     def rd2(self) -> KnownBoolean:
#         """Returns the count of symptoms in all rd2_symptoms across all models."""
#         return any(model.rd2 for model in self.symptom_groups)


class CPRSymptom(KeyQuestionSymptom):
    """A model that extends KnownBoolean to include additional attributes."""

    @computed_field(
        title="CPR Needed",
        description="Indicates if CPR is needed based on the symptoms.",
    )
    @property
    def cpr_needed(self) -> KnownBoolean:
        """Returns True if CPR is needed based on the symptoms."""
        # return any(
        #     getattr(self, name)
        #     for name, field in type(self).model_fields.items()
        #     if field.annotation is KnownBoolean and name != "cpr_needed"
        # )
        return reduce(
            lambda acc, curr_field: acc or curr_field,
            (
                getattr(self, name)
                for name, field in type(self).model_fields.items()
                if field.annotation is CPR_Boolean and name != "urgency_needed"
            ),
            Unknown,
        )


class UrgencySymptom(KeyQuestionSymptom):
    """A model that extends KnownBoolean to include additional attributes for urgency symptoms."""

    @computed_field(
        title="Urgency Needed",
        description="Indicates if urgency is needed based on the symptoms.",
    )
    @property
    def urgency_needed(self) -> KnownBoolean:
        """Returns True if urgency is needed based on the symptoms."""
        # return none_any(
        #     getattr(self, name)
        #     for name, field in type(self).model_fields.items()
        #     if field.annotation is KnownBoolean and name != "urgency_needed"
        # )
        return reduce(
            lambda acc, x: acc or x,
            (
                getattr(self, name)
                for name, field in type(self).model_fields.items()
                if field.annotation is Urgency_Boolean and name != "urgency_needed"
            ),
            Unknown,
        )
