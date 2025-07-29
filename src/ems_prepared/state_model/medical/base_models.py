"""Defines the base classes for modelling the State necessary for emrgency calls.

The state can be modelled as a tree structure, where each node can represent a set of symptoms.
We can fold over the tree to determine the outcome of the call by the symptoms.
This can be done in a divide and conquer fashion, as each node does not need to know about their parent.

This module provides:
- KeyQuestionSymptom: a Pydantic model for lists of KnownBoolean values and an optional question.
"""
# - KnownBooleanParent: a Pydantic model that contains a list of KnownBooleanLeaf instances and can be chained recursively.

from pydantic import BaseModel

from ems_prepared.state_model.type_defs import KnownBoolean

# todo maybe move these elswhere
type RD1_Boolean = KnownBoolean
type RD2_Boolean = KnownBoolean
type CPR_Boolean = RD2_Boolean
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
