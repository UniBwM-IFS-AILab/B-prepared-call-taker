"""Shared constants for the NLU benchmark."""

from __future__ import annotations

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency

MEDICAL_FIELDS: tuple[str, ...] = tuple(MedicalEmergency.model_fields.keys())
NON_MEDICAL_FIELDS: tuple[str, ...] = tuple(
    sorted(set(EmergencyCall.model_fields) - set(MedicalEmergency.model_fields))
)
