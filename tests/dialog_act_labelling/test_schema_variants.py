from __future__ import annotations

import pytest
from pydantic import ValidationError

from ems_prepared.dialogue_state.schema_variants import (
    EmergencyCallSlotEntries,
    EmergencyCallSlotNames,
    inline_json_schema_refs,
)


def _entry_schema_for(slot_name: str) -> dict:
    schema = inline_json_schema_refs(EmergencyCallSlotEntries.model_json_schema())
    return next(
        entry
        for entry in schema["items"]["anyOf"]
        if entry["properties"]["name"]["const"] == slot_name
    )


def test_slot_name_type_from_model_validates_known_names() -> None:
    parsed = EmergencyCallSlotNames.model_validate(["caller_name", "emergency_location"])
    assert parsed.root == ["caller_name", "emergency_location"]


def test_slot_name_type_from_model_rejects_unknown_names() -> None:
    with pytest.raises(ValidationError):
        EmergencyCallSlotNames.model_validate(["not_a_real_slot"])


def test_slot_entry_type_from_model_requires_value_and_rejects_null() -> None:
    with pytest.raises(ValidationError):
        EmergencyCallSlotEntries.model_validate([{"name": "caller_name"}])

    with pytest.raises(ValidationError):
        EmergencyCallSlotEntries.model_validate(
            [{"name": "caller_name", "value": None}]
        )


def test_slot_entry_type_from_model_rejects_empty_string_values() -> None:
    with pytest.raises(ValidationError):
        EmergencyCallSlotEntries.model_validate(
            [{"name": "caller_name", "value": ""}]
        )


def test_slot_entry_type_from_model_schema_is_required_and_non_null() -> None:
    entry_schema = _entry_schema_for("caller_name")

    assert entry_schema["required"] == ["name", "value"]
    assert entry_schema["properties"]["value"]["type"] == "string"
    assert entry_schema["properties"]["value"]["minLength"] == 1
