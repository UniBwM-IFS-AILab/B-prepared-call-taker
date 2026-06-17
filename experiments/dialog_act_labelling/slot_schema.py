import json
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, create_model

from ems_prepared.dialogue_state.generate_strict_model import inline_json_schema_refs
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.schema_variants import (
    field_attrs_from_original,
    slot_entry_type_from_model,
    slot_name_type_from_model,
)


class InlinedSchemaRootModel(RootModel):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        return inline_json_schema_refs(schema)


def slot_value_schema(field_name: str) -> dict:
    """Build an object schema for the value of one `EmergencyCall` slot.

    The result is a single-field object schema with a `value` property whose
    type and metadata are copied from the matching `EmergencyCall` field.

    Example output:
    {
      "type": "object",
      "properties": {
        "value": {
          "anyOf": [{"type": "string"}, {"type": "null"}],
          "title": "Caller Name"
        }
      },
      "required": ["value"]
    }
    """
    exposed_field_names = set(EmergencyCall.model_json_schema().get("properties", {}))
    if field_name not in exposed_field_names:
        raise ValueError(f"{field_name!r} is not exposed in EmergencyCall JSON schema.")

    field_info = EmergencyCall.model_fields[field_name]
    schema_model = create_model(
        f"EmergencyCall{''.join(part.title() for part in field_name.split('_'))}ValueOutput",
        __config__=ConfigDict(extra="forbid"),
        value=(field_info.annotation, Field(**field_attrs_from_original(field_info))),
    )
    schema = inline_json_schema_refs(schema_model.model_json_schema())
    value_schema = schema["properties"]["value"]

    if value_schema.get("type") == "string" and "const" not in value_schema:
        value_schema.setdefault("minLength", 1)

    for branch in value_schema.get("anyOf", []):
        if branch.get("type") == "string" and "const" not in branch:
            branch.setdefault("minLength", 1)

    return schema

class EmergencyCallSlotNames(RootModel[list[slot_name_type_from_model(EmergencyCall)]]):  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
    pass


class EmergencyCallSlotEntries(
    InlinedSchemaRootModel[list[slot_entry_type_from_model(EmergencyCall)]]  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
):
    pass


if __name__ == "__main__":
    # T = TypeVar('T', bound=slot_name_type_from_model(EmergencyCall))
    class Dummy(BaseModel):
        slot: slot_name_type_from_model(EmergencyCall)

    # print(json.dumps(inline_json_schema_refs(Dummy.model_json_schema()), indent=2))
    print(json.dumps(Dummy.model_json_schema(), indent=2))

    assert Dummy.model_json_schema() == inline_json_schema_refs(
        Dummy.model_json_schema()
    )
