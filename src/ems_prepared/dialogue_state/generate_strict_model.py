from typing import TypeVar

import jsonref
from pydantic import BaseModel, ConfigDict, Field, create_model

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.schema_variants import (
    field_attrs_from_original,
    strip_none_from_annotation,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class InlinedSchemaBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        return inline_json_schema_refs(schema)


def inline_json_schema_refs(schema: dict) -> dict:
    """Inline `$ref` targets in a JSON schema and drop the now-unused `$defs`.

    This is used to produce a flatter schema for consumers that handle inline
    object schemas better than schemas with indirection.

    Example output:
    {
      "type": "object",
      "properties": {
        "caller_name": {"type": "string"}
      }
    }
    """
    schema = jsonref.replace_refs(
        schema,
        proxies=False,
        merge_props=True,
    )
    schema.pop("$defs", None)
    return schema

def generate_strict_model(model: type[ModelT]) -> type[InlinedSchemaBaseModel]:
    """Generate a model with the same fields but non-null input types.

    The generated model keeps the original field descriptions and initializes
    omitted fields to `None`, but explicit `null` inputs are rejected because
    `None` has been removed from each field annotation.

    Example schema fragment:
    {
      "type": "object",
      "properties": {
        "caller_name": {
          "type": "string",
          "default": null
        }
      }
    }
    """
    exposed_field_names = set(model.model_json_schema().get("properties", {}))
    strict_fields: dict[str, tuple[Any, Field]] = {}

    for field_name, field_info in model.model_fields.items():
        if field_name not in exposed_field_names:
            continue

        strict_fields[field_name] = (
            strip_none_from_annotation(field_info.annotation),
            Field(**field_attrs_from_original(field_info, default_none=True)),
        )

    return create_model(
        f"Strict{model.__name__}",
        __base__=InlinedSchemaBaseModel,
        __module__=__name__,
        **strict_fields,
    )


StrictEmergencyCall = generate_strict_model(EmergencyCall)


if __name__ == "__main__":
    import json

    print(json.dumps(StrictEmergencyCall.model_json_schema(), indent=2))
