from __future__ import annotations

from types import NoneType, UnionType
from typing import Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model


def field_attrs_from_original(
    field_info, *, default_none: bool = False
) -> dict[str, Any]:
    """Copy `Field(...)` metadata from an existing model field.

    This preserves descriptive metadata such as `title`, `description`, and
    `examples`, removes `default_factory`, and either drops the default or sets
    it to `None`, depending on `default_none`.

    Example output:
    {
      "default": null,
      "title": "Caller Name",
      "description": "Name of the person making the emergency call."
    }
    """
    attrs = field_info.asdict()["attributes"].copy()

    if default_none:
        attrs["default"] = None
    else:
        attrs.pop("default", None)

    attrs.pop("default_factory", None)

    examples = attrs.get("examples")
    if isinstance(examples, list):
        attrs["examples"] = [example for example in examples if example is not None]
    elif isinstance(examples, tuple):
        attrs["examples"] = tuple(
            example for example in examples if example is not None
        )

    return attrs


def strip_none_from_annotation(annotation: Any) -> Any:
    """Remove `None` from an annotation while preserving other structure.

    This rewrites nullable types such as `str | None` into `str`, and does the
    same recursively for `Annotated[...]` and union types.

    Example output:
    {
      "input": "str | null",
      "output": "str"
    }
    """
    if isinstance(annotation, TypeAliasType):
        return strip_none_from_annotation(annotation.__value__)

    origin = get_origin(annotation)

    if origin is Annotated:
        annotated_type, *metadata = get_args(annotation)
        return Annotated[strip_none_from_annotation(annotated_type), *metadata]

    if origin in {Union, UnionType}:
        non_none_args = [
            strip_none_from_annotation(arg)
            for arg in get_args(annotation)
            if arg is not NoneType
        ]
        if len(non_none_args) == 1:
            return non_none_args[0]

        merged = non_none_args[0]
        for arg in non_none_args[1:]:
            merged = merged | arg
        return merged

    return annotation


def _iter_exposed_slot_fields(model: type[BaseModel]):
    """Yield fields that are exposed as JSON schema properties for a model."""
    exposed_field_names = set(model.model_json_schema().get("properties", {}))

    for field_name, field_info in model.model_fields.items():
        if field_name in exposed_field_names:
            yield field_name, field_info


def _annotation_base_type(annotation: Any) -> Any:
    if isinstance(annotation, TypeAliasType):
        return _annotation_base_type(annotation.__value__)

    origin = get_origin(annotation)
    if origin is Annotated:
        annotated_type, *_ = get_args(annotation)
        return _annotation_base_type(annotated_type)

    return annotation


def _field_attrs_with_non_empty_strings(
    field_info, *, default_none: bool = False
) -> dict[str, Any]:
    attrs = field_attrs_from_original(field_info, default_none=default_none)
    non_null_annotation = strip_none_from_annotation(field_info.annotation)

    if _annotation_base_type(non_null_annotation) is str:
        attrs["min_length"] = 1

    return attrs


def slot_name_type_from_model(model: type[BaseModel]):
    """Build a slot-name union type from the model's exposed JSON properties.

    The returned type is a union of annotated string literals, one per exposed
    field. When used in a schema, this becomes an `anyOf` over the allowed slot
    names with each branch carrying the original field description.

    Example schema fragment:
    {
      "items": {
        "anyOf": [
          {"const": "caller_name", "description": "..."},
          {"const": "emergency_location", "description": "..."}
        ]
      }
    }
    """
    field_types = [
        Annotated[
            Literal[field_name],  # ty:ignore[invalid-type-form] # pyrefly: ignore [not-a-type]
            Field(**field_attrs_from_original(field_info)),
        ]
        for field_name, field_info in _iter_exposed_slot_fields(model)
    ]

    return Union[tuple(field_types)]  # pyrefly: ignore [not-a-type]


def slot_entry_type_from_model(model: type[BaseModel]):
    """Build a union of required `{name, value}` models for exposed fields.

    Each union branch fixes `name` to one literal field name and constrains
    `value` to the corresponding non-null field type. String values are required
    to be non-empty.

    Example schema fragment:
    {
      "items": {
        "anyOf": [
          {
            "type": "object",
            "properties": {
              "name": {"const": "caller_name"},
              "value": {"type": "string", "minLength": 1}
            },
            "required": ["name", "value"]
          }
        ]
      }
    }
    """
    variants: list[type[BaseModel]] = []

    for field_name, field_info in _iter_exposed_slot_fields(model):
        variant = create_model(
            f"{model.__name__}{''.join(part.title() for part in field_name.split('_'))}Slot",
            __config__=ConfigDict(extra="forbid"),
            name=(
                Literal[field_name],  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
                Field(**field_attrs_from_original(field_info)),
            ),
            value=(
                strip_none_from_annotation(field_info.annotation),
                Field(**_field_attrs_with_non_empty_strings(field_info)),
            ),
        )
        variants.append(variant)

    return Union[tuple(variants)]  # pyrefly: ignore [not-a-type]
