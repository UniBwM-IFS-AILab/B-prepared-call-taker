from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.generate_strict_model import inline_json_schema_refs

from .data_row import DataRow


def _build_property_groups_from_declaring_classes(
    model: type[BaseModel],
) -> tuple[list[str], dict[str, str]]:
    group_order: list[str] = []
    field_groups: dict[str, str] = {}

    for field_name in model.model_fields:
        declaring_class = next(
            (
                cls
                for cls in model.__mro__
                if field_name in getattr(cls, "__annotations__", {})
            ),
            None,
        )
        if declaring_class is None:
            continue

        name = declaring_class.__name__
        group_title = re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()
        group_title = re.sub(r"\bCpr\b", "CPR", group_title)

        field_groups[field_name] = group_title
        if group_title not in group_order:
            group_order.append(group_title)

    return group_order, field_groups


def _is_object_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type") == "object" or (
        isinstance(schema.get("properties"), dict)
        and schema.get("additionalProperties") is False
    )


def _has_only_read_only_properties(schema: dict[str, Any]) -> bool:
    properties = schema.get("properties")
    return (
        isinstance(properties, dict)
        and bool(properties)
        and all(
            property_schema.get("readOnly") is True
            for property_schema in properties.values()
            if isinstance(property_schema, dict)
        )
    )


def rewrite_boolean_defaults(schema: dict[str, Any]) -> None:
    for field_schema in schema.get("properties", {}).values():
        if (
            isinstance(field_schema, dict)
            and field_schema.get("type") == "boolean"
            and field_schema.get("default") is None
        ):
            field_schema["default"] = False


def _patch_state_schema(schema: dict[str, Any]) -> None:
    schema["x-enablePropertiesToggle"] = True

    group_order, groups = _build_property_groups_from_declaring_classes(EmergencyCall)
    schema["x-propGroupOrder"] = group_order

    for field_name, field_schema in schema.get("properties", {}).items():
        group_title = groups.get(field_name)
        if group_title is not None:
            field_schema["x-propGroup"] = group_title

    rewrite_boolean_defaults(schema)


def _walk_schema(node: Any, visit) -> None:
    if isinstance(node, dict):
        visit(node)
        for value in node.values():
            _walk_schema(value, visit)
        return

    if isinstance(node, list):
        for item in node:
            _walk_schema(item, visit)


def _patch_dialogue_act_object_schema(schema: dict[str, Any]) -> None:
    if not _is_object_schema(schema):
        return

    schema["x-enablePropertiesToggle"] = False
    if _has_only_read_only_properties(schema):
        schema["x-startCollapsed"] = True


def data_row_jedison_schema_patch_schema() -> dict[str, Any]:
    schema = copy.deepcopy(inline_json_schema_refs(DataRow.model_json_schema()))
    schema["x-enableCollapseToggle"] = False

    state_schema = schema["properties"]["state"]
    _patch_state_schema(state_schema)

    dialogue_acts_items_schema = schema["properties"]["dialogue_acts"]["items"]
    _walk_schema(dialogue_acts_items_schema, _patch_dialogue_act_object_schema)

    return schema


if __name__ == "__main__":
    print(
        json.dumps(data_row_jedison_schema_patch_schema(), ensure_ascii=False, indent=2)
    )
