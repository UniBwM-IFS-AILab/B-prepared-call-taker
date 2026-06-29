import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictBool, create_model

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.generate_strict_model import inline_json_schema_refs
from ems_prepared.dialogue_state.schema_variants import (
    _field_attrs_with_non_empty_strings,
    _iter_exposed_slot_fields,
    field_attrs_from_original,
    strip_none_from_annotation,
)
from src.ems_prepared.dialogue_state.generate_strict_model import StrictEmergencyCall

type InstructAction = Literal[
    "stay_on_line",
    "stay_with_patient",
    "wait_for_ambulance",
    "call_again_if_change",
    "enable_responder_access",
]


class DialogueActBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        field_title_generator=lambda field_name, field_info: field_name,
        model_title_generator=lambda model: model.__name__.removesuffix("Act").lower(),
    )


def _read_only_default(value: str, *, title: str = "name") -> dict[str, Any]:
    return {
        "default": value,
        "title": title,
        "json_schema_extra": {"readOnly": True},
    }


def _read_only_slot_field_attrs(field_name: str, field_info) -> dict[str, Any]:
    attrs = field_attrs_from_original(field_info)
    json_schema_extra = dict(attrs.get("json_schema_extra") or {})
    json_schema_extra["readOnly"] = True
    attrs["default"] = field_name
    attrs["json_schema_extra"] = json_schema_extra
    return attrs


def _inform_value_field(field_info) -> tuple[Any, Any]:
    return (
        strip_none_from_annotation(field_info.rebuild_annotation()),
        Field(**_field_attrs_with_non_empty_strings(field_info)),
    )


def _slot_fields_for_act(
    act_name: str,
    field_name: str,
    field_info,
) -> dict[str, tuple[Any, Any]]:
    return {
        "name": (
            Literal[act_name],  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
            Field(
                default=act_name,
                title="name",
                json_schema_extra={"readOnly": True},
            ),
        ),
        "slot": (
            Literal[field_name],  # pyrefly: ignore [not-a-type] # ty: ignore [invalid-type-form]
            Field(**_read_only_slot_field_attrs(field_name, field_info)),
        ),
    }


def question_act_type_from_model(model: type[BaseModel]):
    """Build slot-specific question acts with read-only const name/slot fields."""
    variants: list[Any] = []

    for field_name, field_info in _iter_exposed_slot_fields(model):
        slot_attrs = field_attrs_from_original(field_info)
        variant_label = slot_attrs.get("title") or field_name
        variant = create_model(
            f"QuestionAct_{field_name}",
            __base__=DialogueActBase,
            __config__=ConfigDict(
                extra="forbid",
                title=variant_label,
            ),
            **_slot_fields_for_act("question", field_name, field_info),
        )
        variants.append(Annotated[variant, Field(title=variant_label)])

    return Annotated[
        Union[tuple(variants)],  # pyrefly: ignore [not-a-type]
        Field(title="question"),
    ]


def inform_act_type_from_model(model: type[BaseModel]):
    """Build slot-specific inform acts with typed value fields."""
    variants: list[Any] = []

    for field_name, field_info in _iter_exposed_slot_fields(model):
        slot_attrs = field_attrs_from_original(field_info)
        variant_label = slot_attrs.get("title") or field_name
        variant = create_model(  # type: ignore
            f"InformAct_{field_name}",
            __base__=DialogueActBase,
            __config__=ConfigDict(
                extra="forbid",
                title=variant_label,
            ),
            **{
                **_slot_fields_for_act("inform", field_name, field_info),
                "value": _inform_value_field(field_info),
            },
        )
        variants.append(Annotated[variant, Field(title=variant_label)])

    return Annotated[
        Union[tuple(variants)],  # pyrefly: ignore [not-a-type]
        Field(title="inform"),
    ]


class GreetingAct(DialogueActBase):
    name: Literal["greeting"] = Field(
        default="greeting",
        title="name",
        json_schema_extra={"readOnly": True},
    )


class GoodbyeAct(DialogueActBase):
    name: Literal["goodbye"] = Field(
        default="goodbye",
        title="name",
        json_schema_extra={"readOnly": True},
    )


type QuestionAct = question_act_type_from_model(StrictEmergencyCall)


type InformAct = inform_act_type_from_model(StrictEmergencyCall)


class ConfirmAct(DialogueActBase):
    # name: Literal["confirm"] = Field(**_read_only_default("confirm"))
    name: Literal["confirm"] = Field(
        default="confirm",
        title="name",
        json_schema_extra={"readOnly": True},
    )
    value: StrictBool


class FeedbackAct(DialogueActBase):
    # name: Literal["feedback"] = Field(**_read_only_default("feedback"))
    name: Literal["feedback"] = Field(
        default="feedback",
        title="name",
        json_schema_extra={"readOnly": True},
    )
    value: StrictBool


class InstructAct(DialogueActBase):
    name: Literal["instruct"] = Field(
        default="instruct",
        title="name",
        json_schema_extra={"readOnly": True},
    )
    value: InstructAction


type DialogueAct = (
    Annotated[GreetingAct, Field(title="greeting")]
    | Annotated[GoodbyeAct, Field(title="goodbye")]
    | Annotated[QuestionAct, Field(title="question")]
    | Annotated[InformAct, Field(title="inform")]
    | Annotated[ConfirmAct, Field(title="confirm")]
    | Annotated[FeedbackAct, Field(title="feedback")]
    | Annotated[InstructAct, Field(title="instruct")]
)


class DialogueActModel(RootModel[DialogueAct]):  # pyrefly: ignore [invalid-annotation]
    model_config = ConfigDict(title="Emergency Call Dialogue Act")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        return inline_json_schema_refs(schema)


if __name__ == "__main__":
    print(
        json.dumps(
            dict(DialogueActModel.model_json_schema()),
            ensure_ascii=False,
            indent=2,
        )
    )
