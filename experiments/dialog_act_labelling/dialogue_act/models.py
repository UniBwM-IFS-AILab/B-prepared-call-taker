import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictBool, StrictInt

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.generate_strict_model import inline_json_schema_refs
from ems_prepared.dialogue_state.schema_variants import slot_name_type_from_model

type SlotId = slot_name_type_from_model(EmergencyCall)
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


class GreetingAct(DialogueActBase):
    act: Literal["greeting"]


class GoodbyeAct(DialogueActBase):
    act: Literal["goodbye"]


class QuestionAct(DialogueActBase):
    act: Literal["question"]
    slot: SlotId


class InformAct(DialogueActBase):
    act: Literal["inform"]
    slot: SlotId
    value: Annotated[str, Field(min_length=1)] | StrictBool | StrictInt


class ConfirmAct(DialogueActBase):
    act: Literal["confirm"]
    value: StrictBool


class FeedbackAct(DialogueActBase):
    act: Literal["feedback"]
    value: StrictBool


class InstructAct(DialogueActBase):
    act: Literal["instruct"]
    action: InstructAction


type DialogueAct = Annotated[
    GreetingAct
    | GoodbyeAct
    | QuestionAct
    | InformAct
    | ConfirmAct
    | FeedbackAct
    | InstructAct,
    Field(discriminator="act"),
]


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
