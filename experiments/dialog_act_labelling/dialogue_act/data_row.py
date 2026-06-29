import json

from pydantic import BaseModel, ConfigDict, Field

from ems_prepared.dialogue_state.generate_strict_model import StrictEmergencyCall

from ..schemas import valid_speaker
from .models import DialogueActModel


class DataRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dialog_id: int = Field(frozen=True, json_schema_extra={"readOnly": True})
    turn_index: int = Field(frozen=True, json_schema_extra={"readOnly": True})
    speaker: valid_speaker = Field(title="Speaker")
    text: str = Field(min_length=1)
    state: StrictEmergencyCall = Field(title="State")  # type: ignore
    dialogue_acts: list[DialogueActModel] = Field(default_factory=list)


if __name__ == "__main__":
    print(
        json.dumps(
            dict(DataRow.model_json_schema()),
            ensure_ascii=False,
            indent=2,
        )
    )
