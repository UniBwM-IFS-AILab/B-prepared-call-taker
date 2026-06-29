from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .comm_func_schema import FunctionLabel

LOGGER = logging.getLogger("dialog_act_labelling")
MAX_ACTS_PER_TURN = 3

type valid_speaker = Literal["CALLER", "DISPATCHER", "EXTRA", "PATIENT", "BYSTANDER"]


class DialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: valid_speaker
    utterance: str = Field(min_length=1)


class DialogueLabellingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: list[DialogueTurn] = Field(default_factory=list)
    speaker: valid_speaker
    utterance: str = Field(min_length=1)


class DialogueLabellingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acts: list[FunctionLabel] = Field(
        default_factory=list, max_length=MAX_ACTS_PER_TURN
    )

    @model_validator(mode="after")
    def normalize_acts(self) -> "DialogueLabellingOutput":
        deduplicated_acts: list[FunctionLabel] = []
        seen: set[str] = set()
        for act in self.acts:
            signature = repr(act)
            if signature in seen:
                LOGGER.warning("duplicate_dialogue_act_removed: %s", act)
                continue
            seen.add(signature)
            deduplicated_acts.append(act)
            if len(deduplicated_acts) >= MAX_ACTS_PER_TURN:
                break
        self.acts = deduplicated_acts
        return self
