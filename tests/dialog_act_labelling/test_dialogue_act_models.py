from __future__ import annotations

import pytest
from pydantic import ValidationError

from experiments.dialog_act_labelling.dialogue_act.models import DialogueActModel


def test_dialogue_act_model_accepts_valid_variants() -> None:
    assert DialogueActModel.model_validate({"act": "greeting"}).root.act == "greeting"
    assert DialogueActModel.model_validate(
        {"act": "question", "slot": "caller_name"}
    ).root.act == "question"
    assert DialogueActModel.model_validate(
        {"act": "inform", "slot": "caller_name", "value": "Max"}
    ).root.act == "inform"
    assert DialogueActModel.model_validate(
        {"act": "confirm", "value": True}
    ).root.act == "confirm"
    assert DialogueActModel.model_validate(
        {"act": "instruct", "action": "stay_on_line"}
    ).root.act == "instruct"


def test_dialogue_act_model_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        DialogueActModel.model_validate({"act": "question"})

    with pytest.raises(ValidationError):
        DialogueActModel.model_validate({"act": "inform", "slot": "caller_name"})


def test_dialogue_act_model_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        DialogueActModel.model_validate(
            {"act": "inform", "slot": "caller_name", "value": ""}
        )

    with pytest.raises(ValidationError):
        DialogueActModel.model_validate({"act": "confirm", "value": "yes"})

    with pytest.raises(ValidationError):
        DialogueActModel.model_validate(
            {"act": "instruct", "action": "do_something_else"}
        )


def test_dialogue_act_schema_titles() -> None:
    schema = DialogueActModel.model_json_schema()

    assert schema["title"] == "Medical Emergency Call Dialogue Act"

    greeting = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["act"]["const"] == "greeting"
    )
    assert greeting["title"] == "greeting"
    assert greeting["type"] == "object"
