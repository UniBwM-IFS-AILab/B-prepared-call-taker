from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyjson5 as json

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency

def test_pilot_dataset_loads_and_has_expected_size() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "nlu_eval"
        / "datasets"
        / "pilot_master.jsonl"
    )

    items = pd.read_json(dataset_path, lines=True)

    assert len(items) == 24
    gt_outcomes = items["gt_medical_state"].map(lambda state: MedicalEmergency(**state).get_outcome())
    assert set(gt_outcomes) == {"rd1", "rd2", "cpr"}
    assert items["gt_non_medical_state"].fillna({}).map(bool).any()
    assert items["gt_medical_state"].map(
        lambda state: sum(value is True for value in state.values()) >= 2
    ).any()

    questions_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "ems_prepared"
        / "locale"
        / "questions.jsonc"
    )
    locale_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    intro_questions = locale_questions["en"]["Intro"]
    canonical_question = intro_questions[2]

    assert items["operator_question"].isin(intro_questions).all()
    assert items["operator_question"].eq(canonical_question).sum() >= 20
