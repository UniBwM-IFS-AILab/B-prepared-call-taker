from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyjson5 as json

from experiments.nlu_eval.common.constants import MEDICAL_FIELDS

def test_main_dataset_loads_and_has_expected_coverage() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = (
        repo_root
        / "experiments"
        / "nlu_eval"
        / "datasets"
        / "main_master.jsonl"
    )

    items = pd.read_json(dataset_path, lines=True)

    assert len(items) == 180
    assert items["gt_non_medical_state"].fillna({}).map(bool).any()
    assert items["gt_medical_state"].map(
        lambda state: sum(value is True for value in state.values()) >= 2
    ).sum() >= 60

    questions_path = (
        repo_root
        / "src"
        / "ems_prepared"
        / "locale"
        / "questions.jsonc"
    )
    locale_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    intro_questions = locale_questions["en"]["Intro"]
    canonical_question = intro_questions[2]

    assert items["operator_question"].isin(intro_questions).all()
    assert items["operator_question"].eq(canonical_question).sum() >= 160

    covered_fields = {field for state in items["gt_medical_state"] for field in state}
    assert covered_fields == set(MEDICAL_FIELDS)
