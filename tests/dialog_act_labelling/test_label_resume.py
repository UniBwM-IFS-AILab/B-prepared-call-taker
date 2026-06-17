from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.dialog_act_labelling.label import label_dataset
from experiments.dialog_act_labelling.resume_checkpoint import load_resume_checkpoint
from experiments.dialog_act_labelling.schemas import DialogueLabellingOutput


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class FakeAgent:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def run_sync(self, payload: str) -> SimpleNamespace:
        self.inputs.append(json.loads(payload))
        return SimpleNamespace(output=DialogueLabellingOutput(acts=[]))


def test_label_dataset_skips_existing_rows_without_losing_context(
    monkeypatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "dialogues.jsonl"
    output_path = tmp_path / "dialog_acts.jsonl"
    rows = [
        {
            "dialog_id": "dialog-1",
            "turn_index": 0,
            "speaker": "DISPATCHER",
            "text": "Emergency services, what happened?",
        },
        {
            "dialog_id": "dialog-1",
            "turn_index": 1,
            "speaker": "CALLER",
            "text": "My father collapsed.",
        },
        {
            "dialog_id": "dialog-1",
            "turn_index": 1,
            "speaker": "CALLER",
            "text": "My father collapsed.",
        },
        {
            "dialog_id": "dialog-1",
            "turn_index": 2,
            "speaker": "DISPATCHER",
            "text": "Is he breathing?",
        },
    ]
    _write_jsonl(input_path, rows)
    _write_jsonl(
        output_path,
        [
            rows[0] | {"acts": []},
            rows[1] | {"acts": []},
        ],
    )

    fake_agent = FakeAgent()
    monkeypatch.setattr(
        "experiments.dialog_act_labelling.label.build_labelling_agent",
        lambda **_kwargs: fake_agent,
    )

    label_dataset(input_path, output_path, context_window=2)

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [(row["dialog_id"], row["turn_index"]) for row in output_rows] == [
        ("dialog-1", 0),
        ("dialog-1", 1),
        ("dialog-1", 2),
    ]
    assert fake_agent.inputs == [
        {
            "context": [
                {
                    "speaker": "DISPATCHER",
                    "utterance": "Emergency services, what happened?",
                },
                {
                    "speaker": "CALLER",
                    "utterance": "My father collapsed.",
                },
            ],
            "speaker": "DISPATCHER",
            "utterance": "Is he breathing?",
        }
    ]


def test_load_resume_checkpoint_reads_last_nonempty_row(tmp_path: Path) -> None:
    output_path = tmp_path / "dialog_acts.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dialog_id": "dialog-1",
                        "turn_index": 0,
                        "speaker": "DISPATCHER",
                        "text": "Hello",
                        "acts": [],
                    }
                ),
                json.dumps(
                    {
                        "dialog_id": "dialog-1",
                        "turn_index": 1,
                        "speaker": "CALLER",
                        "text": "Help",
                        "acts": [],
                    }
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert load_resume_checkpoint(output_path) == ("dialog-1", 1)
