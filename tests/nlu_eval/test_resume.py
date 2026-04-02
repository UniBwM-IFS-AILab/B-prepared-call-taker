from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from experiments.nlu_eval.common.models import PredictionRecord, RunConfig
from experiments.nlu_eval.experiment1 import run_experiment1
from experiments.nlu_eval.experiment2 import run_experiment2


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_dataset(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "id": "item-1",
                "operator_question": "What has just happened acutely?",
                "caller_utterance": "She suddenly cannot speak properly.",
                "gt_medical_state": {"acute_speech_disorder": True},
                "gt_non_medical_state": {},
                "notes": "",
            }
        ],
    )


def test_experiment1_resumes_completed_rows(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    raw_predictions_path = results_root / "resume-e1" / "experiment1_outcomes" / "raw_predictions.jsonl"
    _write_jsonl(
        raw_predictions_path,
        [
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
        ],
    )

    async def should_not_run(**kwargs):
        raise AssertionError("resume should skip completed rows")

    monkeypatch.setattr("experiments.nlu_eval.experiment1.run_full_agent_once", should_not_run)
    monkeypatch.setattr("experiments.nlu_eval.experiment1.run_outcome_baseline_once", should_not_run)

    results_dir = asyncio.run(
        run_experiment1(
            run_config=RunConfig(
                experiment_id="experiment1",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="resume-e1",
            )
        )
    )

    records = pd.read_json(raw_predictions_path, lines=True)
    assert len(records) == 2
    assert (results_dir / "summaries" / "experiment1_summary.json").exists()


def test_experiment2_resumes_completed_rows(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    raw_predictions_path = results_root / "resume-e2" / "experiment2_slots" / "raw_predictions.jsonl"
    _write_jsonl(
        raw_predictions_path,
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={"acute_speech_disorder": True},
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump()
        ],
    )

    async def should_not_run(**kwargs):
        raise AssertionError("resume should skip completed rows")

    monkeypatch.setattr("experiments.nlu_eval.experiment2.run_full_agent_once", should_not_run)

    results_dir = asyncio.run(
        run_experiment2(
            run_config=RunConfig(
                experiment_id="experiment2",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="resume-e2",
            )
        )
    )

    records = pd.read_json(raw_predictions_path, lines=True)
    assert len(records) == 1
    assert (results_dir / "summaries" / "experiment2_summary.json").exists()


def test_experiment2_drops_legacy_parse_errors_before_resuming(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    raw_predictions_path = results_root / "resume-e2-clean" / "experiment2_slots" / "raw_predictions.jsonl"
    _write_jsonl(
        raw_predictions_path,
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="parse_error",
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={"acute_speech_disorder": True},
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
        ],
    )

    async def should_not_run(**kwargs):
        raise AssertionError("resume should skip completed rows")

    monkeypatch.setattr("experiments.nlu_eval.experiment2.run_full_agent_once", should_not_run)

    asyncio.run(
        run_experiment2(
            run_config=RunConfig(
                experiment_id="experiment2",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="resume-e2-clean",
            )
        )
    )

    records = pd.read_json(raw_predictions_path, lines=True)
    assert len(records) == 1
    assert records.iloc[0]["response_kind"] == "state"
