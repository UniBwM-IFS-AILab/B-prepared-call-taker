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
    raw_predictions_path = (
        results_root / "resume-e1" / "experiment1_raw_predictions.jsonl"
    )
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

    monkeypatch.setattr(
        "experiments.nlu_eval.experiment1.run_full_agent_once", should_not_run
    )
    monkeypatch.setattr(
        "experiments.nlu_eval.experiment1.run_outcome_baseline_once", should_not_run
    )

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
    assert (results_dir / "experiment1_summary.json").exists()


def test_experiment2_resumes_completed_rows(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    raw_predictions_path = (
        results_root / "resume-e2" / "experiment2_raw_predictions.jsonl"
    )
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

    monkeypatch.setattr(
        "experiments.nlu_eval.experiment2.run_full_agent_once", should_not_run
    )

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
    assert (results_dir / "experiment2_summary.json").exists()


def test_experiment2_drops_legacy_parse_errors_before_resuming(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    raw_predictions_path = (
        results_root / "resume-e2-clean" / "experiment2_raw_predictions.jsonl"
    )
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

    monkeypatch.setattr(
        "experiments.nlu_eval.experiment2.run_full_agent_once", should_not_run
    )

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


def test_experiment1_records_parse_errors_and_keeps_running(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)

    async def always_fail(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("experiments.nlu_eval.experiment1.run_full_agent_once", always_fail)
    monkeypatch.setattr("experiments.nlu_eval.experiment1.run_outcome_baseline_once", always_fail)
    monkeypatch.setattr("experiments.nlu_eval.experiment1.build_outcome_baseline_agent", lambda **kwargs: object())

    results_dir = asyncio.run(
        run_experiment1(
            run_config=RunConfig(
                experiment_id="experiment1",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="parse-error-e1",
            )
        )
    )

    records = pd.read_json(results_root / "parse-error-e1" / "experiment1_raw_predictions.jsonl", lines=True)
    assert records["response_kind"].tolist() == ["parse_error", "parse_error"]
    assert (results_dir / "experiment1_summary.json").exists()


def test_experiment2_records_parse_errors_and_keeps_running(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)

    async def always_fail(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("experiments.nlu_eval.experiment2.run_full_agent_once", always_fail)

    results_dir = asyncio.run(
        run_experiment2(
            run_config=RunConfig(
                experiment_id="experiment2",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="parse-error-e2",
            )
        )
    )

    records = pd.read_json(results_root / "parse-error-e2" / "experiment2_raw_predictions.jsonl", lines=True)
    assert records["response_kind"].tolist() == ["parse_error"]
    assert (results_dir / "experiment2_summary.json").exists()



def test_experiment1_uses_shared_rate_limiter(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    captured: dict[str, object] = {}

    async def fake_full_agent(**kwargs):
        captured["full_limiter"] = kwargs["rate_limiter"]
        return PredictionRecord(
            experiment_id="experiment1",
            system_id="full_agent",
            item_id=kwargs["item_id"],
            repeat_index=kwargs["repeat_index"],
            response_kind="state",
            predicted_outcome="rd1",
            latency_ms=1.0,
        )

    def fake_build_baseline_agent(*, rate_limiter=None):
        captured["baseline_limiter"] = rate_limiter
        return object()

    async def fake_run_outcome_baseline_once(**kwargs):
        return PredictionRecord(
            experiment_id="experiment1",
            system_id="outcome_baseline",
            item_id=kwargs["item_id"],
            repeat_index=kwargs["repeat_index"],
            response_kind="state",
            predicted_outcome="rd1",
            latency_ms=1.0,
        )

    monkeypatch.setattr("experiments.nlu_eval.experiment1.run_full_agent_once", fake_full_agent)
    monkeypatch.setattr(
        "experiments.nlu_eval.experiment1.build_outcome_baseline_agent",
        fake_build_baseline_agent,
    )
    monkeypatch.setattr(
        "experiments.nlu_eval.experiment1.run_outcome_baseline_once",
        fake_run_outcome_baseline_once,
    )

    asyncio.run(
        run_experiment1(
            run_config=RunConfig(
                experiment_id="experiment1",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="rate-limit-e1",
                requests_per_minute=10,
            )
        )
    )

    manifest = json.loads((results_root / "rate-limit-e1" / "manifest.json").read_text())
    assert manifest["requests_per_minute"] == 10
    assert captured["full_limiter"] is captured["baseline_limiter"]
    assert captured["full_limiter"].requests_per_minute == 10



def test_experiment2_passes_configured_rate_limiter(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    results_root = tmp_path / "results"
    _write_dataset(dataset_path)
    captured: dict[str, object] = {}

    async def fake_full_agent(**kwargs):
        captured["limiter"] = kwargs["rate_limiter"]
        return PredictionRecord(
            experiment_id="experiment2",
            system_id="full_agent",
            item_id=kwargs["item_id"],
            repeat_index=kwargs["repeat_index"],
            response_kind="state",
            predicted_medical_state={"acute_speech_disorder": True},
            predicted_outcome="rd1",
            latency_ms=1.0,
        )

    monkeypatch.setattr("experiments.nlu_eval.experiment2.run_full_agent_once", fake_full_agent)

    asyncio.run(
        run_experiment2(
            run_config=RunConfig(
                experiment_id="experiment2",
                dataset_path=dataset_path,
                results_root=results_root,
                repeats=1,
                run_id="rate-limit-e2",
                requests_per_minute=10,
            )
        )
    )

    manifest = json.loads((results_root / "rate-limit-e2" / "manifest.json").read_text())
    assert manifest["requests_per_minute"] == 10
    assert captured["limiter"].requests_per_minute == 10
