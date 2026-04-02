"""Shared runtime and artifact helpers for the NLU benchmark."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ems_prepared.agents.state_fill_agent import run_state_fill
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from ems_prepared.model.context import Locale, Settings
from experiments.nlu_eval.common.constants import MEDICAL_FIELDS, NON_MEDICAL_FIELDS
from experiments.nlu_eval.common.models import PredictionRecord, RunConfig

FULL_AGENT_ID = "full_agent"


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json())
        handle.write("\n")


def normalize_existing_predictions(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["system_id", "item_id", "repeat_index", "response_kind"])
    records = pd.read_json(path, lines=True)
    records = records.loc[records["response_kind"] != "parse_error"].drop_duplicates(
        subset=["system_id", "item_id", "repeat_index"],
        keep="last",
    )
    path.write_text("", encoding="utf-8")
    if not records.empty:
        path.write_text(records.to_json(orient="records", lines=True), encoding="utf-8")
    return records


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def init_run(run_config: RunConfig, output_dir: str) -> tuple[str, Path, Path]:
    run_id = run_config.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = run_config.results_root / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_predictions_path = results_dir / output_dir / "raw_predictions.jsonl"
    raw_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_id": run_config.experiment_id,
                "dataset_path": str(run_config.dataset_path),
                "raw_predictions_path": str(raw_predictions_path),
                "repeats": run_config.repeats,
                "git_commit": git_commit(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return run_id, results_dir, raw_predictions_path


def build_session_settings(
    *,
    run_id: str,
    system_id: str,
    item_id: str,
    repeat_index: int,
) -> Settings:
    user_id = uuid5(NAMESPACE_URL, f"nlu_eval:{run_id}:{system_id}")
    session_id = uuid5(NAMESPACE_URL, f"nlu_eval:{run_id}:{system_id}:{item_id}:{repeat_index}")
    return Settings(
        name="nlu_eval",
        locale=Locale.EN,
        user_id=user_id,
        session_id=session_id,
        experiment_name=f"nlu_eval/results/{run_id}",
        scenario_name=item_id,
        policy_name=system_id,
    )


async def run_full_agent_once(
    *,
    deps: Settings,
    experiment_id: str,
    item_id: str,
    repeat_index: int,
    operator_question: str,
    caller_utterance: str,
) -> PredictionRecord:
    started = perf_counter()
    result = await run_state_fill(
        question=operator_question,
        user_response=caller_utterance,
        deps=deps,
    )

    latency_ms = (perf_counter() - started) * 1000
    output = result.output

    if isinstance(output, EmergencyCall):
        dumped = output.model_dump(exclude_none=True)
        predicted_medical_state = {name: dumped[name] for name in MEDICAL_FIELDS if name in dumped}
        predicted_non_medical_state = {
            name: dumped[name] for name in NON_MEDICAL_FIELDS if name in dumped
        }
        return PredictionRecord(
            experiment_id=experiment_id,
            system_id=FULL_AGENT_ID,
            item_id=item_id,
            repeat_index=repeat_index,
            response_kind="state",
            predicted_medical_state=predicted_medical_state,
            predicted_non_medical_state=predicted_non_medical_state,
            predicted_outcome=MedicalEmergency(**predicted_medical_state).get_outcome(),
            raw_output=output.model_dump(exclude_none=True),
            latency_ms=latency_ms,
            token_usage=to_jsonable_python(result.usage()),
        )

    return PredictionRecord(
        experiment_id=experiment_id,
        system_id=FULL_AGENT_ID,
        item_id=item_id,
        repeat_index=repeat_index,
        response_kind="followup",
        followup_text=str(output),
        raw_output=str(output) if isinstance(output, str) else to_jsonable_python(output),
        latency_ms=latency_ms,
        token_usage=to_jsonable_python(result.usage()),
    )
