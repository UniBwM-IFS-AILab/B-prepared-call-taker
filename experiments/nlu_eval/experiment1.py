#!/usr/bin/env python3
"""Run Experiment 1: outcome comparison between full agent and outcome baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from pydantic_core import to_jsonable_python

from ems_prepared.agents.state_fill_agent import build_state_fill_run_args, state_fill_prompt
from ems_prepared.dialogue_state.structured_output import NonEmptyStr
from ems_prepared.model.context import Settings
from ems_prepared.util.models import build_fallback_agent
from experiments.nlu_eval.common.metrics import experiment1_summary
from experiments.nlu_eval.common.models import PredictionRecord, RunConfig
from experiments.nlu_eval.common.runner import (
    FULL_AGENT_ID,
    append_jsonl,
    build_session_settings,
    init_run,
    normalize_existing_predictions,
    run_full_agent_once,
)

OUTCOME_BASELINE_ID = "outcome_baseline"


class OutcomeOnlyDecision(BaseModel):
    """Outcome-only baseline schema with hierarchical dispatch labels.

    The labels are cumulative:
    - `rd1=True` means the utterance supports a Rettungsdienst / ambulance response.
    - `rd2=True` means the utterance supports a Notarzteinsatz / emergency physician response and therefore also implies `rd1=True`.
    - `cpr_needed=True` means the utterance supports immediate CPR / tele-CPR and therefore also implies `rd2=True` and `rd1=True`.

    Multiple labels may be true at the same time to reflect this hierarchy.
    """

    rd1: bool | None = Field(
        default=None,
        description=(
            "Defines if an ambulance response is required. "
        ),
    )
    rd2: bool | None = Field(
        default=None,
        description=(
            "Defines if an emergency physician response is required. "
        ),
    )
    cpr_needed: bool | None = Field(
        default=None,
        description=(
            "Defines if the patient needs immediate CPR or tele-CPR guidance. "
        ),
    )

    def get_outcome(self) -> str | None:
        if self.cpr_needed:
            return "cpr"
        if self.rd2:
            return "rd2"
        if self.rd1:
            return "rd1"
        return None


def build_outcome_baseline_agent() -> Agent[Settings, OutcomeOnlyDecision | NonEmptyStr]:
    return build_fallback_agent(
        output_type=[OutcomeOnlyDecision, NonEmptyStr],
        instructions=state_fill_prompt.full_prompt,
        deps_type=Settings,
    )


async def run_outcome_baseline_once(
    *,
    agent: Agent[Settings, OutcomeOnlyDecision | NonEmptyStr],
    deps: Settings,
    experiment_id: str,
    item_id: str,
    repeat_index: int,
    operator_question: str,
    caller_utterance: str,
) -> PredictionRecord:
    run_args = build_state_fill_run_args(
        question=operator_question,
        user_response=caller_utterance,
        locale=deps.locale.value,
    )
    started = perf_counter()
    result = await agent.run(**run_args, deps=deps)

    latency_ms = (perf_counter() - started) * 1000
    output = result.output
    if isinstance(output, OutcomeOnlyDecision):
        return PredictionRecord(
            experiment_id=experiment_id,
            system_id=OUTCOME_BASELINE_ID,
            item_id=item_id,
            repeat_index=repeat_index,
            response_kind="state",
            predicted_outcome=output.get_outcome(),
            raw_output=output.model_dump(exclude_none=True),
            latency_ms=latency_ms,
            token_usage=to_jsonable_python(result.usage()),
        )

    return PredictionRecord(
        experiment_id=experiment_id,
        system_id=OUTCOME_BASELINE_ID,
        item_id=item_id,
        repeat_index=repeat_index,
        response_kind="followup",
        followup_text=str(output),
        raw_output=str(output) if isinstance(output, str) else to_jsonable_python(output),
        latency_ms=latency_ms,
        token_usage=to_jsonable_python(result.usage()),
    )


async def run_experiment1(*, run_config: RunConfig) -> Path:
    run_id, results_dir, raw_predictions_path = init_run(run_config, "experiment1_outcomes")
    items = pd.read_json(run_config.dataset_path, lines=True)
    existing_records = normalize_existing_predictions(raw_predictions_path)
    completed = set(
        existing_records[["system_id", "item_id", "repeat_index"]].itertuples(
            index=False, name=None
        )
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] starting experiment1 run_id={run_id} items={len(items)} repeats={run_config.repeats}",
        flush=True,
    )
    if completed:
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] resuming experiment1 run_id={run_id} completed={len(completed)}",
            flush=True,
        )

    baseline_agent = build_outcome_baseline_agent()

    total_calls = len(items) * run_config.repeats * 2
    call_index = 0
    for system_id, run_once, agent in (
        (FULL_AGENT_ID, run_full_agent_once, None),
        (OUTCOME_BASELINE_ID, run_outcome_baseline_once, baseline_agent),
    ):
        for item_index, item in enumerate(items.itertuples(index=False), start=1):
            for repeat_index in range(run_config.repeats):
                call_index += 1
                key = (system_id, item.id, repeat_index)
                if key in completed:
                    print(
                        f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] skipping completed "
                        f"experiment1 {call_index}/{total_calls} system={system_id} item={item.id} "
                        f"item_index={item_index}/{len(items)} repeat={repeat_index + 1}/{run_config.repeats}",
                        flush=True,
                    )
                    continue
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] experiment1 {call_index}/{total_calls} system={system_id} item={item.id} item_index={item_index}/{len(items)} repeat={repeat_index + 1}/{run_config.repeats}",
                    flush=True,
                )
                deps = build_session_settings(
                    run_id=run_id,
                    system_id=system_id,
                    item_id=item.id,
                    repeat_index=repeat_index,
                )
                if system_id == FULL_AGENT_ID:
                    try:
                        record = await run_once(
                            deps=deps,
                            experiment_id=run_config.experiment_id,
                            item_id=item.id,
                            repeat_index=repeat_index,
                            operator_question=item.operator_question,
                            caller_utterance=item.caller_utterance,
                        )
                    except Exception as exc:
                        print(
                            f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] failing experiment1 "
                            f"system={system_id} item={item.id} repeat={repeat_index + 1} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                else:
                    try:
                        record = await run_once(
                            agent=agent,
                            deps=deps,
                            experiment_id=run_config.experiment_id,
                            item_id=item.id,
                            repeat_index=repeat_index,
                            operator_question=item.operator_question,
                            caller_utterance=item.caller_utterance,
                        )
                    except Exception as exc:
                        print(
                            f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] failing experiment1 "
                            f"system={system_id} item={item.id} repeat={repeat_index + 1} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                append_jsonl(raw_predictions_path, record)
                completed.add(key)
                message = (
                    f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] wrote {record.response_kind} "
                    f"for system={system_id} item={item.id} repeat={repeat_index + 1}"
                )
                print(message, flush=True)

    records = pd.read_json(raw_predictions_path, lines=True)
    summary = {
        "experiment_id": run_config.experiment_id,
        "dataset_path": str(run_config.dataset_path),
        "run_id": run_id,
        "metrics": experiment1_summary(
            records=records,
            items=items,
        ),
        "generated_files": {
            "raw_predictions": str(raw_predictions_path),
        },
    }
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "experiment1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] finished experiment1 run_id={run_id}",
        flush=True,
    )
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    args = parser.parse_args()
    run_config = RunConfig.model_validate(json.loads(args.run_config.read_text(encoding="utf-8")))
    results_dir = asyncio.run(run_experiment1(run_config=run_config))
    print(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
