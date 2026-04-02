#!/usr/bin/env python3
"""Run Experiment 2: slot extraction benchmark for the full agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.nlu_eval.common.metrics import experiment2_summary
from experiments.nlu_eval.common.models import RunConfig
from experiments.nlu_eval.common.runner import (
    FULL_AGENT_ID,
    append_jsonl,
    build_session_settings,
    init_run,
    normalize_existing_predictions,
    run_full_agent_once,
)


async def run_experiment2(*, run_config: RunConfig) -> Path:
    run_id, results_dir, raw_predictions_path = init_run(run_config, "experiment2_slots")
    items = pd.read_json(run_config.dataset_path, lines=True)
    existing_records = normalize_existing_predictions(raw_predictions_path)
    completed = set(
        existing_records[["system_id", "item_id", "repeat_index"]].itertuples(
            index=False, name=None
        )
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] starting experiment2 run_id={run_id} items={len(items)} repeats={run_config.repeats}",
        flush=True,
    )
    if completed:
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] resuming experiment2 run_id={run_id} completed={len(completed)}",
            flush=True,
        )

    total_calls = len(items) * run_config.repeats
    call_index = 0
    for item_index, item in enumerate(items.itertuples(index=False), start=1):
        for repeat_index in range(run_config.repeats):
            call_index += 1
            key = (FULL_AGENT_ID, item.id, repeat_index)
            if key in completed:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] skipping completed "
                    f"experiment2 {call_index}/{total_calls} system={FULL_AGENT_ID} item={item.id} "
                    f"item_index={item_index}/{len(items)} repeat={repeat_index + 1}/{run_config.repeats}",
                    flush=True,
                )
                continue
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] experiment2 {call_index}/{total_calls} system={FULL_AGENT_ID} item={item.id} item_index={item_index}/{len(items)} repeat={repeat_index + 1}/{run_config.repeats}",
                flush=True,
            )
            deps = build_session_settings(
                run_id=run_id,
                system_id=FULL_AGENT_ID,
                item_id=item.id,
                repeat_index=repeat_index,
            )
            try:
                record = await run_full_agent_once(
                    deps=deps,
                    experiment_id=run_config.experiment_id,
                    item_id=item.id,
                    repeat_index=repeat_index,
                    operator_question=item.operator_question,
                    caller_utterance=item.caller_utterance,
                )
            except Exception as exc:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] failing experiment2 "
                    f"system={FULL_AGENT_ID} item={item.id} repeat={repeat_index + 1} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            append_jsonl(raw_predictions_path, record)
            completed.add(key)
            message = (
                f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] wrote {record.response_kind} "
                f"for system={FULL_AGENT_ID} item={item.id} repeat={repeat_index + 1}"
            )
            print(message, flush=True)

    records = pd.read_json(raw_predictions_path, lines=True)
    summary = {
        "experiment_id": run_config.experiment_id,
        "dataset_path": str(run_config.dataset_path),
        "run_id": run_id,
        "metrics": experiment2_summary(
            records=records,
            items=items,
        ),
        "generated_files": {
            "raw_predictions": str(raw_predictions_path),
        },
    }
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "experiment2_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [nlu_eval] finished experiment2 run_id={run_id}",
        flush=True,
    )
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    args = parser.parse_args()
    run_config = RunConfig.model_validate(json.loads(args.run_config.read_text(encoding="utf-8")))
    results_dir = asyncio.run(run_experiment2(run_config=run_config))
    print(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
