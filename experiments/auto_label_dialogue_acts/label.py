from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult

from experiments.dialog_act_labelling.schemas import (
    DialogueLabellingInput,
    DialogueLabellingOutput,
    DialogueTurn,
)

from .resume_checkpoint import load_resume_checkpoint, record_identity

AGENT_SPEC_PATH = Path(__file__).with_name("dialog_act_labeller.yaml")


def build_labelling_agent(
    *,
    model_name: str | None = None,
    base_url: str | None = None,
) -> Agent[DialogueLabellingOutput]:
    if base_url:
        os.environ["OLLAMA_BASE_URL"] = base_url
    overrides: dict[str, object] = {}
    if model_name:
        overrides["model"] = model_name
    return Agent.from_file(
        AGENT_SPEC_PATH,
        output_type=DialogueLabellingOutput,
        defer_model_check=True,
        **overrides,
    )


def label_dataset(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str | None = None,
    base_url: str | None = None,
    context_window: int = 8,
) -> None:
    agent = build_labelling_agent(model_name=model_name, base_url=base_url)
    context: list[DialogueTurn] = []
    current_dialog_id: object | None = None
    has_seen_dialog = False
    last_observed_identity: tuple[object, object] | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = load_resume_checkpoint(output_path)
    checkpoint_pending = resume_checkpoint is not None

    with (
        input_path.open("r", encoding="utf-8") as handle,
        output_path.open("a", encoding="utf-8") as output_handle,
    ):
        def remember_turn(turn: DialogueTurn) -> None:
            if context_window <= 0:
                return
            context.append(turn)
            if len(context) > context_window:
                del context[:-context_window]

        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            dialog_id, turn_index = record_identity(record)
            row_identity = (dialog_id, turn_index)

            if has_seen_dialog and dialog_id != current_dialog_id:
                context.clear()
            current_dialog_id = dialog_id
            has_seen_dialog = True

            if row_identity == last_observed_identity:
                continue

            current_turn = DialogueTurn(
                speaker=record["speaker"],
                utterance=record["text"],
            )

            if checkpoint_pending:
                remember_turn(current_turn)
                last_observed_identity = row_identity
                if row_identity == resume_checkpoint:
                    checkpoint_pending = False
                continue

            labelling_input = DialogueLabellingInput(
                context=context if context_window > 0 else [],
                speaker=record["speaker"],
                utterance=record["text"],
            )
            result: AgentRunResult[DialogueLabellingOutput] = agent.run_sync(
                labelling_input.model_dump_json()
            )
            output = result.output
            remember_turn(current_turn)
            last_observed_identity = row_identity
            labelled_record = record | output.model_dump(mode="json")
            output_handle.write(json.dumps(labelled_record))
            output_handle.write("\n")
            output_handle.flush()

    if checkpoint_pending:
        raise RuntimeError(
            "Output resume checkpoint was not found in the input dataset. "
            "The input ordering or contents changed, so checkpoint-based resume "
            "cannot safely continue."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label dialogue acts for a JSONL dataset."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
    )
    parser.add_argument("--context-window", type=int, default=8)
    args = parser.parse_args()
    label_dataset(
        args.input_path,
        args.output_path,
        model_name=args.model_name,
        base_url=args.base_url,
        context_window=args.context_window,
    )


if __name__ == "__main__":
    main()
