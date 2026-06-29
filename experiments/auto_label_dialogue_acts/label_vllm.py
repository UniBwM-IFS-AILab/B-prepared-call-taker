import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import Field, RootModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.schema_variants import (
    field_attrs_from_original,
    inline_json_schema_refs,
    slot_name_type_from_model,
    slot_value_schema,
)
from experiments.dialog_act_labelling.comm_func_schema import (
    COMMUNICATIVE_FUNCTION_TYPE_BY_VALUE,
    SlotOnlyFunction,
    SlotValueFunction,
)
from experiments.dialog_act_labelling.enums import CommunicativeFunction
from experiments.dialog_act_labelling.schemas import (
    DialogueLabellingInput,
    DialogueLabellingOutput,
    DialogueTurn,
)

from .resume_checkpoint import load_resume_checkpoint, record_identity
from .system_prompts import (
    FUNCTION_STAGE_SYSTEM_PROMPT,
    SINGLE_STAGE_SYSTEM_PROMPT,
    SLOT_FALLBACK_CHOICE,
    SLOT_STAGE_SYSTEM_PROMPT,
    SLOT_STAGE_USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    VALUE_STAGE_SYSTEM_PROMPT,
    VALUE_STAGE_USER_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger("dialog_act_labelling")

type LabellingMode = Literal["single-stage", "multi-stage"]


class CommunicativeFunctionList(
    RootModel[
        list[
            Union[
                tuple(
                    Annotated[
                        function_type.model_fields["communicative_function"].annotation,
                        Field(
                            **field_attrs_from_original(
                                function_type.model_fields["communicative_function"]
                            )
                        ),
                    ]
                    for function_type in COMMUNICATIVE_FUNCTION_TYPE_BY_VALUE.values()
                    if function_type.annotation_relevant
                )
            ]
        ]  # pyrefly: ignore [invalid-annotation]
    ]
):
    """Top-level structured-output schema for the function stage.

    The function stage predicts only communicative-function labels, not full acts.
    """


class SlotChoice(
    RootModel[slot_name_type_from_model(EmergencyCall) | Literal[SLOT_FALLBACK_CHOICE]]
):
    """Top-level structured-output schema for slot selection.

    The slot stage returns one EmergencyCall slot name directly, or the
    SLOT_FALLBACK_CHOICE sentinel when no slot applies.
    """


@dataclass(frozen=True)
class RuntimeConfig:
    llm: LLM
    tokenizer: Any
    max_output_tokens: int
    confidence_threshold: float | None


def _build_chat_prompt(tokenizer, *, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _input_json(
    labelling_input: DialogueLabellingInput, *, pretty: bool = False
) -> str:
    payload = labelling_input.model_dump(
        mode="json",
        exclude_none=True,
        exclude_unset=True,
        exclude_defaults=True,
    )
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _format_response_for_log(response_text: str) -> str:
    try:
        parsed_output = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text
    return json.dumps(parsed_output, ensure_ascii=False, indent=2)


def _average_token_probability(completion_output: Any) -> float | None:
    cumulative_logprob = completion_output.cumulative_logprob
    if cumulative_logprob is None:
        return None
    token_count = len(completion_output.token_ids)
    if token_count <= 0:
        return None
    return math.exp(cumulative_logprob / token_count)


def _generate_stage_response(
    runtime: RuntimeConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    structured_outputs: StructuredOutputsParams,
    stage_name: str,
    log_context: str | None = None,
) -> str | None:
    sampling_params = SamplingParams(
        temperature=0.0,
        structured_outputs=structured_outputs,
        max_tokens=runtime.max_output_tokens,
        logprobs=1 if runtime.confidence_threshold is not None else None,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        n=1,
    )
    prompt = _build_chat_prompt(
        runtime.tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    request_outputs = runtime.llm.generate(prompt, sampling_params=sampling_params)

    if len(request_outputs) != 1:
        raise RuntimeError("Expected a single request output from llm.generate().")
    [request_output] = request_outputs
    if len(request_output.outputs) != 1:
        raise RuntimeError(
            "Expected a single completion. Set SamplingParams(n=1) or adjust usage."
        )
    [completion] = request_output.outputs

    response_text = completion.text.strip()
    if completion.finish_reason == "length":
        raise RuntimeError(
            "Generation hit max_tokens before completion (finish_reason=length). "
            "Increase --max-output-tokens or reduce prompt/context size."
        )

    confidence = _average_token_probability(completion)
    LOGGER.warning(
        "confidence=%s finish_reason=%s",
        confidence,
        completion.finish_reason,
    )
    if (
        runtime.confidence_threshold is not None
        and confidence is not None
        and confidence < runtime.confidence_threshold
    ):
        suffix = f" {log_context}" if log_context else ""
        LOGGER.info(
            "%s_abstained: confidence=%.4f threshold=%.4f%s",
            stage_name,
            confidence,
            runtime.confidence_threshold,
            suffix,
        )
        return None
    return response_text


def _predict_slot_choice(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
    communicative_function: CommunicativeFunction,
) -> str | None:
    user_prompt = SLOT_STAGE_USER_PROMPT_TEMPLATE.format(
        communicative_function=communicative_function.value,
        input_json=_input_json(labelling_input),
    )
    if (
        response_text := _generate_stage_response(
            runtime,
            system_prompt=SLOT_STAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            structured_outputs=StructuredOutputsParams(
                json=SlotChoice.model_json_schema()
            ),
            stage_name="slot_stage",
            log_context=f"communicative_function={communicative_function.value}",
        )
    ) is None:
        return None

    choice = SlotChoice.model_validate_json(response_text).root
    return None if choice == SLOT_FALLBACK_CHOICE else choice


def _predict_value_for_slot(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
    communicative_function: CommunicativeFunction,
    slot: str,
) -> str | bool | int | None:
    user_prompt = VALUE_STAGE_USER_PROMPT_TEMPLATE.format(
        communicative_function=communicative_function.value,
        slot=slot,
        input_json=_input_json(labelling_input),
    )
    if (
        response_text := _generate_stage_response(
            runtime,
            system_prompt=VALUE_STAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            structured_outputs=StructuredOutputsParams(json=slot_value_schema(slot)),
            stage_name="value_stage",
            log_context=(
                f"communicative_function={communicative_function.value} slot={slot}"
            ),
        )
    ) is None:
        return None

    value = json.loads(response_text).get("value")
    if isinstance(value, str):
        return value.strip() or None
    return value


def _predict_functions(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
) -> list[CommunicativeFunction]:
    user_prompt = USER_PROMPT_TEMPLATE.format(input_json=_input_json(labelling_input))
    if (
        response_text := _generate_stage_response(
            runtime,
            system_prompt=FUNCTION_STAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            structured_outputs=StructuredOutputsParams(
                json=CommunicativeFunctionList.model_json_schema()
            ),
            stage_name="function_stage",
        )
    ) is None:
        return []

    deduplicated_functions: list[CommunicativeFunction] = []
    seen: set[CommunicativeFunction] = set()
    for communicative_function in CommunicativeFunctionList.model_validate_json(
        response_text
    ).root:
        if communicative_function in seen:
            continue
        seen.add(communicative_function)
        deduplicated_functions.append(communicative_function)
    return deduplicated_functions


def _assemble_act_for_function(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
    communicative_function: CommunicativeFunction,
) -> dict[str, object]:
    function_type = COMMUNICATIVE_FUNCTION_TYPE_BY_VALUE[communicative_function]

    slot = (
        _predict_slot_choice(
            runtime,
            labelling_input=labelling_input,
            communicative_function=communicative_function,
        )
        if issubclass(function_type, SlotOnlyFunction)
        else None
    )

    value = (
        _predict_value_for_slot(
            runtime,
            labelling_input=labelling_input,
            communicative_function=communicative_function,
            slot=slot,
        )
        if issubclass(function_type, SlotValueFunction) and slot is not None
        else None
    )

    return {
        "communicative_function": communicative_function.value,
        **({"slot": slot} if issubclass(function_type, SlotOnlyFunction) else {}),
        **({"value": value} if issubclass(function_type, SlotValueFunction) else {}),
    }


def _label_turn_multi_stage(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
) -> DialogueLabellingOutput:
    communicative_functions = _predict_functions(
        runtime, labelling_input=labelling_input
    )
    output = DialogueLabellingOutput.model_validate(
        {
            "acts": [
                _assemble_act_for_function(
                    runtime,
                    labelling_input=labelling_input,
                    communicative_function=communicative_function,
                )
                for communicative_function in communicative_functions
            ]
        }
    )
    LOGGER.info(
        "multi_stage_labelled_record:\n%s",
        json.dumps(
            {
                "input_data": labelling_input.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                ),
                "labelled_dialog_acts": output.model_dump(
                    mode="json",
                    exclude_none=True,
                )["acts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return output


def _label_turn_single_stage(
    runtime: RuntimeConfig,
    *,
    labelling_input: DialogueLabellingInput,
) -> DialogueLabellingOutput:
    user_prompt = USER_PROMPT_TEMPLATE.format(input_json=_input_json(labelling_input))
    LOGGER.info(
        "model_input:\n%s",
        USER_PROMPT_TEMPLATE.format(
            input_json=_input_json(labelling_input, pretty=True)
        ),
    )
    if (
        response_text := _generate_stage_response(
            runtime,
            system_prompt=SINGLE_STAGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            structured_outputs=StructuredOutputsParams(
                json=inline_json_schema_refs(
                    DialogueLabellingOutput.model_json_schema()
                )
            ),
            stage_name="single_stage",
        )
    ) is None:
        return DialogueLabellingOutput(acts=[])

    LOGGER.info("model_output:\n%s", _format_response_for_log(response_text))
    return DialogueLabellingOutput.model_validate_json(response_text)


def label_dataset(
    input_path: Path,
    output_path: Path,
    *,
    context_window: int = 8,
    max_output_tokens: int | None = None,
    labelling_mode: LabellingMode = "single-stage",
    confidence_threshold: float | None,
) -> None:
    llm = LLM(
        # model="Qwen/Qwen3.6-35B-A3B-FP8",
        model="Qwen/Qwen3.6-27B-FP8",
        max_model_len=8192,  # 16384,
        gpu_memory_utilization=0.85,
        cpu_offload_gb=8,  # try 0 first; if VRAM OOM, try 8, 12, 16
        dtype="auto",
        language_model_only=True,
        enforce_eager=True,
        max_num_seqs=1,  # good for offline labeling
        max_num_batched_tokens=1024,
    )
    runtime = RuntimeConfig(
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        max_output_tokens=4096 if max_output_tokens is None else max_output_tokens,
        confidence_threshold=confidence_threshold,
    )

    context: list[DialogueTurn] = []
    current_dialog_id: object | None = None
    has_seen_dialog = False
    last_observed_identity: tuple[object, object] | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = load_resume_checkpoint(output_path)
    checkpoint_pending = resume_checkpoint is not None

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("a", encoding="utf-8", buffering=1) as output_file,
    ):

        def remember_turn(turn: DialogueTurn) -> None:
            if context_window <= 0:
                return
            context.append(turn)
            if len(context) > context_window:
                del context[:-context_window]

        for line in input_file:
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
                LOGGER.info(
                    "duplicate_row_skipped: dialog_id=%s turn_index=%s",
                    dialog_id,
                    turn_index,
                )
                continue

            current_turn = DialogueTurn(
                speaker=record["speaker"],
                utterance=record["text"],
            )

            if checkpoint_pending:
                remember_turn(current_turn)
                last_observed_identity = row_identity
                if row_identity == resume_checkpoint:
                    LOGGER.info(
                        "resume_checkpoint_reached: dialog_id=%s turn_index=%s",
                        dialog_id,
                        turn_index,
                    )
                    checkpoint_pending = False
                continue

            context_slice = context if context_window > 0 else []
            labelling_input = DialogueLabellingInput(
                context=context_slice,
                speaker=record["speaker"],
                utterance=record["text"],
            )
            output = (
                _label_turn_single_stage(runtime, labelling_input=labelling_input)
                if labelling_mode == "single-stage"
                else _label_turn_multi_stage(runtime, labelling_input=labelling_input)
            )

            remember_turn(current_turn)
            last_observed_identity = row_identity
            output_file.write(
                json.dumps(
                    {
                        **record,
                        "dialog_acts": output.model_dump(
                            mode="json", exclude_none=True
                        )["acts"],
                    },
                    ensure_ascii=False,
                )
            )
            output_file.write("\n")
            output_file.flush()

    if checkpoint_pending:
        raise RuntimeError(
            "Output resume checkpoint was not found in the input dataset. "
            "The input ordering or contents changed, so checkpoint-based resume "
            "cannot safely continue."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label dialogue acts for a JSONL dataset using vLLM.",
        usage="%(prog)s [options] <input_path> <output_path>",
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help=("Maximum generated output tokens. Increase if generation is truncated."),
    )
    parser.add_argument(
        "--mode",
        dest="labelling_mode",
        choices=["single-stage", "multi-stage"],
        default="single-stage",
        help=(
            "Labelling pipeline mode. "
            "single-stage keeps the original one-pass schema call; "
            "multi-stage runs function->slot->value extraction."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help=(
            "Global abstention gate for all prediction stages "
            "(single-stage, function, slot, value). "
            "Uses average per-token confidence in [0,1]. "
            "Use a negative value to disable."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for dialog_act_labelling logger only.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)

    confidence_threshold = args.confidence_threshold
    if confidence_threshold < 0:
        confidence_threshold = None
    label_dataset(
        args.input_path,
        args.output_path,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
        labelling_mode=args.labelling_mode,
        confidence_threshold=confidence_threshold,
    )


if __name__ == "__main__":
    main()
