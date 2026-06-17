# Argilla Dialogue Review UI

This toolchain creates an Argilla dataset where each record maps to one JSONL row, while the UI still shows the full dialogue for that row's `dialogue_id` group.

It supports two review targets on the same record:

1. `state`
   The reviewer edits typed Argilla questions derived from the `EmergencyCall` model in `src/ems_prepared/dialogue_state/emergency_call_state.py`. Boolean fields use one flat native multi-label selector where selected fields mean `true` and unselected fields mean missing / not explicitly true.
2. `dialog_acts`
   The reviewer edits one native Argilla `SpanQuestion` attached to the current utterance text. Export validates the resulting list against `DialogueActModel` from `experiments/dialog_act_labelling/dialogue_act/models.py`.

## Install

```bash
uv sync --group argilla
```

## Environment

Set the standard Argilla connection variables, or pass them as CLI flags:

```bash
export ARGILLA_API_URL=http://localhost:6900
export ARGILLA_API_KEY=...
export ARGILLA_WORKSPACE=argilla
```

## Upload The Review Dataset

```bash
uv run python -m experiments.dialogue_labelling.argilla.sync upload \
  --input dialog_acts.jsonl \
  --dataset-name dialogue-labelling \
  --replace
```

Notes:

- `--replace` deletes and recreates the dataset. This is the safest mode because Argilla dataset settings are immutable and stale records are otherwise easy to keep around by accident.
- The current `state` values seed the structured state questions.
- The UI shows the full dialogue for the matching `dialogue_id` and marks the current record inline.
- The dataset includes a required `Review status` control because the Argilla server will not publish datasets that have no required questions.
- `dialog_acts` uses one native span question on the `Current utterance` field.
- Each span answer stores `label`, `start`, and `end` offsets into the utterance text.
- Dialogue acts stay separate from the belief state. The exporter never reads `inform.value` from the reviewed `state`.
- The source row is shown once as `Record JSON` at the bottom of the field pane, together with validation details for `state` and `dialog_acts`.

## Export Reviewed JSONL

```bash
uv run python -m experiments.dialogue_labelling.argilla.sync export \
  --input dialog_acts.jsonl \
  --dataset-name dialogue-labelling \
  --output dialog_acts.reviewed.jsonl
```

Useful options:

- `--reviewer-user-id <uuid>` when multiple annotators answered the same records.
- `--skip-unanswered` to export only rows that already have both review answers.
- `--in-place` to overwrite the source JSONL.

## Output Rules

- `state` is assembled from the structured state questions and validated with `EmergencyCall.model_validate(...)`.
- `dialog_acts` is assembled from the span-question answers and validated as a JSON array of `DialogueActModel`.
- `dialog_acts` is always written back as a JSON array.
- If a reviewer leaves seeded state questions untouched, export falls back to the stored Argilla suggestions for those fields.
- If a reviewer leaves `dialog_acts` untouched, export falls back to the stored Argilla suggestions for those questions.
- `state` preserves the original column shape:
  - if the source row stored `state` as a JSON string, the export writes a JSON string;
  - if the source row stored `state` as an object, the export writes an object.

## Dialogue Act Mapping

- `greeting`
  - highlight the greeting phrase
  - exports `{"act": "greeting"}`
- `question:caller_name`
  - highlight the part of the utterance asking for the caller name
  - exports `{"act": "question", "slot": "caller_name"}`
- `inform:caller_name`
  - highlight the actual caller name span, for example `Max`
  - exports `{"act": "inform", "slot": "caller_name", "value": "Max"}`
- `inform:persons_affected`
  - highlight the numeric evidence span, for example `2`
  - exports `{"act": "inform", "slot": "persons_affected", "value": 2}`
- `inform:heavy_injury:true`
  - highlight the evidence span for the boolean claim
  - exports `{"act": "inform", "slot": "heavy_injury", "value": true}`
- `inform:patient_gender:male`
  - highlight the evidence span for the enum claim
  - exports `{"act": "inform", "slot": "patient_gender", "value": "male"}`
- `confirm:false`
  - highlight the supporting phrase such as `no`
  - exports `{"act": "confirm", "value": false}`
- `instruct:stay_on_line`
  - highlight the instruction phrase
  - exports `{"act": "instruct", "action": "stay_on_line"}`

For string and integer `inform` labels, the exported value comes only from the highlighted span text, not from the belief state.
