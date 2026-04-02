# Synthetic Data Plan

## Goal
Produce additional benchmark items so that all medical slots are represented as evenly as possible, while keeping utterances realistic, single-turn, and layperson phrased.

## Step 1: Freeze The Target Schema
1. Read the current `MedicalEmergency` and `EmergencyCall` schemas.
2. Freeze the list of medical slots and non-medical fields that the benchmark should cover.
3. Mark any slots that should not appear in synthetic generation because they are deprecated or not benchmark-relevant.

## Step 2: Define Coverage Targets
1. Set a target number of single-slot items per medical slot.
2. Set a target number of multi-slot items per medical slot for clinically plausible combinations.
3. Set a target number of non-medical plus medical mixed items.
4. Set a target number of explicit negation items if negation should remain in the dataset.

## Step 3: Create Canonical Seed Examples
1. Write one canonical layperson utterance for each medical slot.
2. Write canonical multi-slot utterances for clinically plausible combinations.
3. Write canonical mixed utterances that include volunteered name or location.
4. Keep each canonical utterance short, realistic, and single-turn solvable.

## Step 4: Build A Balanced Generation Table
1. Create a table with one row per desired synthetic item.
2. Each row should specify:
   - target medical slots
   - target non-medical fields
   - whether the item is single-slot or multi-slot
   - a seed utterance or scenario note
3. Ensure each medical slot appears the planned number of times.
4. Ensure no slot gets overrepresented just because it is easier to paraphrase.

## Step 5: Generate Layperson Paraphrases
1. For each row, generate several paraphrases from the canonical seed.
2. Constrain generation to:
   - no medical jargon unless realistic for a caller
   - no dispatcher-style phrasing
   - one-turn solvable answers
   - no extra unsupported facts
3. Prefer several shallow paraphrases over one highly creative paraphrase.

## Step 6: Validate Against The Intended Slot Set
1. Run each generated utterance through a validation pass.
2. Reject any utterance that clearly implies slots outside the intended target set.
3. Reject any utterance that is too ambiguous to support the intended slots.
4. Reject any utterance that sounds medically trained instead of layperson.

## Step 7: De-Duplicate And De-Correlate
1. Remove near-duplicate utterances.
2. Remove repeated lexical templates that would make the benchmark too easy.
3. Ensure the same slot is expressed with varied wording across items.

## Step 8: Review High-Risk Slot Families Manually
1. Manually review overlapping or commonly confused families, for example:
   - unconsciousness / unresponsiveness
   - breathing distress / apnea / agonal breathing
   - injury severity labels
   - childbirth labels
   - metabolic / temperature labels
2. Tighten wording until the intended gold slots are the only defensible interpretation.

## Step 9: Materialize The Dataset
1. Write the accepted rows into JSONL with:
   - `id`
   - `operator_question`
   - `caller_utterance`
   - `gt_medical_state`
   - `gt_non_medical_state`
2. Keep metadata derived fields out of the dataset file.

## Step 10: Run A Pilot Audit
1. Sample items per slot family.
2. Check that slot support is balanced.
3. Check that multi-slot items are still realistic.
4. Check that non-medical mixed items remain a minority.

## Step 11: Run The Benchmark And Inspect Confusions
1. Run the benchmark on the new synthetic set.
2. Inspect:
   - exact match
   - macro F1
   - false positive fields
   - false negative fields
   - confusion pairs
3. Use the confusion report to refine only the problematic slot families.

## Step 12: Freeze The Main Dataset
1. Once coverage and wording are acceptable, freeze the dataset.
2. Do not keep regenerating after seeing benchmark results from the final run.

## Balancing Sketch

The balancing problem should be treated as quota filling, not free-form prompting.

### Core Structures

Use one slot table that tracks the target support and current support for each medical slot.

```python
@dataclass
class SlotSpec:
    name: str
    target_support: int
    current_support: int = 0
    group: str | None = None
    allow_single: bool = True
```

Example:

```python
SlotSpec("acute_speech_disorder", target_support=12, group="neuro")
SlotSpec("apnea", target_support=12, group="breathing")
```

Use one generation request structure that defines what the next synthetic item should contain.

```python
@dataclass
class GenerationRequest:
    medical_slots: tuple[str, ...]
    non_medical_fields: tuple[str, ...]
    operator_question: str
    style: str
    notes: str = ""
```

Use one candidate structure before validation:

```python
@dataclass
class CandidateItem:
    operator_question: str
    caller_utterance: str
    gt_medical_state: dict[str, bool]
    gt_non_medical_state: dict[str, str]
```

Use one accepted-item structure for the final benchmark rows:

```python
@dataclass
class AcceptedItem:
    id: str
    operator_question: str
    caller_utterance: str
    gt_medical_state: dict[str, bool]
    gt_non_medical_state: dict[str, str]
```

### Support Accounting

Recompute support from the accepted items:

```python
def recompute_support(slot_specs, accepted_items):
    for spec in slot_specs.values():
        spec.current_support = 0
    for item in accepted_items:
        for slot, value in item.gt_medical_state.items():
            if value is True:
                slot_specs[slot].current_support += 1
```

Compute deficits:

```python
def deficits(slot_specs):
    return {
        name: spec.target_support - spec.current_support
        for name, spec in slot_specs.items()
    }
```

Choose the most underfilled slot:

```python
def pick_anchor_slot(slot_specs):
    remaining = [
        spec for spec in slot_specs.values()
        if spec.current_support < spec.target_support
    ]
    return max(remaining, key=lambda s: s.target_support - s.current_support)
```

### Generation Loop

```python
while any(spec.current_support < spec.target_support for spec in slot_specs.values()):
    anchor = pick_anchor_slot(slot_specs)
    request = build_generation_request(anchor, slot_specs, accepted_items)
    candidates = generate_candidates(request)
    accepted = False

    for candidate in candidates:
        if not validate_candidate(candidate, request):
            continue
        accepted_items.append(materialize(candidate))
        recompute_support(slot_specs, accepted_items)
        accepted = True
        break

    if not accepted:
        mark_request_as_blocked(request)
```

### Generation Policy

The benchmark policy should decide whether the next item is:
- a single-slot item for the most underrepresented slot
- a multi-slot item for a clinically plausible combination that includes that slot
- a non-medical plus medical mixed item

Example sketch:

```python
def build_generation_request(anchor, slot_specs, accepted_items):
    if should_make_single_slot(anchor):
        medical_slots = (anchor.name,)
    else:
        partner = pick_plausible_partner(anchor, slot_specs)
        medical_slots = (anchor.name, partner.name)

    return GenerationRequest(
        medical_slots=medical_slots,
        non_medical_fields=pick_non_medical_fields(medical_slots),
        operator_question=pick_operator_question(medical_slots),
        style="layperson single-turn"
    )
```

### Validation

Validation should reject candidates that do not match the intended gold slots exactly.

```python
def validate_candidate(candidate, request):
    if set(true_slots(candidate.gt_medical_state)) != set(request.medical_slots):
        return False
    if set(candidate.gt_non_medical_state) != set(request.non_medical_fields):
        return False
    if sounds_too_medical(candidate.caller_utterance):
        return False
    if not one_turn_solvable(candidate):
        return False
    if is_near_duplicate(candidate):
        return False
    return True
```

### Extra Policy Tables

Keep a table of clinically plausible multi-slot pairings.

```python
allowed_pairs = {
    ("acute_speech_disorder", "acute_paralysis"),
    ("severe_dyspnoea", "cyanosis"),
    ("frequent_pregnancy_contraction", "ongoing_delivery"),
}
```

Keep a target distribution for operator questions so the third intro question remains dominant.

```python
question_budget = {
    "What has just happened acutely?": 0.9,
    "With whom am I speaking, please?": 0.05,
    "Where exactly is the emergency location?": 0.05,
}
```

Keep simple slot-family tags for reporting and targeted balancing:

```python
group = "breathing" | "neuro" | "pregnancy" | "injury" | ...
```

### Why This Needs Custom Logic

The balancing rule is benchmark-specific:
- all slots should matter equally
- support should be evenly distributed
- only clinically plausible multi-slot combinations should be generated
- utterances must remain layperson-like and single-turn solvable

A library can help with structured generation or orchestration, but it will not infer these benchmark policies automatically.
