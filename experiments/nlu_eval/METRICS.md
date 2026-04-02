# NLU Eval Metrics

Definitions for the NLU benchmark metrics.

## Shared Conventions

- `gt_outcome` is derived from `gt_medical_state` with
  `MedicalEmergency(**gt_medical_state).get_outcome()`.
- Medical slot-set metrics treat a slot as positive only when its value is
  `True`.
- For the positive-slot-set metrics, `False` and a missing field both behave as
  "not in the positive set".
- Experiment 1 collapses repeats to one decision per `system_id` and `item_id`
  before scoring outcome metrics.
- Experiment 1 outcome booleans are collapsed independently by strict majority;
  ties resolve to `False`.
- Experiment 2 scores most metrics directly on the raw prediction rows, so each
  repeat contributes one observation unless stated otherwise.

## Experiment 1

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| `outcome_subset_accuracy` | For each system and item, build the 3-label outcome vector `{rd1, rd2, cpr}`. Use explicit outcome flags from `raw_output` when present; otherwise derive the cumulative vector from `predicted_outcome`. Collapse each label across repeats by strict majority, then score whether the full 3-label vector matches the gold vector exactly. | Higher is better. This is the strict multilabel accuracy for the outcome decision vector. |
| `final_outcome_accuracy` | Collapse the same per-item 3-label prediction vector to a single final label by taking the highest positive outcome in `{cpr, rd2, rd1}` and compare it with `gt_outcome`. | Higher is better. This is the coarse single-label outcome accuracy after hierarchical collapse. |
| `outcome_hamming_loss` | On the same collapsed 3-label outcome vectors, run `hamming_loss` over `{rd1, rd2, cpr}`. | Lower is better. This is the fraction of wrong outcome-label decisions after repeat collapse. |
| `outcome_jaccard` | On the same collapsed 3-label outcome vectors, run `jaccard_score(..., average="samples", zero_division=1.0)`. | Higher is better. This gives partial credit for overlap between predicted and gold positive outcome labels. |
| `followup_rate` | Fraction of raw prediction rows for that system where `response_kind == "followup"`. | Lower is better. A high value means the system often refuses to commit to a dispatch outcome in one turn. |
| `parse_failure_rate` | Fraction of raw prediction rows for that system where `response_kind == "parse_error"`. | Lower is better. This measures output-format/runtime failures rather than clinical reasoning quality. |

## Experiment 2

### Summary Metrics

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| `medical_slot_accuracy` | Construct a binary indicator matrix over `MEDICAL_FIELDS` using `value is True` for both gold and predicted medical state. Flatten the matrix and run `accuracy_score` over all per-field decisions. | Higher is better, but it can look optimistic when most medical fields are absent because shared negatives count as correct. |
| `medical_hamming_loss` | On the same medical-only indicator matrix, run `hamming_loss`. This is the fraction of medical field decisions that are wrong. | Lower is better. Each missed positive or spurious positive medical slot contributes equally. This metric excludes non-medical fields. |
| `medical_subset_accuracy` | On the same medical-only indicator matrix, run `accuracy_score` on whole rows instead of flattened labels. A row is correct only if every medical field matches. | Higher is better. This is a strict exact-match metric for the full medical state. |
| `medical_slot_jaccard` | Convert each row to the set of medical fields where `value is True` and compute `jaccard_score(..., average="samples", zero_division=1.0)` on the medical indicator matrix. | Higher is better. This measures overlap between the predicted and gold positive medical slot sets while ignoring shared negatives. |
| `non_medical_exact_match_accuracy` | Restrict to items with a non-empty `gt_non_medical_state`. For each raw prediction row, mark whether `predicted_non_medical_state == gt_non_medical_state`. Average within each item across repeats, then average equally across eligible items. | Higher is better. `None` means the subset being summarized had no items with gold non-medical fields. |
| `macro_precision` | Run `precision_recall_fscore_support` on the medical indicator matrix, but only for medical fields with at least one positive gold example. Take the macro average across those fields. | Higher is better. This answers: when the system predicts a medical slot, how often is that positive claim correct? |
| `macro_recall` | Same setup as macro precision, using recall. | Higher is better. This answers: how often does the system recover the positive medical slots that are actually present? |
| `macro_f1` | Harmonic mean of per-field precision and recall, macro-averaged over medical fields with positive support. | Higher is better. This is the main positive-slot retrieval balance metric. |
| `followup_rate` | Fraction of raw prediction rows where `response_kind == "followup"`. | Lower is better. A high value means the agent often asks for more information instead of returning a state. |
| `parse_failure_rate` | Fraction of raw prediction rows where `response_kind == "parse_error"`. | Lower is better. This captures schema/runtime failures, not slot quality directly. |
| `repeat_instability_rate` | For each item, build a repeat signature from `response_kind`, `predicted_medical_state`, `predicted_non_medical_state`, `predicted_outcome`, and `followup_text`. Report the fraction of items whose repeats contain more than one unique signature. | Lower is better. High instability means the same input yields materially different outputs across repeats. |

### Stratified Metrics

| Metric Group | Calculation | Interpretation |
| --- | --- | --- |
| `per_slot_count` | Recompute the full Experiment 2 summary metric set on item subsets grouped by `gt_positive_slot_count`, the number of gold medical fields set to `True` for the item. Each bucket also includes `item_count`. | Use this to see whether performance degrades as more positive medical findings are present. |
| `per_slot` | For each medical field with at least one positive gold example, report `support` and one-vs-rest `precision`, `recall`, and `f1` from `precision_recall_fscore_support(..., average=None)`. | Use this to see which specific medical findings are easy to recover, often hallucinated, or often missed. Fields with zero positive support are omitted. |

### Error Analysis

The error analysis is item-level and uses the first available prediction row for
each item. It compares only the positive medical slot sets.

| Metric | Calculation | Interpretation |
| --- | --- | --- |
| `error_analysis.item_pattern_counts.exact_match_items` | Count items where the predicted positive medical slot set exactly equals the gold positive medical slot set. | Higher is better. |
| `error_analysis.item_pattern_counts.extra_only_items` | Count items where the prediction adds positive medical slots that are not in gold, but does not miss any gold positives. | High values indicate over-calling without omissions. |
| `error_analysis.item_pattern_counts.missing_only_items` | Count items where the prediction misses one or more gold positive medical slots, but adds no extra positives. | High values indicate under-calling. |
| `error_analysis.item_pattern_counts.mixed_error_items` | Count items with both missed gold positives and extra predicted positives. | High values indicate substitutions or generally noisy slot selection. |
| `error_analysis.false_positive_fields` | Across items, count how often each medical field appears in the predicted positive set but not in the gold positive set. | High counts identify hallucinated positive medical findings. |
| `error_analysis.false_negative_fields` | Across items, count how often each medical field is present in the gold positive set but absent from the predicted positive set. | High counts identify commonly missed positive medical findings. |
| `error_analysis.confusion_pairs` | For each item with both misses and extras, count every `(missing_field, extra_field)` pair from the Cartesian product of missing and extra positive medical fields. | High counts suggest systematic substitutions, such as predicting one clinical concept in place of another. |
