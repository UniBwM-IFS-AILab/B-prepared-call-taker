# LLM vs Graph Reporting Template

## Recommended Reporting Style

Use a descriptive results table in the paper and keep the CSV exports as the reproducibility artifact.

Recommended emphasis:

- primary metrics:
  - `outcome_match`
  - `slot_accuracy_predicted_outcome`
  - `dialogue_turns`
- secondary diagnostics:
  - `slot_accuracy_any_valid`
  - `exact_match_predicted_outcome`
  - `precision_yes_predicted_outcome`
  - `recall_yes_predicted_outcome`
  - `f1_yes_predicted_outcome`
- descriptive only:
  - macro PRF
  - `Unknown` and `No` class metrics
  - Likert questions
  - free-text feedback

The current notebook is descriptive. The writeup should match that.

## Suggested Main Table

Report:

- point estimates for the binary and slot metrics
- mean and standard deviation for caller turns
- median with quartiles for caller turns

Suggested structure:

| Metric | agent | graph |
| --- | ---: | ---: |
| Outcome accuracy | 0.394 | 0.500 |
| Outcome-consistent slot accuracy | 0.368 | 0.484 |
| Mean caller turns (SD) | 5.58 (2.69) | 4.33 (3.09) |
| Median caller turns (Q1, Q3) | 4 (4, 8) | 4 (3, 4) |

Optional secondary table:

| Metric | agent | graph |
| --- | ---: | ---: |
| Slot accuracy, any valid GT | 0.941 | 0.951 |
| Exact final-state match, predicted outcome GT | 0.000 | 0.067 |
| Precision Yes | 0.220 | 0.317 |
| Recall Yes | 0.152 | 0.253 |
| F1 Yes | 0.165 | 0.269 |

## Suggested Results Text

### Evaluation Setup

We compared the `agent` and `graph` policies on the `experiments/llm_vs_graph/logs` evaluation set. Each dialogue was treated as one independent observation. We report descriptive summaries from the final dialogue state together with caller-turn summaries and survey means.

### Main Results

Suggested paragraph:

The graph-based policy is descriptively stronger than the baseline on the operational metrics. It achieves higher outcome accuracy (`0.50` vs `0.39`) and higher outcome-consistent slot accuracy (`0.48` vs `0.37`). It also requires fewer caller turns on average (`4.33` vs `5.58`). Although both policies have the same median caller turn count of `4`, the quartiles differ substantially: the graph-based policy yields `4 (3, 4)` turns, whereas the LLM-only baseline yields `4 (4, 8)`, indicating that the baseline more often requires longer calls.

### Positive-State Diagnostics

Suggested paragraph:

The `Yes`-class PRF metrics show the same pattern. The graph-based policy has higher precision (`0.317` vs `0.220`), recall (`0.253` vs `0.152`), and F1 (`0.269` vs `0.165`) for positive slot recovery. This suggests that it more reliably recovers the positive findings that support the final dispatch outcome.

### Survey Results

Suggested paragraph:

Survey results are broadly positive for both policies and show much smaller differences than the state-based metrics. This suggests that the main advantage of the graph-based policy is task behavior and consistency rather than a large perceived interaction-quality gain.

## Wording Guidance

Prefer descriptive wording such as:

- `numerically higher`
- `descriptively stronger`
- `shorter on average`
- `more consistent`
- `suggests`

Avoid stronger inferential wording such as:

- `significantly better`
- `statistically superior`
- `proved`

unless the notebook and paper are explicitly changed back to an inferential analysis.

## Mapping To Exported Files

- state metrics: [metrics_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/metrics_summary.csv)
- dialogue turns: [dialogue_turns_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/dialogue_turns_summary.csv)
- survey summaries: [likert_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/likert_summary.csv)
