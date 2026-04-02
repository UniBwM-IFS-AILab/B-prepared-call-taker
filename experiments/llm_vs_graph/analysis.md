# LLM vs Graph Evaluation Summary

## Scope

This report reflects the current descriptive-only evaluation notebook:
[experiment_evaluation.py](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/experiment_evaluation.py).

Data source:

- logs: `experiments/llm_vs_graph/logs`
- ground truth: `experiments/llm_vs_graph/scenario_ground_truth`
- exports:
  - [metrics_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/metrics_summary.csv)
  - [dialogue_turns_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/dialogue_turns_summary.csv)
  - [likert_summary.csv](/home/seapat/Desktop/ems-prepared/experiments/llm_vs_graph/likert_summary.csv)

All results below are descriptive. The notebook does not currently report confidence intervals or p-values.

## Coverage

- Total sample directories discovered: 79
- Dialogues scored for state metrics: 63
- Surveys loaded: 59
- Sessions skipped for state scoring because `final_state.json` was missing: 16
- Policies in scored dialogues: `agent=33`, `graph=30`
- Scenarios represented in the scored set: 17 of 17
- Non-empty free-text survey responses: 17

## Executive Summary

`graph` is descriptively stronger than `agent` on the task and state metrics and also requires fewer caller turns on average. The clearest gaps are:

- higher `outcome_match`: `0.500` vs `0.394`
- higher `slot_accuracy_predicted_outcome`: `0.484` vs `0.368`
- better `Yes`-class recovery:
  - precision: `0.317` vs `0.220`
  - recall: `0.253` vs `0.152`
  - F1: `0.269` vs `0.165`
- fewer caller turns on average: `4.33` vs `5.58`

At the same time, both policies are very high on `slot_accuracy_any_valid` (`0.951` vs `0.941`). That means both usually end close to some valid scenario interpretation, but `graph` more often ends in a state that is consistent with the exact outcome it chose.

Survey scores are close and broadly positive for both policies. They do not show the same separation as the state-based metrics.

## Primary State Metrics

The table below reports descriptive policy means and sample standard deviations.

| Metric | agent | graph |
| --- | ---: | ---: |
| Outcome accuracy | 0.394 (SD 0.496) | 0.500 (SD 0.509) |
| Outcome-consistent slot accuracy | 0.368 (SD 0.465) | 0.484 (SD 0.492) |
| Slot accuracy, any valid GT | 0.941 (SD 0.037) | 0.951 (SD 0.050) |
| Exact final-state match, predicted outcome GT | 0.000 (SD 0.000) | 0.067 (SD 0.254) |
| Macro F1, predicted outcome GT | 0.182 (SD 0.234) | 0.253 (SD 0.271) |

## Positive-Slot Recovery

Because positive findings are sparse and operationally important, the `Yes`-class PRF metrics are the most informative classwise diagnostics.

| Metric | agent | graph |
| --- | ---: | ---: |
| Precision Yes | 0.220 (SD 0.368) | 0.317 (SD 0.427) |
| Recall Yes | 0.152 (SD 0.222) | 0.253 (SD 0.364) |
| F1 Yes | 0.165 (SD 0.249) | 0.269 (SD 0.368) |

Interpretation:

- `graph` is better at recovering positive state on all three descriptive summaries.
- `No`-class PRF is `0` for both policies in this dataset, so it is not informative here.
- `Unknown`-class and macro metrics remain useful context, but they should not replace the `Yes`-focused view.

## Dialogue Efficiency

The dialogue-turn summary is reported separately because efficiency is easier to interpret with both moment-based and robust summaries.

| Policy | Mean turns | SD | Median | Q1 | Q3 | IQR | Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| agent | 5.576 | 2.693 | 4 | 4 | 8 | 4 | 33 |
| graph | 4.333 | 3.089 | 4 | 3 | 4 | 1 | 30 |

Interpretation:

- `graph` is shorter on average.
- Both policies have the same median caller turn count of `4`.
- The spread is very different:
  - `agent`: `4 [4, 8]`
  - `graph`: `4 [3, 4]`
- This suggests the baseline more often produces longer calls, while `graph` is more consistent.

## Survey Results

Survey scores are descriptive means and sample standard deviations on the Likert items.

| Q | Label | agent | graph |
| --- | --- | ---: | ---: |
| 1 | Scenario Understanding | 4.333 (SD 0.957) | 4.269 (SD 1.079) |
| 2 | Agents' Understanding | 4.061 (SD 1.029) | 4.231 (SD 1.177) |
| 3 | Relevance of messages | 4.121 (SD 1.023) | 4.269 (SD 1.002) |
| 4 | Response Time | 4.242 (SD 1.032) | 4.115 (SD 1.366) |
| 5 | Agent adjustment to context | 4.212 (SD 0.960) | 4.077 (SD 1.164) |
| 6 | Human-like responses | 4.152 (SD 0.906) | 4.154 (SD 1.047) |
| 7 | Overall Satisfaction | 4.061 (SD 1.059) | 4.077 (SD 1.129) |

Survey interpretation:

- Ratings are broadly positive for both policies.
- The survey does not separate the policies as clearly as the state metrics do.
- `agent` is descriptively higher on Q1, Q4, and Q5.
- `graph` is descriptively higher on Q2, Q3, and Q7.
- Q6 is essentially tied.

## Practical Takeaway

For this evaluation set, `graph` is the stronger policy on the operational metrics that matter most:

- it reaches the correct final outcome more often
- it ends in an outcome-consistent final state more often
- it recovers positive slot state more effectively
- it uses fewer caller turns on average and has a tighter middle spread

The user survey is more neutral. That suggests the main advantage of `graph` is task behavior and consistency rather than a large perceived interaction-quality gain.
