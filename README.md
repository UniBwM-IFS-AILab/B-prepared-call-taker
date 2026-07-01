# ems-prepared

Code for *Preparing Citizens for Emergency Calls with a Hybrid FSM-LLM Dialogue Agent*.

This repository contains the emergency-call simulator, the graph-based hybrid policy, the LLM-only baseline, and the scenario description and ground truths used in the study.

![System Architecture](assets/system_arch.svg)

## Run

Use `uv` and set `GOOGLE_API_KEY` to run the agent:

| Purpose | Command | Notes |
| --- | --- | --- |
| Main web UI | `uv run python -m ems_prepared.adapters.gradio.app --help` | Recommended entrypoint. The scenario directory must be passed explicitly in this checkout. |
| Graph policy in the terminal | `uv run python -m ems_prepared.policies.pydantic_graph.emergency_main_graph` | Runs the hybrid FSM/graph-based policy directly. |
| LLM-only baseline in the terminal | `uv run python -m ems_prepared.policies.llm_only.agent` | Runs the baseline agent directly. |

## Repository Structure

```text
ems-prepared/
├── src/ems_prepared/
│   ├── adapters/                 # User-facing application entrypoints
│   ├── agents/                   # LLM prompting and task-specific agent logic
│   ├── dialogue_state/           # Structured emergency-call state definitions
│   ├── locale/                   # Static questions and instructions used by the agent
│   ├── policies/
│   │   ├── llm_only/             # LLM-only baseline
│   │   ├── pydantic_graph/       # Hybrid FSM/graph-based policy
│   │   └── shared.py             # Shared save/merge helpers
│   └── util/                     # Logging, settings, model helpers
├── assets/
│   ├── scenarios/                # Scenario description and ground truths
│   └── slack_webhook/            # Optional deployment/support 
├── notebooks/                    # Evaluation and exploratory analysis
├── logs/                         # Per-session outputs written at runtime
├── pyproject.toml
└── mise.toml
```

## Scenarios

- Scenario descriptions are stored in [assets/scenarios/](/home/seapat/Desktop/ems-prepared-main/assets/scenarios).
- [_Instructions.md](/home/seapat/Desktop/ems-prepared-main/assets/scenarios/_Instructions.md) contains shared caller instructions shown with every description
- [ScenarioGroundTruth/](/home/seapat/Desktop/ems-prepared-main/assets/scenarios/ScenarioGroundTruth) contains reference outcome/state annotations
- The evaluation of the user experiments is done in [notebooks/experiment_evaluation.py](/home/seapat/Desktop/ems-prepared-main/notebooks/experiment_evaluation.py).

## Session Outputs

Each run writes session artifacts under:

```text
logs/<experiment>/<user>/<session>/
```

Typical outputs include:

- `stdout.log`
- `messages.tsv`
- `state_changes.jsonl`
- `final_state.json`
- `message_history.json`
- `deps.json`
- graph persistence files and Mermaid exports for graph-based runs

## Citation

```bibtex
@inproceedings{,
  author = {},
  title = {},
  booktitle = {},
  year = {},
}
```
