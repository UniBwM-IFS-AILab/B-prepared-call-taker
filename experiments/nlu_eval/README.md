# NLU Eval

Single-turn benchmark assets for evaluating:

- outcome accuracy of the full slot-based agent versus an outcome-only baseline
- exact medical slot-set recovery of the full agent
- opportunistic non_medical extraction alongside medical state

The benchmark assumes one shared underlying model/runtime. Experiment 1 compares
two system variants built on top of that same model:

- the full slot-based agent
- the outcome-only baseline

It is not intended for comparing different foundation models against each other.

## Layout

- `datasets/pilot_master.jsonl`: pilot dataset source of truth
- `datasets/main_master.jsonl`: main dataset source of truth
- `common/`: shared benchmark code used by both analyses
- `configs/`: JSON run config templates
- `experiment1.py`: Experiment 1 entrypoint
- `experiment2.py`: Experiment 2 entrypoint
- `METRICS.md`: metric definitions, calculation details, and interpretation notes
- `results/`: per-run artifacts. Each `results/<run_id>/` directory contains
  `manifest.json`, one `experiment*_raw_predictions.jsonl`, and one
  `experiment*_summary.json`.

## Dataset Contract

Each JSONL row is a plain JSON object. The checked-in benchmark uses `gt_*`
field names:

- `gt_medical_state`
- `gt_non_medical_state`

`gt_outcome` is derived at load time from `gt_medical_state` and is not stored in
the JSONL files.

## Run Config Format

Config files are JSON rather than YAML so the benchmark can use the repo's
existing standard-library JSON tooling without adding a new parser dependency.

The runner uses a single shared runtime configuration per run. Configs may set
`requests_per_minute` to proactively cap model calls for the run before any
provider-side `429` is observed. Experiment 1 always evaluates the fixed pair of
system variants:

- `full_agent`
- `outcome_baseline`

Experiment 2 always evaluates only:

- `full_agent`

## CLI Flow

1. Create or revise `*_master.jsonl`.
2. Run `experiments/nlu_eval/experiment1.py` or `experiments/nlu_eval/experiment2.py`.
The dataset is loaded directly by the experiment runner. There is no separate
materialization step and no derived experiment dataset copies.

Example:

```bash
python -m experiments.nlu_eval.experiment1 \
  --run-config experiments/nlu_eval/configs/pilot_experiment1.json
```

## Synthetic Pipeline

Synthetic dataset generation is a separate experiment under
`experiments/synthetic_single_turn/`, and that experiment now depends on the cached
`knowledge_corpus.jsonl` produced by `experiments/knowledge_extraction/`.

The benchmark datasets and shared evaluation code remain here under
`experiments/nlu_eval/`, but the generation pipeline, backend setup, example
configs, and `mise` tasks are documented in
`experiments/synthetic_single_turn/README.md`.

For the full dependency chain from source PDFs to evaluation datasets, see
`experiments/README.md`.
