# Experiments

- `knowledge_extraction`
  Converts the `Notrufabfrage_*.pdf` source material into a bilingual retrieval corpus. It is the first step in the data-production chain and exists so later experiments do not need to re-read raw PDFs.
- `synthetic_single_turn`
  Generates synthetic single-turn emergency-call benchmark rows from the retrieval corpus. It produces `accepted_dataset.jsonl`, which is then consumed by the NLU evaluation experiment.
- `nlu_eval`
  Evaluates benchmark datasets, including datasets produced by `synthetic_single_turn`, against the slot-filling systems and their baselines. It consumes dataset artifacts, not raw PDFs or extracted knowledge directly.
- `llm_vs_graph`
  Compares different policy/runtime strategies on scenario-based conversations. It is an analysis experiment and does not sit in the main data-production chain.
- `debug`
  Scratch area for debugging artifacts and logs. It is not a data-production or evaluation experiment.

## Dependency Chain

The main dependency chain is:

1. `~/Spaces/B-prepared/Useful materials/Leitstelle_Material/Notrufabfrage_*.pdf`
2. `experiments/knowledge_extraction`
3. `knowledge_corpus.jsonl`
4. `experiments/synthetic_single_turn`
5. `accepted_dataset.jsonl`
6. `experiments/nlu_eval`

In other words:

- raw PDFs are the source material
- `knowledge_extraction` turns them into a bilingual `knowledge_corpus.jsonl`
- `synthetic_single_turn` uses that corpus to generate benchmark rows
- `nlu_eval` evaluates those benchmark rows
