# ems-prepared

## Usage

### Install

```bash
mise i
uv sync --group server
```

### Generic launcher (`main`)

List discovered plugins:

```bash
uv run main list
```

Run a frontend subcommand:

```bash
# CLI frontend
uv run main cli
uv run main cli --policy agent --locale english
uv run main cli --experiment exp_2026_03

# Gradio frontend
uv run main gradio --locale german --scenario-dir ./experiments/iva/scenarios

# FastAPI frontend
uv run main fastapi --experiment exp_2026_03 --host 127.0.0.1 --port 8000

# Dedicated reload-capable FastAPI entry point
uv run --group server fastapi-server --reload --experiment exp_2026_03
```

Get frontend-specific help:

```bash
uv run main cli --help
uv run main gradio --help
uv run main fastapi --help
uv run --group server fastapi-server --help
```

`--policy`, `--locale`, and `--experiment`/`--experiment-name` are shared launcher flags available on every frontend subcommand.

### Mise tasks

```bash
mise run cli
mise run llm_only
mise run gradio
mise run gradio_dev
mise run gradio_guided_llm_vs_graph
mise run server
mise run server_dev
mise run openai_server_vllm
mise run openai_server_sglang
mise run openai_server_vllm --model Qwen/Qwen3-14B --host 127.0.0.1 --port 8000
mise run openai_server_sglang --model Qwen/Qwen3-14B --host 127.0.0.1 --port 8000 --device cuda
mise run openai_server_sglang --model Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 --device cpu
mise run openai_server_ollama
mise run synthetic_single_turn_run
```

On NixOS, enter the default dev shell first so `CC`, `CUDA_HOME`,
`TRITON_LIBCUDA_PATH`, `LD_LIBRARY_PATH`, and helper tools are set up for the
model-server tasks:

```bash
direnv allow
# or: nix develop
```

## Structure

### Package structure

```text
src/ems_prepared/
├── main.py                # subcommand launcher + shared policy/locale/experiment flags
├── plugins.py             # entry-point discovery + validation
├── model/
│   ├── contracts.py       # core backend contracts/DTOs
│   ├── context.py         # Settings deps object for runtime
│   ├── session_service.py # shared backend orchestration
│   ├── session_backend_file.py
│   └── session_backend_sqlalchemy.py
├── adapters/
│   ├── fastapi/app.py     # FastAPIFrontend plugin class
│   ├── gradio/app.py      # GradioFrontend plugin class
│   └── cli/app.py         # CliFrontend plugin class
└── policies/
    ├── pydantic_graph/runtime.py  # policy factory
    ├── llm_only/runtime.py        # policy factory
    └── ...
```



## Motivation

Im Eckpunktepapier 20162 von 30 wissenschaftlichen Fachgesellschaften, Institutionen und Organisa-tionen zur notfallmedizinischen Versorgung der Bevölkerung in der Prähospitalphase und in der Klinik wird auch die Erste Hilfe durch Laien thematisiert: „Gezielte und regelmäßige Schulung der Bevölke-rung – insbesondere Schüler im Rahmen des regulären Unterrichts – sowie von Präventions- und Auf-klärungsprogrammen sollen medizinische Laien befähigen, Vitalstörungen frühzeitig zu erkennen, einen Notruf korrekt abzusetzen und danach selbst effektive und lebensrettende Maßnahmen durch-zuführen.“

(Taken from the "Empfehlung 001/01-2017 vom 10.07.2017 des Rettungsdienstausschuss Bayern")

## Known Bugs

- Determination of Emergency Type
  -
- Handling of Fire- and Non-Emergency Cases
- sometimes RD1 and RD2 Variables are triggered together because they are too similar
  - now_unresponsive & unconscious
- Completion of data after triggering disposition causes endless loop (disabled for now)
- When the agent could not deduce new state from a caller's response, it will ask for some variables (*fine*) in a weird manner (*not fine*)
  - examples
    - include the state key verbatim in the question (with underscores)
    - return a json string with value sit wants filled
  - We need to give better instructions
- the agent cannot self-loop, ie it can only ask once for more information
  - no message history is attached
  - RD1 takes precedence over repeated question loops
  - Solution:
    - attach history of all messages between first return of str, until a struct is returned,
    - Third return_type that indicates that the agent decided to not ask further but that there is no point in asking further (does this allow the agent to cheat?)

## Todos

- Decouple state filling from conversation
  - two separate agents, conversation agent decided how long they want to continue asking
- ditch `situation_description` and use message history instead
  - let agent decide how long he want to talk to determine Emrgency type while passing history along
- remove global variables, find other ways to deal with it (probably graph state)
- Could give the merger (for system state) to agent as a tool
- Handle if we run out of question for "subgraph"
  - currently if medical questions run through, they are just repeated
- Proper Subgraph for Medical Emergency
  - some nodes could be shared with main graph while others are only required in either, reducing complexity
- Transform Emergency type into a queue / stack /list
  - if none, force agent to fill it
  - if empty, start end of dialogue
