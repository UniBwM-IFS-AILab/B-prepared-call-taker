# Architecture

The project uses a **ports-and-adapters** structure with runtime plugin discovery:

- shared orchestration backend: `SessionService`
- pluggable policies via entry points: `ems_prepared.policies`
- pluggable frontends via entry points: `ems_prepared.frontends`
- one launcher script: `main`

## Module responsibilities

- `src/ems_prepared/model/context.py`
  - session-scoped deps object (`Settings`)
  - runtime transport/storage handles (`emit`, optional `request_input`)
- `src/ems_prepared/model/contracts.py`
  - shared protocols and DTOs (`SessionManager`, `ConversationPolicy`, `PolicyFactory`, `FrontendPlugin`, `SessionRecorder`)
- `src/ems_prepared/model/session_service.py`
  - shared session lifecycle/orchestration (`SessionManager` implementation)
  - in-memory live session registry (`_SessionEntry`)
  - resolves policies from configured factories and delegates recording
- `src/ems_prepared/model/session_recording.py`
  - default file-based session recording implementation (`FileSessionRecorder`)
  - owns canonical session output files:
    `manifest.json`, `events.jsonl`, `survey.json`,
    `final_state.json`, `state_schema.json`, `message_history.json`, `deps.json`
- `src/ems_prepared/model/errors.py`
  - shared backend boundary exceptions (`SessionNotFoundError`, `UnsupportedPolicyError`)
- `src/ems_prepared/plugins.py`
  - runtime discovery/validation for policy and frontend entry points
- `src/ems_prepared/main.py`
  - launcher composition and subcommand dispatch (`list`, `<frontend>`)
  - constructs one `SessionService` from discovered policy factories + `FileSessionRecorder`
- `src/ems_prepared/policies/*`
  - concrete policy runtimes/factories (for example `graph`, `agent`)
- `src/ems_prepared/adapters/*`
  - transport/UI plugins implementing `FrontendPlugin`
  - maps HTTP/WebSocket/UI payloads to model DTOs and back
- `experiments/*`
  - experiment-specific datasets, configs, and executable entry scripts
  - may depend on reusable code in `src/ems_prepared/*`
  - should not define project-wide generic CLI entrypoints under top-level `scripts/`

## Dependency rules

Allowed direction:

- `adapters -> model`
- `policies -> model`
- `main/plugins -> model` plus runtime loading of adapters/policies via entry points

Forbidden direction:

- `model` importing concrete adapters or concrete policies
- `policies` importing adapters
- session output recording depending on frontend frameworks
- model contracts/context carrying framework-specific objects

## Runtime flow

1. `main` discovers frontend and policy plugins.
2. `main` builds `SessionService(policy_factories=..., session_recorder=FileSessionRecorder())`.
3. Selected frontend plugin receives the shared `SessionManager`.
4. Frontend builds `SessionParameters` and starts a session.
5. `SessionService` creates `Settings`, resolves the selected `PolicyFactory`, and creates one session policy runtime.
6. `ConversationPolicy` emits backend events.
7. `SessionService` records manifest/events/survey/completion artifacts via `SessionRecorder`.
8. Frontend serializes backend events to its transport.

## Two storage concerns

- Policy-state persistence (resume/recovery) belongs to policy implementations (for graph policy under `src/ems_prepared/policies/pydantic_graph/custom_persistence/`).
- Session output recording (manifest/events/survey/final artifacts) belongs to shared backend code in `src/ems_prepared/model/session_recording.py`.

## Invariants

- one active `_SessionEntry` per session ID
- completion is detected from backend events (`kind == "completed"`)
- policy runtimes are session-scoped (not shared globally)
- session output files are scoped to each session save path
