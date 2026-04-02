# Runtime flow

```text
main.py
  ├─ load frontend plugins (entry points: ems_prepared.frontends)
  ├─ load policy factories (entry points: ems_prepared.policies)
  ├─ create SessionService(policy_factories, FileSessionRecorder)
  └─ run selected frontend subcommand plugin
       └─ adapter -> SessionManager -> ConversationPolicy -> BackendEvents
```

## Notes

- Frontend and policy selection happens at runtime through entry-point discovery.
- Frontend option discoverability is parser-backed (`main --help`, `main <frontend> --help`).
- `SessionService` receives policy factories and resolves policy names itself.
- Invalid policy names are rejected in the backend (`UnsupportedPolicyError`).
- `main` defines shared `--policy`, `--locale`, and `--experiment` once for all frontend subcommands.
