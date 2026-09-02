# tapir

Minimal, extensible sdk coagent.

## Build & test

`just` drives everything; run `just --list` for the full recipe set. Tests run
under `cargo-nextest`, so `cargo test` misses the config — use the recipes.

- `just check` — full local CI gate; run before calling any change done.
- `just test` — fast tests (nextest, all targets/features).
- `just test-doc` — doctests (nextest doesn't run these).
- `cargo nextest run <name>` — a single test.

## Agent skills

- **Issue tracker** — GitHub Issues (`tapir-dev/tapir`) via `gh`. See `docs/agents/issue-tracker.md`.
- **Triage labels** — canonical triage labels. See `docs/agents/triage-labels.md`.
- **Domain docs** — `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.
