# Changelog

All notable changes to this project are documented in this file.

This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-14

First release of tapir, a library for building LLM agents in Rust: a
provider-neutral LLM layer and an agent engine, and nothing else.

### Added

- `tapir-ai` — the LLM layer: providers, model catalog, OAuth/credentials, and
  the provider-neutral message vocabulary.
- `tapir-core` — the agent engine: turn loop, sessions, runtime, tools, hooks,
  observers, commands, prompts, skills, store.
- `tapir` — facade crate re-exporting both layers as one.
- Examples: `hello-agent`, `custom-provider`, `custom-tool`, and
  `hooks-and-memory`.

[0.1.0]: https://github.com/tapir-dev/tapir/releases/tag/v0.1.0
