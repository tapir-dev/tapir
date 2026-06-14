# tapir

This is the main source code repository for `tapir`, a library for building
LLM agents in Rust. It gives you a provider-neutral LLM layer and an agent
engine — the turn loop, sessions, tools, and hooks — and nothing else.

## Why tapir?

- **Provider-neutral.** One message vocabulary and model catalog across LLM
  backends; bring your own provider by implementing a trait.
- **Just a library.** No terminal, no CLI, no bot. Frontends are separate
  projects that depend on this one.
- **Composable.** A turn loop, sessions, tools, and hooks you wire together —
  dependencies point strictly downward, no cycles.

## Quick start

```toml
[dependencies]
tapir = { git = "https://github.com/tapir-dev/tapir" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

```rust
use tapir::runtime::Runtime;
use tapir::agent::Input;

#[tokio::main]
async fn main() {
    let rt = Runtime::builder().build();
    let mut session = rt.session(".".into());
    let _events = session.run(Input::text("Hello!"));
    // drive `_events` (a stream of TurnEvent) — see examples/hello-agent.
}
```

## Examples

The `examples/` directory is its own workspace with runnable agents:

- `hello-agent` — build a runtime, drive a turn.
- `custom-provider` — plug in your own LLM backend (a no-network echo provider).
- `custom-tool` — register a tool the model can call (a `multiply` calculator).
- `hooks-and-memory` — the `rt.agent()` builder with a hook, plus persisting and
  resuming a session through a `SessionStore`.

```sh
cd examples
cargo run --bin hello-agent                          # offline: prints setup
ANTHROPIC_API_KEY=sk-… cargo run --bin hello-agent "Say hi"
```

## Building from source

```sh
cargo build --workspace
cargo test --workspace      # some tests bind localhost / spawn processes
cargo doc -p tapir --no-deps --open
```

Common tasks are wrapped in a [`justfile`](justfile) — run `just` to list them:

```sh
just check        # type-check the workspace
just test         # run the test suite
just deny         # lint the dependency graph (cargo-deny)
```

Releasing is GitHub-only and also driven through `just`; see
[RELEASING.md](RELEASING.md).

## License

tapir is free and open-source software licensed under the
[ISC License](LICENSE).

It builds on third-party crates whose licenses are listed in
[THIRD-PARTY-LICENSE](THIRD-PARTY-LICENSE).
