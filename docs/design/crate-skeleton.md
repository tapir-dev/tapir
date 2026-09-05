# tapir — crate skeleton and public API surface (design spec)

Status: **decided** (wayfinder map #1, ticket #10). This is a design spec, not
implementation. It synthesizes the 13 resolved decisions into a single, mechanical
build target: the workspace layout, feature flags, dependency set, module tree, the
public re-export surface, and the end-to-end quick-start. Implementation is a separate
later effort.

Built on the sibling crate `tapir-provider` (path `../tapir-provider`, unpublished —
so `tapir` is path/workspace-consumed too; crates.io is not on the table until the
provider publishes).

## 1. Workspace layout

Two crates are physically required: the `#[tool]` derive is a proc-macro, and Rust
forbids a proc-macro crate from also exporting normal items. Users still depend only
on `tapir`, which re-exports the derive.

```
tapir/                      (workspace root)
├── Cargo.toml              [workspace] + [workspace.dependencies]
├── CONTEXT.md              domain glossary
├── docs/design/crate-skeleton.md   (this file)
├── tapir/                  the library crate
│   ├── Cargo.toml
│   └── src/...
└── tapir-macros/           the proc-macro crate (#[tool] derive)
    ├── Cargo.toml
    └── src/lib.rs
```

- Reserved for future members: `tapir-store-*` sibling backends (non-reference
  SessionStore implementations; the reference file backend ships inside `tapir` under
  a feature — see §4).
- The current `src/main.rs` coding-agent binary is **out of scope** for this effort
  (the concrete agent / TUI is a separate future effort per the map). `tapir` is
  **lib-only**.

### Root `Cargo.toml`

```toml
[workspace]
resolver = "3"
members = ["tapir", "tapir-macros"]

[workspace.dependencies]
tapir-provider = { path = "../tapir-provider", default-features = false }
async-trait   = "0.1"
futures-core  = "0.3"
futures-util  = { version = "0.3", default-features = false }
serde         = { version = "1", features = ["derive"] }
serde_json    = "1"
tokio         = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time"] }
schemars      = "1.0"
thiserror     = "2"
fs4           = "1"
```

Version rationale: `async-trait`, `futures-core`/`futures-util`, `serde`,
`serde_json` match the pins `tapir-provider` already uses. `tokio` and `schemars` are
dev-only in `tapir-provider` but runtime deps here (the run loop and the `#[tool]`
schema emit need them). `thiserror = "2"` (current major; the provider hand-rolls its
error, so there is no precedent to match). `fs4` backs the file store's advisory lock.
`tokio` features: `sync` → `Mutex`/`broadcast` (state, `subscribe`); `time` →
`idle_timeout` for follow-up.

## 2. `tapir` package Cargo.toml

```toml
[package]
name = "tapir"
version = "0.1.0"
edition = "2024"
license = "ISC"

[dependencies]
tapir-macros  = { path = "../tapir-macros", version = "0.1.0" }
tapir-provider = { workspace = true }
async-trait   = { workspace = true }
futures-core  = { workspace = true }
futures-util  = { workspace = true }
serde         = { workspace = true }
serde_json    = { workspace = true }
tokio         = { workspace = true }
schemars      = { workspace = true }
thiserror     = { workspace = true }
fs4           = { workspace = true, optional = true }

[features]
default = []
anthropic        = ["tapir-provider/anthropic", "tapir-provider/models", "tapir-provider/reqwest"]
openai           = ["tapir-provider/openai",    "tapir-provider/models", "tapir-provider/reqwest"]
models           = ["tapir-provider/models"]
token-store-file = ["tapir-provider/token-store-file"]
store-file       = ["dep:fs4"]
```

Feature story:

- Nothing from `tapir-provider` is enabled by default (`default = []`). The core path
  — `.provider(impl Provider)` — needs no provider feature; you pass any constructed
  provider.
- `models` is **not** hard-enabled. v1 schema handling is pure `NonStrict` passthrough
  (strict/Gemini profiles are fog) and the builder never reads `Model.api`, so the
  core does not need the catalog layer.
- The `anthropic` / `openai` features are what make the `.model("id")` convenience
  (§5) work: they pull the provider adapter, the `models` catalog (for the offline
  baseline + `find_by_id`), and `reqwest` (the default HTTP client).
- `store-file` gates the reference `FileSessionStore` (JSONL) and its `fs4` lock.

### `tapir-macros` Cargo.toml

```toml
[package]
name = "tapir-macros"
version = "0.1.0"
edition = "2024"
license = "ISC"

[lib]
proc-macro = true

[dependencies]
syn        = { version = "2", features = ["full"] }
quote      = "1"
proc-macro2 = "1"
```

## 3. Module tree (`tapir`)

```
tapir
├── lib.rs      crate root re-exports; `pub use tapir_provider;`; `pub mod prelude;`
├── agent       Agent, AgentBuilder, Run, RunHandle, RunId, SteerMode
├── message     AgentMessage, CustomMessage, NoCustom
├── tool        Tool, ErasedTool, ToolOutput, ToolError, ToolUpdate, Concurrency, ToolDecision
├── event       AgentEvent
├── store       SessionStore; store::file::FileSessionStore  (cfg feature = "store-file")
├── schema      SchemaProfile, NonStrict
├── error       Error, Result
└── prelude
```

`ToolDecision` lives in `tool` (it is about a tool call, even though the gate that
produces it is configured on the agent builder, per #15).

## 4. Public API surface

### Crate root (`tapir::`)

Headline nouns, for non-glob users:

```rust
pub use agent::Agent;
pub use event::AgentEvent;
pub use error::{Error, Result};        // Result<T> = core::result::Result<T, Error>
pub use tapir_macros::tool;            // the #[tool] derive, re-exported
pub use tapir_provider;                // reach Provider/Context/Model without a 2nd dep
```

Traits (`Tool`, `SessionStore`, `SchemaProfile`) are intentionally **not** at the root
— prelude-only, to avoid method-resolution surprises.

### `tapir::prelude`

The "just works" glob for the quick-start:

```rust
pub use crate::agent::{Agent, Run, RunHandle};
pub use crate::event::AgentEvent;
pub use crate::tool::{Tool, ToolOutput, ToolError};
pub use crate::store::SessionStore;
pub use crate::error::{Error, Result};
pub use tapir_macros::tool;
// provider essentials
pub use tapir_provider::{
    Provider, Context, Message, ContentPart, AssistantMessage,
    CompletionOptions, ThinkingLevel,
};
```

### Builder surface (synthesis of #8 + the ergonomics decisions)

- `.provider(impl Provider)` — the **canonical** seam. Also the escape hatch for a
  custom HTTP client or explicit credentials (construct e.g.
  `AnthropicProvider::new(my_http, cred, "id")` and pass it here).
- `.model(impl Into<String>)` — feature-gated convenience (§5), mutually exclusive
  with `.provider`.
- `.system(impl Into<String>)`
- `.tool(impl Tool)` — primary; chainable; the only way to add tools of different
  types (a literal array of heterogeneous `#[tool]` fns will not compile). Erases
  internally to `Arc<dyn ErasedTool>`.
- `.tools(impl IntoIterator<Item = Arc<dyn ErasedTool>>)` — bulk, for an
  already-erased collection.
- `.max_tool_iterations(usize)` — default 25 (#8).
- `.before_tool_call(..)` / `.after_tool_call(..)` — approval gate / observe hook (#15).
- `.schema_profile(impl SchemaProfile)` — override the default `NonStrict` (#12).
- `.build() -> Result<Agent, Error>` — **sync**; validates (missing/both of
  provider|model → `Error::Build`) and, on the `.model` path, resolves the provider
  (all registry resolution is sync).

## 5. `.model("id")` convenience (amends #8)

#8 removed `.model()` in favor of `.provider()` because the model rides on the provider
instance (confirmed: `tapir-provider`'s `Provider::complete(ctx, opts)` carries no
model; the model is bound at provider construction). The ergonomics pass reinstates
`.model()` as a feature-gated convenience that builds the provider from the string —
which also restores the map's original north-star shape.

Mechanism (all sync, fully offline):

1. `ModelRegistry::load(None, None)` — offline baseline catalog compiled in via the
   `anthropic`/`openai` provider features.
2. `registry.find_by_id("claude-haiku-4-5")` — resolves a `ModelEntry` (which carries
   `model.provider` and `model.api`).
3. `create_provider(entry, http)` → `Arc<dyn Provider>`, with `http` defaulting to
   `ReqwestClient::new()`.

Credentials: fall back to the provider's env var (e.g. `ANTHROPIC_API_KEY`). HTTP: the
reqwest default. For an explicit key or a custom transport, use `.provider()` instead
(keeps the builder non-generic — no `.http()` override on the `.model()` path).

Caveat (accepted for v1): `find_by_id` is first-match-wins across enabled providers, so
a bare id is ambiguous if two providers expose the same id. In practice anthropic/openai
ids do not collide; disambiguation is fog.

Resolution failures (unknown/ambiguous id, missing adapter) surface at `build()` as
`Error::Build` or `Error::Provider`.

## 6. Error surface note (#6/#7)

`tapir::Result<T> = Result<T, tapir::Error>` covers the sync/build paths. The **run
future** yields `Result<AssistantMessage, Arc<tapir::Error>>` — `Arc` is deliberate
(#6), so the terminal error can also ride the `subscribe()` broadcast to multiple
receivers. `Arc<Error>` does not `?`-convert into `Error`, so quick-start `main`
returns `Result<(), Arc<tapir::Error>>` (or uses an application error type). No
`From<Arc<Error>>` shim in v1 — the type stays honest. Reply text comes from
`AssistantMessage::text_content()`.

## 7. Quick-start (the <15-line north-star)

```rust
use tapir::prelude::*;

#[tool]
/// Get the weather for a city.
async fn weather(city: String) -> String {
    format!("sunny in {city}")
}

#[tokio::main]
async fn main() -> Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5")        // ANTHROPIC_API_KEY + reqwest default
        .system("You are helpful.")
        .tool(weather)
        .build()?;

    let reply = agent.prompt("Weather in Paris?").await?;
    println!("{}", reply.text_content());
    Ok(())
}
```

Requires `tapir = { path = "../tapir", features = ["anthropic"] }`. The agent-facing
core (`builder().model().system().tool().build()` + `prompt().await`) is six lines and
matches the north-star.

## 8. Synthesis findings (amendments and fog)

Amendments to prior decisions, made while assembling:

- **Amends #8** — builder regains a feature-gated `.model(str)` convenience alongside
  the canonical `.provider(p)` (§5).
- **Amends #12** — the "auto-select `SchemaProfile` from `Model.api`" idea is inert in
  v1 (all profiles are `NonStrict` passthrough, and the builder holds a bare
  `dyn Provider` with no `Api`). It graduates to fog: it can fire only once a
  non-passthrough profile ships **and** the `Provider` trait surfaces its `Api`.
  Consequently `tapir` does **not** hard-depend on `tapir-provider/models`.

Graduated fog (new):

- **`SchemaProfile` auto-selection from `Api`** — needs a provider-side `Api` accessor
  + a non-passthrough profile; neither exists today.
- **Heterogeneous tuple `.tools((a, b, c))`** — one-call multiple tools of differing
  types via tuple trait impls (axum/bevy style). Deferred; chaining `.tool()` covers
  v1.
- **`.http()` override on the `.model()` path** — would make the whole builder generic
  over the HTTP client; the `.provider()` seam covers custom transport for v1.
- **Bare-id disambiguation** — a provider-qualified `.model("provider", "id")` (or
  `"provider/id"`) to resolve the `find_by_id` first-match ambiguity.
