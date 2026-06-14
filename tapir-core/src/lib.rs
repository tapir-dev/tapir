//! tapir-core — the engine SDK: the agent loop and Sessions, the lifecycle
//! Event vocabulary, and the pluggable seams (Tools, Hooks, Observers,
//! Commands, session storage), all registered on a [`runtime::Runtime`] built
//! once per process. The LLM layer (providers, catalog, auth, credentials,
//! config) lives in [`tapir-ai`](../tapir_ai/index.html); this crate depends on
//! it and re-exports its modules for convenience. Frontends (Adapters) — the
//! TUI, RPC or bot adapter — depend on this crate (or the `tapir` facade) and
//! drive Sessions; this crate depends on no terminal or UI machinery.

pub mod agent;
pub mod command;
pub mod context;
pub mod hook;
pub mod observer;
pub mod prompts;
pub mod queue;
pub mod runtime;
pub mod session;
pub mod skills;
pub mod store;
pub mod tools;

// The LLM layer moved to `tapir-ai`. Re-exported here so this crate's own
// `crate::providers`/`crate::config`/… paths and existing dependents that
// reference `tapir_core::providers` keep resolving unchanged.
pub use tapir_ai::{auth, catalog, config, credentials, message, providers};
