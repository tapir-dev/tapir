//! tapir — build agents in Rust.
//!
//! `tapir` is the headline crate: a thin facade over two layers that you can
//! also depend on directly.
//!
//! - [`tapir-ai`](tapir_ai) — the **LLM layer**: pluggable [`providers`] over a
//!   provider-neutral [`message`] vocabulary, a model [`catalog`], OAuth
//!   [`auth`], provider [`credentials`], and the on-disk [`config`] paths.
//! - [`tapir-core`](tapir_core) — the **agent engine**: the turn loop and
//!   [`session`]s, the [`runtime`] registry, [`tools`], [`hook`]s,
//!   [`observer`]s, [`command`]s, [`prompts`], [`skills`], and the [`store`].
//!
//! Frontends (a TUI, a CLI, a Slack bot, …) are separate crates that depend on
//! this one; the library itself pulls in no terminal or UI machinery.
//!
//! # Getting started
//!
//! [`prelude`] gathers the common front door — the types to build and drive an
//! agent, the values you read off a turn, and the traits you implement to
//! extend it. The deeper, adapter-level surface (session persistence in
//! [`store`]/[`session`], the steering [`queue`], the round internals in
//! [`providers`]) is reached by its module path.
//!
//! ```no_run
//! use tapir::prelude::*;
//!
//! # async fn run() {
//! // A runtime seeds the built-in providers and tools; a session drives turns.
//! let rt = Runtime::builder().build();
//! let mut session = rt.session(".".into());
//! session.submit(Input::text("Hello!"));
//! # }
//! ```

// The LLM layer.
#[doc(inline)]
pub use tapir_ai::{auth, catalog, config, credentials, message, providers};

// The agent engine.
#[doc(inline)]
pub use tapir_core::{
    agent, command, context, hook, observer, prompts, queue, runtime, session,
    skills, store, tools,
};

/// The common front door: one `use tapir::prelude::*;` for the types you build
/// and drive an agent with, the values you read off a turn, and the traits you
/// implement to extend it.
///
/// Everything here is also reachable by its module path. The deeper,
/// adapter-level surface (session persistence, the steering queue, the round
/// internals) is intentionally *not* in the prelude — import it by path when
/// you need it.
pub mod prelude {
    // Build and drive an agent.
    pub use crate::agent::Agent;
    pub use crate::runtime::Runtime;

    // Drive a turn and read what comes back.
    pub use crate::message::{
        Image, Input, ModelRef, Role, RoundError, ToolCall, TurnEvent, Usage,
    };

    // Extension points — implement a trait to plug in your own.
    pub use crate::command::Command;
    pub use crate::credentials::CredentialProvider;
    pub use crate::hook::Hook;
    pub use crate::observer::Observer;
    pub use crate::providers::Provider;
    pub use crate::store::SessionStore;
    pub use crate::tools::Tool;
}
