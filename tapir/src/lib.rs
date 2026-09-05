// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! `tapir` is a stateful LLM agent SDK built on the sibling `tapir-provider`
//! crate. It adds a stateful [`Agent`], a run loop, a typed tool system, a flat
//! [`AgentEvent`] stream, and a persistence seam on top of the provider's
//! single-completion surface.
//!
//! This is the workspace skeleton: the module tree and public re-export surface
//! are in place; the behaviour lands in later tickets.

pub mod agent;
pub mod error;
pub mod event;
pub mod message;
pub mod prelude;
pub mod schema;
pub mod store;
pub mod tool;

pub use agent::Agent;
pub use error::{Error, Result};
pub use event::AgentEvent;
// The `#[tool]` derive lives in the sibling proc-macro crate, re-exported here
// so users depend only on `tapir`.
pub use tapir_macros::tool;
// Reach `Provider`/`Context`/`Model` without a second dependency.
pub use tapir_provider;
