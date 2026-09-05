// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The typed tool system. Skeleton stubs; the typed `Tool` trait, blanket
//! erasure, and `#[tool]` derive land in the tool ticket.

/// A typed capability the agent can invoke.
pub trait Tool {}

/// The object-safe, erased form of a [`Tool`]; `invoke` is the dispatch
/// boundary.
pub trait ErasedTool {}

/// A tool's successful output, normalized for the model.
pub struct ToolOutput;

/// A tool failure split into a model-visible message and operator detail.
pub struct ToolError;

/// A flat progress update streamed while a tool executes.
pub struct ToolUpdate;

/// The concurrency class governing how a batch of tool calls executes.
#[non_exhaustive]
pub enum Concurrency {
    /// Parallelizable reads.
    Safe,
    /// Serialized mutations, run behind a barrier.
    Exclusive,
}

/// The decision returned by the pre-batch approval gate for a single call.
#[non_exhaustive]
pub enum ToolDecision {
    /// Run the call as requested.
    Proceed,
    /// Rewrite the call's arguments, re-running validation.
    Modify {
        /// The replacement arguments.
        arguments: serde_json::Value,
    },
    /// Reject the call with a model-visible message.
    Deny {
        /// The rejection message surfaced to the model.
        message: String,
    },
}
