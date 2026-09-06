// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The "just works" glob for the quick-start. Traits (`Tool`, `SessionStore`,
//! `SchemaProfile`) live here rather than at the crate root to avoid
//! method-resolution surprises.

pub use crate::agent::{Agent, Run, RunHandle, SteerMode};
pub use crate::error::{Error, Result};
pub use crate::event::AgentEvent;
pub use crate::message::UserInput;
pub use crate::schema::{NonStrict, SchemaProfile};
pub use crate::store::SessionStore;
pub use crate::tool::{
    BoxFuture, Tool, ToolCall, ToolDecision, ToolError, ToolOutput,
};
pub use tapir_macros::tool;

// Provider essentials, so the quick-start needs no second dependency.
pub use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, ImageSource,
    MediaType, Message, Provider, ThinkingLevel,
};
