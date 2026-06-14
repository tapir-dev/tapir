//! tapir-ai — the LLM layer: a neutral conversation history driving multiple
//! providers (Anthropic, the Chat-Completions shape, Gemini, the OpenAI
//! Responses API), the model [`catalog`], provider [`credentials`] and OAuth
//! [`auth`] (incl. GitHub Copilot), plus the on-disk [`config`] paths shared
//! across the stack. Knows nothing about the agent loop, tools, or sessions —
//! [`tapir-core`](../tapir_core/index.html) sits on top of this crate.

pub mod auth;
pub mod catalog;
pub mod config;
pub mod credentials;
pub mod message;
pub mod providers;
