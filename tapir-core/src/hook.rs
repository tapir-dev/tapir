//! Hooks: asynchronous interception around tool execution.
//!
//! A Hook is registered on the [`RuntimeBuilder`](crate::runtime::RuntimeBuilder)
//! and consulted by the turn loop around every tool call. [`before_tool`] returns
//! a [`ToolDecision`] — allow the call, deny it with a reason, or modify its
//! arguments — and is the dynamic half of the safety story: a deny policy blocks
//! a dangerous call before it runs, and it is the same seam a TUI can later use
//! for interactive approval. [`after_tool`] may replace the result the model sees
//! (redaction / post-processing). Hooks are `async` so a verdict can consult an
//! external service (a permission policy), and they can read the session metadata
//! to decide per conversation.
//!
//! [`before_tool`]: Hook::before_tool
//! [`after_tool`]: Hook::after_tool
//!
//! Hooks are adapter-facing (a bot gates a session by policy). The TUI registers
//! none, so the deny / modify / override paths are exercised only by the seam
//! by adapters — the RPC adapter's per-session deny policy is the first
//! `#[allow(dead_code)]` on the pieces only those callers construct.

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::ToolCall;
use crate::tools::ToolResult;
use crate::tools::tool::Metadata;

/// A before-tool verdict: run the call unchanged, block it with a reason, or run
/// it with replaced arguments.
pub enum ToolDecision {
    /// Run the call as the model requested it.
    Allow,
    /// Block the call; `reason` becomes the tool's (error) result the model sees.
    // Constructed by a policy hook (an adapter) or the seam tests, never by the
    // TUI — see the module note.
    Deny { reason: String },
    /// Run the call, but with these arguments instead of the requested ones.
    ModifyArgs(Value),
}

/// Per-call context handed to a Hook: read-only access to the session's opaque
/// metadata, so a hook can make per-conversation decisions without the engine
/// knowing about the adapter's platform.
pub struct HookCtx<'a> {
    metadata: &'a Metadata,
}

impl<'a> HookCtx<'a> {
    /// Build a context over the session's metadata (the turn loop does this per
    /// call).
    pub fn new(metadata: &'a Metadata) -> Self {
        Self { metadata }
    }

    /// The session's opaque metadata map. (Read by a policy hook; no engine code
    /// interprets it — the RPC adapter's deny policy reads its list here.)
    pub fn metadata(&self) -> &Metadata {
        self.metadata
    }
}

/// An interception point around tool execution, registered on the Runtime and
/// consulted by the turn loop. Object-safe (`async-trait` boxes the future) so a
/// heterogeneous set lives in one registry.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Consulted before a tool call runs. The default allows every call.
    async fn before_tool(
        &self,
        _call: &ToolCall,
        _ctx: &HookCtx<'_>,
    ) -> ToolDecision {
        ToolDecision::Allow
    }

    /// Consulted after a tool call completes (whether it succeeded or errored).
    /// Returning `Some` replaces the [`ToolResult`] the model sees — the call's
    /// error state is preserved across the override. The default leaves the
    /// result unchanged.
    async fn after_tool(
        &self,
        _call: &ToolCall,
        _result: &ToolResult,
        _ctx: &HookCtx<'_>,
    ) -> Option<ToolResult> {
        None
    }
}
