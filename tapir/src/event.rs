// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The flat agent event stream. One `match` sees the whole vocabulary of a run,
//! and one channel type carries all of it. Nesting is by convention, not by
//! type: an `AgentStart` brackets N turns; each `TurnStart..TurnEnd` brackets
//! one message (`MessageStart` / `MessageUpdate`* / `MessageEnd`). Every mid-run
//! event carries a `turn`, so a flat consumer can re-derive the nesting. A
//! tool-requesting turn also brackets each call it runs
//! (`ToolExecutionStart` / `ToolExecutionUpdate`* / `ToolExecutionEnd`), nested
//! inside the turn after its `MessageEnd`.

use std::sync::Arc;

use tapir_provider::{AssistantMessage, StreamEvent, ToolResultMessage};

use crate::agent::RunId;
use crate::error::Error;
use crate::tool::ToolUpdate;

/// The single flat event type streamed from a run.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The run began: one `prompt`/`resume` call, about to span one or more
    /// turns.
    AgentStart {
        /// Distinguishes this run on a session-wide subscription.
        run: RunId,
    },

    /// A turn began: one provider completion (plus, later, its tool batch).
    TurnStart {
        /// Zero-based turn index within the run.
        turn: usize,
    },

    /// The assistant message for this turn began streaming.
    MessageStart {
        /// The turn this message belongs to.
        turn: usize,
    },

    /// A provider delta for the in-flight message, carrying
    /// [`tapir_provider::StreamEvent`] verbatim — the agent re-encodes nothing.
    /// A token-by-token consumer renders these; a consumer that only wants the
    /// settled reply ignores them and waits for [`AgentEvent::MessageEnd`].
    MessageUpdate {
        /// The turn this delta belongs to.
        turn: usize,
        /// The provider stream event, forwarded as-is.
        delta: StreamEvent,
    },

    /// The assistant message settled. The agent already folded the deltas
    /// through a `StreamAccumulator`, so you never fold them yourself. This
    /// message is also appended to the agent's history.
    MessageEnd {
        /// The turn this message belongs to.
        turn: usize,
        /// The settled assistant message.
        message: AssistantMessage,
    },

    /// A tool call in this turn's batch was dispatched. Brackets one call's
    /// handling, told apart from siblings by `call_id`. Emitted for every call
    /// in model order, including one naming an unknown tool (whose
    /// [`ToolExecutionEnd`](Self::ToolExecutionEnd) carries a synthetic
    /// `is_error` result).
    ToolExecutionStart {
        /// The turn whose batch this call belongs to.
        turn: usize,
        /// The model-supplied id of the call being run.
        call_id: String,
        /// The name of the tool being invoked.
        name: String,
    },

    /// A progress update streamed by a tool mid-execution, forwarded verbatim
    /// from the tool's `on_update` sink onto the one flat stream.
    ToolExecutionUpdate {
        /// The turn whose batch this call belongs to.
        turn: usize,
        /// The id of the call that emitted the update.
        call_id: String,
        /// The tool's progress update.
        update: ToolUpdate,
    },

    /// A tool call finished. Carries the settled [`ToolResultMessage`] — the
    /// same result appended to history and fed back to the model — so a consumer
    /// sees the outcome (including [`is_error`](ToolResultMessage::is_error))
    /// without re-deriving it.
    ToolExecutionEnd {
        /// The turn whose batch this call belonged to.
        turn: usize,
        /// The settled result carried back to the model.
        result: ToolResultMessage,
    },

    /// A tool call was rejected by the [`before_tool_call`](crate::agent::AgentBuilder::before_tool_call)
    /// gate before the batch ran. The call never executed, so it emits no
    /// `ToolExecution*` events; instead a synthetic `is_error`
    /// [`ToolResultMessage`] carrying `message` is fed back to the model in the
    /// call's model-order slot. Emitted during the pre-batch gate pass, so in
    /// model order but before any
    /// [`ToolExecutionStart`](Self::ToolExecutionStart): the whole batch is
    /// gated before any of it runs.
    ToolCallDenied {
        /// The turn whose batch this call belonged to.
        turn: usize,
        /// The model-supplied id of the denied call.
        id: String,
        /// The rejection message surfaced to the model.
        message: String,
    },

    /// The turn finished: its message settled and any tool batch drained.
    TurnEnd {
        /// The turn that finished.
        turn: usize,
    },

    /// The run failed. A first-class terminal event carrying the SDK error, so
    /// a session-wide subscriber sees failures the same way it sees success. It
    /// is the last event of a failed run — no [`AgentEvent::AgentEnd`] follows.
    /// `turn` is `None` when the failure is not tied to a specific turn.
    /// `Arc` because [`Error`] is not `Clone`.
    Error {
        /// The turn the failure occurred in, if any.
        turn: Option<usize>,
        /// The terminal error.
        error: Arc<Error>,
    },

    /// A [`converse`](crate::agent::Agent::converse) run parked at a tool-free
    /// reply instead of ending, awaiting more input. Not terminal: the run's
    /// future stays pending. A steer wakes it (resetting the tool-iteration cap)
    /// and the loop resumes with a fresh [`TurnStart`](Self::TurnStart);
    /// [`RunHandle::finish`](crate::agent::RunHandle::finish) or the builder's
    /// `idle_timeout` ends it, resolving the run with
    /// [`AgentEnd`](Self::AgentEnd). Never emitted by a `prompt`/`resume` run,
    /// which always ends at a tool-free reply.
    Idle {
        /// The run that parked.
        run: RunId,
    },

    /// The run finished successfully. Terminal item; carries the settled final
    /// reply — the message that stopped without asking for a tool.
    AgentEnd {
        /// The run that finished.
        run: RunId,
        /// The final assistant message.
        message: AssistantMessage,
    },
}
