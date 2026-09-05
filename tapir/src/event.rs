// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The flat agent event stream. One `match` sees the whole vocabulary of a run,
//! and one channel type carries all of it. Nesting is by convention, not by
//! type: an `AgentStart` brackets N turns; each `TurnStart..TurnEnd` brackets
//! one message (`MessageStart` / `MessageUpdate`* / `MessageEnd`). Every mid-run
//! event carries a `turn`, so a flat consumer can re-derive the nesting. This
//! ticket ships the core (tool-free) variants; tool-execution variants join the
//! set in a later ticket.

use std::sync::Arc;

use tapir_provider::{AssistantMessage, StreamEvent};

use crate::agent::RunId;
use crate::error::Error;

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

    /// The turn finished: its message settled (and, later, any tool batch
    /// drained).
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

    /// The run finished successfully. Terminal item; carries the settled final
    /// reply — the message that stopped without asking for a tool.
    AgentEnd {
        /// The run that finished.
        run: RunId,
        /// The final assistant message.
        message: AssistantMessage,
    },
}
