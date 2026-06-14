//! Observers: synchronous, fire-and-forget consumers of every [`TurnEvent`].
//!
//! An Observer is registered on the [`RuntimeBuilder`](crate::runtime::RuntimeBuilder)
//! and notified — in order, inline with the event stream — of every event a
//! session emits. It is read-only: it cannot alter, delay, or veto a turn (that
//! is a Hook's job, in a later slice). Keep `on_event` cheap; it runs on the
//! relay path between the engine and the frontend.

use crate::agent::TurnEvent;

/// A read-only consumer of a session's [`TurnEvent`] stream. Registered on the
/// Runtime; one process can have many (a logger, a metrics sink, a bot bridge).
pub trait Observer: Send + Sync {
    /// Called once for every event, in emission order, before the frontend sees
    /// it. Must not block for long — observers share the relay task.
    fn on_event(&self, event: &TurnEvent);
}
