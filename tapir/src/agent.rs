// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The stateful agent and its run surface. Skeleton stubs; the builder, run
//! loop, and control surface land in later tickets.

/// The stateful object owning the in-memory conversation history and driving
/// runs.
pub struct Agent;

/// The fluent builder for an [`Agent`].
pub struct AgentBuilder;

/// One invocation of the agent: both a stream of [`crate::event::AgentEvent`]
/// and a future resolving to the final reply.
pub struct Run;

/// The cloneable control surface (abort / steer / finish) that outlives the
/// [`Run`].
pub struct RunHandle;

/// Disambiguates concurrent runs on the session-wide broadcast.
pub struct RunId;

/// How a steer injection combines with the pending user input.
#[non_exhaustive]
pub enum SteerMode {
    /// Append the steer text as a new user turn.
    Append,
    /// Replace the pending user input with the steer text.
    Replace,
}
