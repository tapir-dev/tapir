// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The persistence seam. A [`SessionStore`] is a narrow, object-safe async
//! backend the agent writes history through as a run progresses; absent one, the
//! agent is ephemeral. One store is one session — branch and rewind stay a
//! backend concern, out of the seam.

use async_trait::async_trait;

use crate::error::Error;
use crate::message::AgentMessage;

#[cfg(feature = "store-file")]
pub mod file;

/// A backend failure from a [`SessionStore`]. Normalized to [`Error::Session`]
/// at the run loop boundary so a store error fails the run closed rather than
/// silently losing history.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct SessionError(String);

impl SessionError {
    /// Wrap a backend failure in a store error carrying an operator message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl From<SessionError> for Error {
    fn from(error: SessionError) -> Self {
        Error::Session(error.0)
    }
}

/// The persistence seam: append-one plus load of [`AgentMessage`]s, held by the
/// agent as `Option<Arc<dyn SessionStore<M>>>`. Absent (`None`, the default) the
/// agent is ephemeral.
///
/// Object-safe and async (`#[async_trait]`) so a concrete backend erases behind
/// an `Arc` shared into every run. Durable-on-await: once an [`append`] resolves
/// the message is persisted, so there is no separate `flush`.
///
/// [`append`]: SessionStore::append
#[async_trait]
pub trait SessionStore<M>: Send + Sync {
    /// Persist one message, in order. Resolving `Ok` means it is durable.
    async fn append(
        &self,
        message: &AgentMessage<M>,
    ) -> Result<(), SessionError>;

    /// Load the full session history in append order. A fresh session returns
    /// an empty vector.
    async fn load(&self) -> Result<Vec<AgentMessage<M>>, SessionError>;
}
