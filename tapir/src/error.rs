// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The runtime error surface. `tapir::Error` wraps `tapir_provider::Error`
//! verbatim (preserving `kind`/`status`/`retry_after`) and adds the SDK-native
//! variants. Tool-execution failures are deliberately *not* modeled here: they
//! ride back to the model as a `ToolResult { is_error }`.

/// The SDK's runtime error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A provider-level failure, wrapped verbatim.
    #[error(transparent)]
    Provider(#[from] tapir_provider::Error),

    /// A session-store failure (fail-closed).
    #[error("session store error: {0}")]
    Session(String),

    /// The run exceeded its tool-requesting turn cap.
    #[error("exceeded max tool iterations: {limit}")]
    MaxIterations {
        /// The configured limit that was exceeded.
        limit: usize,
    },

    /// The run was cancelled cooperatively.
    #[error("run cancelled")]
    Cancelled,

    /// The agent builder was misconfigured (validated synchronously at build).
    #[error("agent build error: {0}")]
    Build(String),

    /// A user turn carried an image, but the bound model is known not to accept
    /// image input. Raised by the multimodal entry points before anything is
    /// sent, so the image is rejected outright rather than silently dropped or
    /// left for the provider to refuse. Only the `.model("id")` path knows a
    /// model's modalities; a hand-supplied `.provider(..)` leaves them unknown,
    /// so this never fires there.
    #[error("the bound model does not accept image input")]
    ImageUnsupported,
}

/// The SDK result alias covering the sync/build paths.
pub type Result<T> = core::result::Result<T, Error>;
