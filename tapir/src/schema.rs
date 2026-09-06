// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Per-provider tool-schema normalization on the send path. Each turn the run
//! loop offers its tools to the model; the active [`SchemaProfile`] gets a
//! chance to rewrite each tool's argument schema in place first, so a provider
//! that needs a stricter or reshaped schema than the portable one can adapt it
//! without the tool author knowing. v1 ships only the [`NonStrict`] passthrough
//! (strict/reshaping profiles are deferred); the seam is here so a later profile
//! drops in without touching the run loop.

use serde_json::Value;

/// Per-provider normalization of a tool's JSON schema on the send path.
///
/// Applied to each offered tool's argument schema every turn, in place, before
/// the provider call. Implementations must be cheap and idempotent: the same
/// schema may be normalized on every turn of a run. `Send + Sync` because the
/// active profile is shared into each spawned run.
pub trait SchemaProfile: Send + Sync {
    /// Rewrite `schema` in place for the target provider. The default profile
    /// leaves it untouched.
    fn normalize(&self, schema: &mut Value);
}

/// The default profile: passthrough, no active rewrite. The tool's portable
/// schema is sent to the provider verbatim.
pub struct NonStrict;

impl SchemaProfile for NonStrict {
    fn normalize(&self, _schema: &mut Value) {}
}
