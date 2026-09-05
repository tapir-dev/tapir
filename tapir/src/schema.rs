// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Per-provider tool-schema normalization on the send path. Skeleton stubs; v1
//! ships the `NonStrict` passthrough only.

/// Per-provider normalization of a tool's JSON schema on the send path.
pub trait SchemaProfile {}

/// The default profile: passthrough, no active rewrite.
pub struct NonStrict;
