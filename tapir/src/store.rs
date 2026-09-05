// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The persistence seam. Skeleton stubs; the object-safe async `SessionStore`
//! and the reference file backend land in the store ticket.

#[cfg(feature = "store-file")]
pub mod file;

/// The persistence seam: append-one plus load of `AgentMessage`s. Absent, the
/// agent is ephemeral.
pub trait SessionStore {}
