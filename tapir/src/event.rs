// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The flat agent event stream. Skeleton stub; the variants land in the
//! walking-skeleton ticket, embedding `tapir_provider::StreamEvent` verbatim.

/// The single flat event type streamed from a run. Nesting is by convention via
/// a `turn` field on mid-run variants.
#[non_exhaustive]
pub enum AgentEvent {}
