// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The SDK message supertype. Skeleton stubs; the generic supertype and its
//! conversion seams are fleshed out in the walking-skeleton ticket.

use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

/// The uninhabited default custom-message type: the zero-ceremony common case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoCustom {}

/// A caller-defined message type carried in history. It either converts to a
/// provider `Message` or stays UI-only.
pub trait CustomMessage {}

/// The SDK's message supertype layered over provider messages, generic over a
/// custom-message type `M` defaulting to [`NoCustom`].
#[allow(dead_code)]
pub struct AgentMessage<M = NoCustom> {
    marker: PhantomData<M>,
}
