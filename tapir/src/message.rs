// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The SDK message supertype. `AgentMessage<M>` layers over provider
//! `Message`s: a standard message maps 1:1 to a provider `Message`, while a
//! caller-defined `Custom(M)` either converts to a `Message` through the
//! [`CustomMessage::to_llm`] seam or stays UI-only. The generic parameter `M`
//! defaults to the uninhabited [`NoCustom`], so an agent with no custom
//! messages writes plain `AgentMessage` and pays zero ceremony. Building the
//! type generic from the start keeps the later custom-message and
//! `transform_context` work from being a wide retype.

use serde::{Deserialize, Serialize};
use tapir_provider::Message;

/// The uninhabited default custom-message type: the zero-ceremony common case.
/// It cannot be constructed, so [`AgentMessage::Custom`] is statically
/// unreachable and the compiler monomorphizes the custom paths away.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoCustom {}

/// A caller-defined message type carried in history. [`to_llm`](Self::to_llm)
/// decides how one custom message shows up to the model: `Some(message)` sends
/// it, `None` keeps it UI-only.
pub trait CustomMessage {
    /// Convert this custom message into a provider [`Message`], or `None` to
    /// keep it out of the model's context.
    fn to_llm(&self) -> Option<Message>;
}

impl CustomMessage for NoCustom {
    fn to_llm(&self) -> Option<Message> {
        // Uninhabited: no value of `NoCustom` exists, so there are no arms.
        match *self {}
    }
}

/// The SDK's message supertype layered over provider messages, generic over a
/// custom-message type `M` defaulting to [`NoCustom`].
#[derive(Debug, Clone)]
pub enum AgentMessage<M = NoCustom> {
    /// A standard message that maps 1:1 to a provider [`Message`].
    Llm(Message),
    /// An app-defined message: custom role, UI-only note, approval marker, and
    /// so on. Routed to the model through [`CustomMessage::to_llm`].
    Custom(M),
}

/// The `transform_context` seam: prune, compact, or inject on the agent history
/// *before* [`convert_to_llm`]. Borrow the history in, own the reshaped history
/// out. The default (no transform) skips it entirely. The builder that exposes
/// this seam lands in a later ticket; the generic [`AgentMessage`] spine exists
/// now so that wiring is not a wide retype.
pub type TransformContext<M> =
    dyn Fn(&[AgentMessage<M>]) -> Vec<AgentMessage<M>> + Send + Sync;

/// Flatten agent history into the provider messages one turn sends. Standard
/// messages clone into the request (the provider `Context` owns its
/// `Vec<Message>`); custom messages route through [`CustomMessage::to_llm`] and
/// may drop out.
pub fn convert_to_llm<M: CustomMessage>(
    history: &[AgentMessage<M>],
) -> Vec<Message> {
    history
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Llm(msg) => Some(msg.clone()),
            AgentMessage::Custom(c) => c.to_llm(),
        })
        .collect()
}
