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
use tapir_provider::{ContentPart, ImageSource, Message};

/// The content of one user turn for the multimodal entry points
/// ([`converse_with`](crate::agent::Agent::converse_with),
/// [`prompt_with`](crate::agent::Agent::prompt_with), and
/// [`steer_input`](crate::agent::RunHandle::steer_input)): an ordered list of
/// [`ContentPart`]s mixing text and images.
///
/// A bare string converts in as a single text part, so the text-only path reads
/// exactly as [`prompt`](crate::agent::Agent::prompt) /
/// [`converse`](crate::agent::Agent::converse) do; richer input is composed with
/// [`text`](Self::text) / [`image`](Self::image) and the `with_*` chain:
///
/// ```
/// use tapir::message::UserInput;
/// use tapir::tapir_provider::{ImageSource, MediaType};
///
/// let input = UserInput::text("what is in this screenshot?")
///     .with_image(ImageSource::bytes(MediaType::Png, PNG_BYTES.to_vec()));
/// assert!(input.has_image());
/// # const PNG_BYTES: &[u8] = b"hi";
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserInput {
    parts: Vec<ContentPart>,
}

impl UserInput {
    /// A user turn beginning with one text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![ContentPart::text(text)],
        }
    }

    /// A user turn beginning with one image part.
    #[must_use]
    pub fn image(source: impl Into<ImageSource>) -> Self {
        Self {
            parts: vec![ContentPart::image(source)],
        }
    }

    /// A user turn from an explicit list of content parts.
    #[must_use]
    pub fn parts(parts: impl Into<Vec<ContentPart>>) -> Self {
        Self {
            parts: parts.into(),
        }
    }

    /// Append a text part, returning the input for chaining.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(ContentPart::text(text));
        self
    }

    /// Append an image part, returning the input for chaining.
    #[must_use]
    pub fn with_image(mut self, source: impl Into<ImageSource>) -> Self {
        self.parts.push(ContentPart::image(source));
        self
    }

    /// Whether any part is an image. The multimodal entry points consult this to
    /// reject an image bound for a non-multimodal model before it is sent.
    #[must_use]
    pub fn has_image(&self) -> bool {
        self.parts
            .iter()
            .any(|part| matches!(part, ContentPart::Image(_)))
    }

    /// The content parts, consuming the input.
    #[must_use]
    pub fn into_parts(self) -> Vec<ContentPart> {
        self.parts
    }
}

impl From<String> for UserInput {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for UserInput {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<ContentPart> for UserInput {
    fn from(part: ContentPart) -> Self {
        Self { parts: vec![part] }
    }
}

impl From<Vec<ContentPart>> for UserInput {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self { parts }
    }
}

impl From<UserInput> for Vec<ContentPart> {
    fn from(input: UserInput) -> Self {
        input.parts
    }
}

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
///
/// Serde is externally tagged (`{"Llm": …}` / `{"Custom": …}`), the wire form a
/// [`SessionStore`](crate::store::SessionStore) persists one message per line.
/// The derives bound `M` on `Serialize`/`Deserialize`, so a custom message type
/// round-trips through a store for free.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tapir_provider::{ImageSource, MediaType};

    #[test]
    fn a_string_converts_to_one_text_part_and_no_image() {
        let input: UserInput = "hello".into();
        assert!(!input.has_image());
        assert_eq!(input.into_parts(), vec![ContentPart::text("hello")]);
    }

    #[test]
    fn the_with_chain_orders_text_then_image() {
        let input = UserInput::text("look at this")
            .with_image(ImageSource::bytes(MediaType::Png, b"hi".to_vec()));
        assert!(input.has_image());
        assert_eq!(
            input.into_parts(),
            vec![
                ContentPart::text("look at this"),
                ContentPart::image(ImageSource::bytes(
                    MediaType::Png,
                    b"hi".to_vec()
                )),
            ]
        );
    }

    #[test]
    fn image_constructor_reports_an_image() {
        let input = UserInput::image(ImageSource::url("https://x/cat.png"));
        assert!(input.has_image());
    }

    #[test]
    fn explicit_parts_round_trip() {
        let parts = vec![
            ContentPart::text("a"),
            ContentPart::image(ImageSource::base64(MediaType::Jpeg, "aGk=")),
        ];
        let input = UserInput::parts(parts.clone());
        assert!(input.has_image());
        assert_eq!(input.into_parts(), parts);
    }

    #[test]
    fn default_carries_no_parts_and_no_image() {
        let input = UserInput::default();
        assert!(!input.has_image());
        assert!(input.into_parts().is_empty());
    }
}
