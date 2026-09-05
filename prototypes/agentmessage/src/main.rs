//! PROTOTYPE — throwaway. Answers wayfinder ticket #4: how to encode pi's
//! declaration-merged `AgentMessage` (custom message roles + `convert_to_llm`
//! + `transform_context`) in Rust, which has no declaration merging.
//!
//! NOT production code. Run:
//!   cargo run --manifest-path prototypes/agentmessage/Cargo.toml
//!
//! The verdict lives in the ticket resolution comment; this file is the
//! concrete artifact to react to. It builds against the real `tapir-provider`
//! types to prove the encoding integrates.

use serde::{Deserialize, Serialize};
use tapir_provider::{AssistantMessage, Context, Message, SystemPrompt};

// ===========================================================================
// The encoding
// ===========================================================================

/// The agent's message supertype. `M` is the app's custom-message type and
/// defaults to `NoCustom`, so an agent with no custom messages writes plain
/// `AgentMessage` and pays zero ceremony. A settled `AssistantMessage` is a
/// provider `Message`, so replies append back with no conversion (no lossy
/// round-trip).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessage<M = NoCustom> {
    /// Standard message: maps 1:1 to a provider `Message`.
    Provider(Message),
    /// App-defined message: custom role, UI-only note, approval marker, ...
    Custom(M),
}

/// Uninhabited "this agent has no custom messages" marker. It can't be
/// constructed, so `AgentMessage::Custom` is statically unreachable and the
/// compiler monomorphizes the custom paths away.
#[derive(Debug, Clone)]
pub enum NoCustom {}

/// The `convert_to_llm` seam, per custom type: how one custom message shows up
/// to the model. `None` = invisible (UI-only). Object-safe by construction.
pub trait CustomMessage {
    fn to_llm(&self) -> Option<Message>;
}

impl CustomMessage for NoCustom {
    fn to_llm(&self) -> Option<Message> {
        match *self {} // uninhabited: no arms needed
    }
}

// Erased custom messages (see object_safe_alternative): forward through the box.
impl CustomMessage for Box<dyn CustomMessage> {
    fn to_llm(&self) -> Option<Message> {
        (**self).to_llm()
    }
}

/// `convert_to_llm` at the history level. Standard messages clone into the
/// request (provider `Context` owns `Vec<Message>`, so exactly one
/// clone-per-turn is unavoidable — see FINDINGS); custom messages route
/// through the seam and may drop out.
pub fn convert_to_llm<M: CustomMessage>(history: &[AgentMessage<M>]) -> Vec<Message> {
    history
        .iter()
        .filter_map(|m| match m {
            AgentMessage::Provider(msg) => Some(msg.clone()),
            AgentMessage::Custom(c) => c.to_llm(),
        })
        .collect()
}

/// The `transform_context` seam: prune / compact / inject on the agent history
/// *before* `convert_to_llm`. Borrow-in, own-out, so the no-op common case is
/// skipped entirely (see `build_context`). Compaction lives here — a seam, not
/// a built-in.
pub type TransformContext<M> = dyn Fn(&[AgentMessage<M>]) -> Vec<AgentMessage<M>> + Send + Sync;

/// Assemble the provider `Context` for one turn. The common path (no transform)
/// takes no history snapshot; `convert_to_llm` borrows directly.
pub fn build_context<M: CustomMessage>(
    history: &[AgentMessage<M>],
    system: Option<&SystemPrompt>,
    transform: Option<&TransformContext<M>>,
) -> Context {
    let messages = match transform {
        Some(f) => convert_to_llm(&f(history)),
        None => convert_to_llm(history),
    };
    let mut ctx = Context::new(messages);
    if let Some(s) = system {
        ctx = ctx.with_system(s.clone());
    }
    ctx
}

// ===========================================================================
// Example: an app that DOES add custom messages
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AppMessage {
    /// Visible to the model, injected as a user turn.
    Developer(String),
    /// Purely cosmetic conversation divider — never sent to the model.
    UiDivider,
}

impl CustomMessage for AppMessage {
    fn to_llm(&self) -> Option<Message> {
        match self {
            AppMessage::Developer(t) => Some(Message::user(format!("[dev] {t}"))),
            AppMessage::UiDivider => None,
        }
    }
}

// ===========================================================================
// Drive it — surface state after every step
// ===========================================================================

fn main() {
    common_case();
    rule();
    custom_case();
    rule();
    object_safe_alternative();
}

fn rule() {
    println!("\n{}\n", "=".repeat(72));
}

fn dump<M: std::fmt::Debug>(label: &str, history: &[AgentMessage<M>]) {
    println!("{label} ({} messages):", history.len());
    for m in history {
        println!("   {m:?}");
    }
}

/// Zero-ceremony common case: `M = NoCustom`, custom variant unconstructible.
fn common_case() {
    println!("## Common case — no custom messages (M = NoCustom)\n");
    let history: Vec<AgentMessage> = vec![
        AgentMessage::Provider(Message::user("what's 2 + 2?")),
        AgentMessage::Provider(Message::Assistant(AssistantMessage::text("4"))),
    ];
    // AgentMessage::Custom(..) is impossible here: nothing inhabits NoCustom.

    dump("agent history", &history);
    let ctx = build_context(&history, None, None);
    println!("\n-> provider Context: {} messages", ctx.messages.len());
    for m in &ctx.messages {
        println!("   {m:?}");
    }
}

/// Custom messages + `transform_context` + persistence round-trip.
fn custom_case() {
    println!("## Custom case — Developer + UiDivider, with transform_context\n");
    let history: Vec<AgentMessage<AppMessage>> = vec![
        AgentMessage::Custom(AppMessage::Developer("prefer terse answers".into())),
        AgentMessage::Provider(Message::user("hi")),
        AgentMessage::Custom(AppMessage::UiDivider),
        AgentMessage::Provider(Message::Assistant(AssistantMessage::text("hello"))),
        AgentMessage::Provider(Message::user("and 3 + 3?")),
    ];
    dump("agent history (full)", &history);

    // transform_context = compaction seam: keep only the last 3 agent messages.
    let prune_last3: Box<TransformContext<AppMessage>> =
        Box::new(|h| h.iter().rev().take(3).rev().cloned().collect());

    let ctx = build_context(&history, None, Some(prune_last3.as_ref()));
    println!(
        "\n-> after transform_context (last 3) + convert_to_llm: {} provider messages",
        ctx.messages.len()
    );
    for m in &ctx.messages {
        println!("   {m:?}");
    }
    println!("   (UiDivider dropped as UI-only; Developer pruned out of the window)");

    // Persistence seam: the whole agent history serde round-trips as-is.
    let json = serde_json::to_string(&history).unwrap();
    let back: Vec<AgentMessage<AppMessage>> = serde_json::from_str(&json).unwrap();
    println!(
        "\n-> serde round-trip preserved {} agent messages (SessionStore seam)",
        back.len()
    );
}

/// The trade-off the ticket must settle: erase the custom type instead of
/// carrying `M` as a generic. `CustomMessage` is object-safe, so this compiles
/// for `convert_to_llm`; the cost is that `Box<dyn CustomMessage>` is neither
/// `Serialize` nor `Clone`, so the SessionStore needs `erased-serde` plumbing.
fn object_safe_alternative() {
    println!("## Object-safe alternative — AgentMessage<Box<dyn CustomMessage>>\n");

    let history: Vec<AgentMessage<Box<dyn CustomMessage>>> = vec![
        AgentMessage::Custom(Box::new(AppMessage::Developer("erased!".into()))),
        AgentMessage::Provider(Message::user("ping")),
    ];
    let msgs = convert_to_llm(&history);
    println!("-> erased history -> {} provider messages", msgs.len());
    for m in &msgs {
        println!("   {m:?}");
    }
    println!("   Works for convert_to_llm; loses derive(Serialize)/Clone -> needs erased-serde.");
}
