//! Provider-neutral message vocabulary and the round contract.
//!
//! The conversation types every provider client shares — the message, images,
//! token usage, a tool call, the round-error wrapper, the streamed
//! [`TurnEvent`], the [`ToolDef`] schema — plus the [`RoundRunner`] seam between
//! the turn loop (in [`tapir-core`](../../tapir_core/index.html)) and the
//! provider clients in [`crate::providers`]. The wire protocols themselves live
//! in [`crate::providers`] (one module per API shape).

use serde_json::Value;

/// Token counts from a finished turn. Cost is computed by the caller (it knows
/// the active model's per-token prices from the catalog).
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub total: u64,
}

/// An image attachment (base64-encoded), sent to the model as an image block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub mime: String,
    pub data: String,
}

/// A conversation message to send (shell `!!` output is never included).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    /// Image attachments (from `@image.png` references) — user messages only.
    pub images: Vec<Image>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// A user-authored input to the engine: what the transcript shows (`display`),
/// what the model receives (`model_text`, with `@path` references expanded), and
/// any image attachments. [`Input::text`] builds the common case where the shown
/// and sent text are identical and there are no images.
#[derive(Debug, Clone)]
pub struct Input {
    pub display: String,
    pub model_text: String,
    pub images: Vec<Image>,
}

impl Input {
    /// A plain text input where the shown and sent text are identical.
    // Used by tests and the canonical-history slice; the queue path builds the
    // full struct directly (distinct display vs `@`-expanded model text).
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self { display: text.clone(), model_text: text, images: Vec::new() }
    }
}

/// The model a session is pointed at: a provider id and one of its model ids.
/// Session state — the `model` command sets it and `compact` reads it; an
/// adapter spawning turns does the same. (The TUI tracks its selection in its
/// own auth state for now.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub provider: String,
    pub id: String,
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.id)
    }
}

/// A tool call the model requested in a round.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub call_id: String,
    pub name: String,
    pub args: Value,
}

/// The model-facing schema of a tool: its name, description, and JSON-schema
/// parameters. The tool *registry* and execution live in `tapir-core`; this is
/// just the shape each provider client serializes into its own wire format.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

/// A failed round. Transient failures (connection errors, 429, 5xx) are retried
/// inside the HTTP client (`reqwest-retry`), so by the time one surfaces here it
/// is terminal — only the message is shown.
#[derive(Debug)]
pub struct RoundError {
    pub message: String,
}

impl std::fmt::Display for RoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RoundError {}

/// Streamed updates from an in-flight agent turn, produced by the provider
/// clients and fed into the app's event loop. The app tracks which transcript
/// entry is "current", so events carry no index.
pub enum TurnEvent {
    /// Opens a run (one `run_turn`): the outermost lifecycle boundary.
    AgentStart,
    /// Closes a run — the last event of `run_turn`, after `Done`/`Error`.
    AgentEnd,
    /// Opens a Turn: one round of the loop — an assistant response plus any
    /// tool calls it makes. A run is one or more Turns.
    TurnStart,
    /// Closes a Turn, after its assistant message and any tool calls. Emitted
    /// only when the round concludes normally (like `AgentEnd`, not on error).
    TurnEnd,
    /// Opens the assistant's message within a Turn — the following `Text` /
    /// `Thinking` deltas belong to it.
    MessageStart,
    /// Closes the assistant's message, once its deltas have all streamed.
    MessageEnd,
    /// A chunk of assistant text arrived.
    Text { delta: String },
    /// A chunk of the model's reasoning summary (the "thinking" stream).
    Thinking { delta: String },
    /// The model started a tool call (`title` is the rendered header). `call_id`
    /// correlates this with its [`ToolEnd`](TurnEvent::ToolEnd) (and lets a Hook
    /// match a result to its call).
    ToolStart { call_id: String, name: String, title: String },
    /// A chunk of streaming output from a still-running tool, tagged with the
    /// same `call_id` as its start. Emitted by streaming tools (e.g. `bash`, as
    /// its command writes output); the SDK's start/update/end tool vocabulary.
    ToolUpdate { call_id: String, delta: String },
    /// A tool call finished, tagged with the same `call_id` as its start.
    ToolEnd { call_id: String, output: String, is_error: bool, took_ms: u64 },
    /// One round's token usage — credited to the footer immediately, so the
    /// counters climb as the turn progresses (and survive a mid-turn cancel).
    Usage { usage: Usage },
    /// A queued steering / follow-up message was delivered into the running
    /// turn; show it as a user message and persist it.
    QueuedDelivered { display: String, model_text: String, images: Vec<Image> },
    /// The turn finished.
    Done,
    /// Auto-compaction produced this summary; replace older context with it.
    Compacted { summary: String },
    /// The turn failed; show the message.
    Error { message: String },
}

/// Map a Tapir thinking level to an OpenAI-style `reasoning.effort` string.
/// `None` means: send no reasoning block (the model decides / off).
pub fn reasoning_effort(level: &str) -> Option<&'static str> {
    match level {
        "off" => None,
        "minimal" => Some("low"), // Copilot maps minimal → low
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        _ => Some("medium"),
    }
}

/// Produces one streamed round of a turn — the seam between the turn loop and
/// the provider clients. The live implementation
/// ([`LiveRounds`](crate::providers::LiveRounds)) calls the provider's stream;
/// tests script outcomes so a turn runs with no network. Static dispatch
/// (`run_turn` is generic over `R`), so no boxing and no `async-trait`.
#[allow(async_fn_in_trait)]
pub trait RoundRunner {
    async fn run_round(
        &self,
        history: &[crate::providers::Step],
        tx: &tokio::sync::mpsc::Sender<TurnEvent>,
    ) -> Result<crate::providers::RoundOutcome, RoundError>;
}
