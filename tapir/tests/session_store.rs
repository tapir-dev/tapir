// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the `SessionStore` persistence seam: write-through
//! ordering (user before the provider call, artifacts after), fail-closed on a
//! store error with the partial discarded, `Agent::resume` seeding history and
//! continuing, custom + UI-only `AgentMessage<M>` serde round-trip through a
//! store, and the ephemeral default (no store persists nothing). Everything runs
//! against an in-crate fake `Provider` and an in-memory JSONL store — no network,
//! no filesystem.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tapir::message::{AgentMessage, CustomMessage, NoCustom};
use tapir::store::{SessionError, SessionStore};
use tapir::{Agent, AgentEvent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents, Usage,
};

/// An in-memory reference `SessionStore` backing on serialized JSONL, so serde
/// really round-trips on the append/load path. The `lines` handle is shared so a
/// provider can peek at how much is persisted before a completion. `fail_at`
/// makes the `append` at that index (0-based) fail, to exercise fail-closed.
struct MemStore<M> {
    lines: Arc<Mutex<Vec<String>>>,
    fail_at: Option<usize>,
    _marker: PhantomData<fn() -> M>,
}

impl<M> MemStore<M> {
    fn new() -> Self {
        Self {
            lines: Arc::new(Mutex::new(Vec::new())),
            fail_at: None,
            _marker: PhantomData,
        }
    }

    /// A store whose `append` at index `n` fails.
    fn failing_at(n: usize) -> Self {
        Self {
            fail_at: Some(n),
            ..Self::new()
        }
    }

    /// The shared serialized log, for a provider to peek at or a test to count.
    fn lines(&self) -> Arc<Mutex<Vec<String>>> {
        self.lines.clone()
    }
}

#[async_trait]
impl<M> SessionStore<M> for MemStore<M>
where
    M: CustomMessage + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn append(
        &self,
        message: &AgentMessage<M>,
    ) -> Result<(), SessionError> {
        let mut lines = self.lines.lock().unwrap();
        if self.fail_at == Some(lines.len()) {
            return Err(SessionError::new("simulated store failure"));
        }
        let line = serde_json::to_string(message)
            .map_err(|e| SessionError::new(e.to_string()))?;
        lines.push(line);
        Ok(())
    }

    async fn load(&self) -> Result<Vec<AgentMessage<M>>, SessionError> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .map(|line| {
                serde_json::from_str(line)
                    .map_err(|e| SessionError::new(e.to_string()))
            })
            .collect()
    }
}

/// A `Provider` replaying one scripted reply per completion. It records, at each
/// call, how many messages are already persisted in the shared store log and how
/// many messages the context carries, so a test can prove the user turn is
/// durable before the call and that resumed history seeds the context.
struct RecordingProvider {
    reply: Vec<StreamEvent>,
    lines: Arc<Mutex<Vec<String>>>,
    persisted_at_call: Arc<Mutex<Vec<usize>>>,
    ctx_len_at_call: Arc<Mutex<Vec<usize>>>,
}

impl RecordingProvider {
    fn new(reply: Vec<StreamEvent>, lines: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            reply,
            lines,
            persisted_at_call: Arc::new(Mutex::new(Vec::new())),
            ctx_len_at_call: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn persisted_at_call(&self) -> Arc<Mutex<Vec<usize>>> {
        self.persisted_at_call.clone()
    }

    fn ctx_len_at_call(&self) -> Arc<Mutex<Vec<usize>>> {
        self.ctx_len_at_call.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        self.persisted_at_call
            .lock()
            .unwrap()
            .push(self.lines.lock().unwrap().len());
        self.ctx_len_at_call
            .lock()
            .unwrap()
            .push(ctx.messages.len());
        self.reply.clone()
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    async fn complete(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Ok(StreamAccumulator::fold(&self.record(ctx)))
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        let events = self.record(ctx);
        Ok(Box::pin(futures_util::stream::iter(
            events.into_iter().map(Ok::<StreamEvent, ProviderError>),
        )))
    }
}

/// A scripted reply that streams `text` and stops (no tool calls).
fn text_script(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart,
        StreamEvent::TextStart { index: 0 },
        StreamEvent::TextDelta {
            index: 0,
            text: text.to_string(),
        },
        StreamEvent::TextEnd { index: 0 },
        StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        },
    ]
}

/// Assert an `AgentMessage` is a standard user message, returning its text.
fn assert_user(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Llm(Message::User { content }) => {
            content.iter().fold(String::new(), |mut acc, part| {
                if let tapir_provider::ContentPart::Text(t) = part {
                    acc.push_str(t);
                }
                acc
            })
        }
        other => panic!("expected a user message, got {other:?}"),
    }
}

/// Assert an `AgentMessage` is an assistant reply, returning its text.
fn assert_assistant(message: &AgentMessage) -> String {
    match message {
        AgentMessage::Llm(Message::Assistant(a)) => a.text_content(),
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn run_persists_user_before_call_and_reply_after() {
    let store = Arc::new(MemStore::<NoCustom>::new());
    let provider =
        RecordingProvider::new(text_script("hello back"), store.lines());
    let persisted_at_call = provider.persisted_at_call();

    let agent = Agent::builder()
        .provider(provider)
        .store(store.clone())
        .build()
        .expect("build");

    let reply = agent.prompt("hello").await.expect("run");
    assert_eq!(reply.text_content(), "hello back");

    // The user turn was durable before the provider was called.
    assert_eq!(
        *persisted_at_call.lock().unwrap(),
        vec![1],
        "the user message must be persisted before the provider call"
    );

    // The store holds the user turn, then the settled reply, in that order.
    let history = store.load().await.expect("load");
    assert_eq!(history.len(), 2);
    assert_eq!(assert_user(&history[0]), "hello");
    assert_eq!(assert_assistant(&history[1]), "hello back");
}

#[tokio::test]
async fn store_error_terminates_run_and_discards_partial() {
    // Fail the second append — the assistant reply — after the user turn (append
    // 0) is already durable.
    let store = Arc::new(MemStore::<NoCustom>::failing_at(1));
    let provider =
        RecordingProvider::new(text_script("never durable"), store.lines());

    let agent = Agent::builder()
        .provider(provider)
        .store(store.clone())
        .build()
        .expect("build");

    let err = agent.prompt("hello").await.expect_err("store fails closed");
    assert!(
        matches!(&*err, Error::Session(_)),
        "expected Error::Session, got {err:?}"
    );

    // The partial (the assistant reply) was discarded: only the user turn is
    // durable.
    let history = store.load().await.expect("load");
    assert_eq!(history.len(), 1);
    assert_eq!(assert_user(&history[0]), "hello");
}

#[tokio::test]
async fn resume_seeds_history_and_continues() {
    let store = Arc::new(MemStore::<NoCustom>::new());

    // First session: one prompt persists [user, assistant].
    {
        let provider =
            RecordingProvider::new(text_script("first reply"), store.lines());
        let agent = Agent::builder()
            .provider(provider)
            .store(store.clone())
            .build()
            .expect("build");
        agent.prompt("first").await.expect("first run");
    }
    assert_eq!(store.load().await.expect("load").len(), 2);

    // Reopen: resume seeds the two prior messages, then a new prompt continues.
    let provider =
        RecordingProvider::new(text_script("second reply"), store.lines());
    let ctx_len_at_call = provider.ctx_len_at_call();
    let resumed =
        Agent::resume(Agent::builder().provider(provider), store.clone())
            .await
            .expect("resume");

    let reply = resumed.prompt("second").await.expect("second run");
    assert_eq!(reply.text_content(), "second reply");

    // The resumed run's context carried the two seeded messages plus the new
    // user turn.
    assert_eq!(*ctx_len_at_call.lock().unwrap(), vec![3]);

    // The store now holds the full four-message conversation, in order.
    let history = store.load().await.expect("load");
    assert_eq!(history.len(), 4);
    assert_eq!(assert_user(&history[0]), "first");
    assert_eq!(assert_assistant(&history[1]), "first reply");
    assert_eq!(assert_user(&history[2]), "second");
    assert_eq!(assert_assistant(&history[3]), "second reply");
}

/// A caller-defined message type: one variant converts to a model message, the
/// other stays UI-only. Serde-derived so it round-trips through a store.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Note {
    /// A note surfaced only in the UI; never reaches the model.
    Ui(String),
    /// A note that converts to a user message for the model.
    Say(String),
}

impl CustomMessage for Note {
    fn to_llm(&self) -> Option<Message> {
        match self {
            Note::Ui(_) => None,
            Note::Say(text) => Some(Message::user(text.clone())),
        }
    }
}

#[tokio::test]
async fn custom_and_ui_only_messages_round_trip() {
    let store = Arc::new(MemStore::<Note>::new());

    let originals: Vec<AgentMessage<Note>> = vec![
        AgentMessage::Llm(Message::user("plain")),
        AgentMessage::Custom(Note::Ui("ui-only banner".to_string())),
        AgentMessage::Custom(Note::Say("converts to a turn".to_string())),
    ];
    for message in &originals {
        store.append(message).await.expect("append");
    }

    let loaded = store.load().await.expect("load");

    // Serde fidelity: what came back serializes identically to what went in.
    assert_eq!(
        serde_json::to_value(&originals).unwrap(),
        serde_json::to_value(&loaded).unwrap(),
    );

    // The externally-tagged wire form is what the file backend will write, one
    // message per line.
    let lines = store.lines();
    let lines = lines.lock().unwrap();
    assert!(lines[1].contains("\"Custom\""));
    assert!(lines[1].contains("\"Ui\""));
}

/// A `Provider` advancing through a fixed list of scripted replies, one per
/// completion, clamping at the last. Enough to drive a tool-requesting turn
/// followed by a tool-free reply.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: std::sync::atomic::AtomicUsize,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn next_script(&self) -> Vec<StreamEvent> {
        let i = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.scripts[i.min(self.scripts.len() - 1)].clone()
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn complete(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Ok(StreamAccumulator::fold(&self.next_script()))
    }

    async fn complete_stream(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        Ok(Box::pin(futures_util::stream::iter(
            self.next_script()
                .into_iter()
                .map(Ok::<StreamEvent, ProviderError>),
        )))
    }
}

/// A reply requesting a single tool call.
fn tool_call_script(id: &str, name: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart,
        StreamEvent::ToolCallStart {
            index: 0,
            id: id.to_string(),
            name: name.to_string(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            partial_json: "null".to_string(),
        },
        StreamEvent::ToolCallEnd { index: 0 },
        StreamEvent::Done {
            finish_reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ]
}

/// A trivial no-argument tool returning a fixed line.
struct Ping;

#[async_trait]
impl tapir::tool::Tool for Ping {
    type Args = ();
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "ping"
    }
    fn description(&self) -> &str {
        "returns pong"
    }
    fn concurrency(&self) -> tapir::tool::Concurrency {
        tapir::tool::Concurrency::Safe
    }

    async fn execute(
        &self,
        (): (),
        _ctx: &tapir::tool::ToolCtx,
        _on_update: &mut tapir::tool::UpdateSink<'_>,
    ) -> Result<String, String> {
        Ok("pong".to_string())
    }
}

#[tokio::test]
async fn tool_run_persists_results_in_order() {
    let store = Arc::new(MemStore::<NoCustom>::new());
    let provider = ScriptedProvider::new(vec![
        tool_call_script("call_1", "ping"),
        text_script("all done"),
    ]);

    let agent = Agent::builder()
        .provider(provider)
        .tool(Ping)
        .store(store.clone())
        .build()
        .expect("build");

    let reply = agent.prompt("go").await.expect("run");
    assert_eq!(reply.text_content(), "all done");

    // Full history in order: user, the tool-requesting reply, its tool result,
    // then the settled tool-free reply.
    let history = store.load().await.expect("load");
    assert_eq!(history.len(), 4);
    assert_eq!(assert_user(&history[0]), "go");
    assert!(matches!(
        &history[1],
        AgentMessage::Llm(Message::Assistant(_))
    ));
    match &history[2] {
        AgentMessage::Llm(Message::ToolResult(r)) => {
            assert_eq!(r.tool_call_id, "call_1");
            assert!(!r.is_error);
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
    assert_eq!(assert_assistant(&history[3]), "all done");
}

#[tokio::test]
async fn default_agent_is_ephemeral() {
    // A store exists but is never attached: the default agent must persist
    // nothing to it.
    let store = Arc::new(MemStore::<NoCustom>::new());
    let provider =
        RecordingProvider::new(text_script("ephemeral"), store.lines());

    let agent = Agent::builder().provider(provider).build().expect("build");

    let events: Vec<AgentEvent> = agent.prompt("hi").collect().await;
    assert!(matches!(events.last(), Some(AgentEvent::AgentEnd { .. })));

    assert!(
        store.load().await.expect("load").is_empty(),
        "an agent with no store must be ephemeral"
    );
}
