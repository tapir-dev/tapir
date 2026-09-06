// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the [`RunHandle`] control surface: aborting a live run
//! and steering it from another task. The handle is taken before the `Run` is
//! consumed by `.await`, so these prove it outlives the run and that calls made
//! after termination are silent no-ops. Everything runs against an in-crate
//! scripted [`Provider`] — no network.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ToolCtx, UpdateSink};
use tapir::{Agent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context,
    Error as ProviderError, FinishReason, Message, Provider, StreamAccumulator,
    StreamEvent, StreamEvents, Usage,
};

/// A `Provider` that replays one scripted reply per completion, recording the
/// user-turn texts each completion saw in its context so a test can prove which
/// steered input was picked up, and in what order.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
    user_texts: Arc<Mutex<Vec<Vec<String>>>>,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
            user_texts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle onto the per-completion ordered user-text log.
    fn user_texts(&self) -> Arc<Mutex<Vec<Vec<String>>>> {
        self.user_texts.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        let texts: Vec<String> = ctx
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(
                    content
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                _ => None,
            })
            .collect();
        self.user_texts.lock().unwrap().push(texts);
        let i = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.scripts[i.min(self.scripts.len() - 1)].clone()
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
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

/// A scripted reply requesting one tool call.
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
            partial_json: json!({}).to_string(),
        },
        StreamEvent::ToolCallEnd { index: 0 },
        StreamEvent::Done {
            finish_reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ]
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

/// A `Safe` tool that flags when it starts, then parks forever — so the only way
/// it ever stops is the batch executor aborting its task on the cascade.
struct Blocker {
    started: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

#[async_trait]
impl tapir::tool::Tool for Blocker {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "blocker"
    }
    fn description(&self) -> &str {
        "parks until aborted"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        _args: NoArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        self.started.store(true, Ordering::SeqCst);
        std::future::pending::<()>().await;
        // Unreachable while aborted; flips only if the task ran to completion.
        self.finished.store(true, Ordering::SeqCst);
        Ok("done".to_string())
    }
}

/// A `Safe` tool that flags when it starts, then waits for the test to release
/// it. The gate lets a test enqueue a steer while the batch is genuinely in
/// flight, so the injection is deterministic rather than racing the turn.
struct Gate {
    started: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

#[async_trait]
impl tapir::tool::Tool for Gate {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "gate"
    }
    fn description(&self) -> &str {
        "waits until the test releases it"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        _args: NoArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        self.started.store(true, Ordering::SeqCst);
        while !self.release.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Ok("ok".to_string())
    }
}

#[tokio::test]
async fn abort_cancels_the_run_and_cascades_the_batch() {
    let started = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let provider =
        ScriptedProvider::new(vec![tool_call_script("c1", "blocker")]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Blocker {
            started: started.clone(),
            finished: finished.clone(),
        })
        .build()
        .expect("build");

    // Take the handle before `.await` consumes the run, then drive the run on
    // another task so the handle is genuinely controlling it from the outside.
    let run = agent.prompt("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // Abort only once the blocker is genuinely in flight.
    while !started.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    handle.abort();
    // Idempotent: a second abort is harmless.
    handle.abort();

    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("abort must terminate the run, not hang")
        .expect("run task");
    let error = result.expect_err("aborted run must fail");
    assert!(
        matches!(*error, Error::Cancelled),
        "an aborted run terminates with Error::Cancelled, got {error:?}"
    );
    assert!(
        !finished.load(Ordering::SeqCst),
        "the in-flight tool must be aborted, not run to completion"
    );

    // Handle calls after termination are silent no-ops.
    handle.abort();
    handle.steer("too late");
}

#[tokio::test]
async fn steer_appends_a_user_turn_picked_up_next_turn() {
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    // Turn 0 requests the gate; turn 1 (after the steered turn lands) answers.
    let provider = ScriptedProvider::new(vec![
        tool_call_script("g", "gate"),
        text_script("done"),
    ]);
    let user_texts = provider.user_texts();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Gate {
            started: started.clone(),
            release: release.clone(),
        })
        .build()
        .expect("build");

    let run = agent.prompt("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    while !started.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // Enqueue the steer while the batch is in flight, then let the turn finish.
    handle.steer("injected");
    release.store(true, Ordering::SeqCst);

    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must complete")
        .expect("run task")
        .expect("run");
    assert_eq!(reply.text_content(), "done");

    // The second completion saw the steered user turn appended after the first.
    let seen = user_texts.lock().unwrap().clone();
    assert_eq!(seen.len(), 2, "the run ran two turns");
    assert_eq!(
        seen[1],
        ["go", "injected"],
        "the steered turn must be appended and picked up next turn"
    );
}

#[tokio::test]
async fn steer_with_replace_overrides_pending_input() {
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let provider = ScriptedProvider::new(vec![
        tool_call_script("g", "gate"),
        text_script("done"),
    ]);
    let user_texts = provider.user_texts();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Gate {
            started: started.clone(),
            release: release.clone(),
        })
        .build()
        .expect("build");

    let run = agent.prompt("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    while !started.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    // Append then Replace before the turn boundary: Replace discards the pending
    // append, so only the replacement lands.
    handle.steer("first");
    handle.steer_with("second", tapir::agent::SteerMode::Replace);
    release.store(true, Ordering::SeqCst);

    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must complete")
        .expect("run task")
        .expect("run");
    assert_eq!(reply.text_content(), "done");

    let seen = user_texts.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1],
        ["go", "second"],
        "Replace must override the pending appended input"
    );
}
