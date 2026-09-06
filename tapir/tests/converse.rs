// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for [`Agent::converse`]: an interactive run that parks at a
//! tool-free reply in the idle state instead of ending. It emits
//! [`AgentEvent::Idle`], wakes on a steer (which resets the tool-iteration cap),
//! and ends only on [`RunHandle::finish`], the builder's `idle_timeout`, or an
//! abort — at which point the run's future resolves with the final reply. A
//! `prompt` run, by contrast, still terminates at a tool-free reply. Everything
//! runs against an in-crate scripted [`Provider`] — no network.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ToolCtx, UpdateSink};
use tapir::{Agent, AgentEvent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Provider, StreamAccumulator, StreamEvent, StreamEvents,
    Usage,
};
use tokio::sync::broadcast;

/// A `Provider` that replays one scripted reply per completion; the last script
/// repeats once the cursor runs past the list, so a parked run that wakes and
/// re-prompts keeps getting an answer.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
        }
    }

    fn next_script(&self) -> Vec<StreamEvent> {
        let i = self.cursor.fetch_add(1, Ordering::SeqCst);
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
        let events = self.next_script();
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

/// The empty argument object shared by the test tools.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

/// A trivial `Safe` tool that records that it ran and returns at once. Used to
/// consume the tool-iteration budget so a test can prove a steer re-arms it.
struct Noop {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl tapir::tool::Tool for Noop {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok("ok".to_string())
    }
}

/// A `Safe` tool that flags when it starts, then waits for the test to release
/// it — so a test can act (e.g. call `finish`) while a turn's batch is genuinely
/// in flight, before the run reaches a park.
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

/// Spin until `flag` is set, so a test only acts once the tool is genuinely in
/// flight. Callers wrap it in a timeout.
async fn wait_set(flag: &AtomicBool) {
    while !flag.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// Watch `events` and report whether the run parks (an [`AgentEvent::Idle`]
/// arrives) before it ends ([`AgentEvent::AgentEnd`]) or the stream closes.
async fn parked_before_end(
    mut events: broadcast::Receiver<AgentEvent>,
) -> bool {
    loop {
        match events.recv().await {
            Ok(AgentEvent::Idle { .. }) => return true,
            Ok(AgentEvent::AgentEnd { .. }) | Err(_) => return false,
            Ok(_) => continue,
        }
    }
}

/// Await the next [`AgentEvent::Idle`] on `events`, bounded by a timeout so a run
/// that fails to park does not hang the test. Panics if the stream ends or the
/// run reaches [`AgentEvent::AgentEnd`] first.
async fn wait_for_idle(events: &mut broadcast::Receiver<AgentEvent>) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match events.recv().await {
                Ok(AgentEvent::Idle { .. }) => return,
                Ok(AgentEvent::AgentEnd { .. }) => {
                    panic!("run ended before it parked in Idle")
                }
                Ok(_) => continue,
                Err(err) => panic!("event stream closed before Idle: {err}"),
            }
        }
    })
    .await
    .expect("a run must park in Idle within the timeout");
}

#[tokio::test]
async fn converse_parks_in_idle_then_finish_ends_the_run() {
    let provider = ScriptedProvider::new(vec![text_script("hi")]);
    let agent = Agent::builder().provider(provider).build().expect("build");

    let mut events = agent.subscribe();
    let run = agent.converse("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // A tool-free reply parks the run rather than ending it.
    wait_for_idle(&mut events).await;
    // Still parked: the future stays pending until the run is finished.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !task.is_finished(),
        "converse must park at a tool-free reply, not resolve its future"
    );

    // finish() ends the parked run, resolving it with the parked reply.
    handle.finish();
    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("finish must end the parked run, not hang")
        .expect("run task")
        .expect("run");
    assert_eq!(reply.text_content(), "hi");

    // A call after termination is a silent no-op.
    handle.finish();
}

#[tokio::test]
async fn steer_wakes_the_parked_run_and_resets_the_cap() {
    // A cap of 1 admits exactly one tool-requesting turn per parked segment.
    // Segment one spends it (tool call, then a tool-free reply that parks);
    // after a steer wakes the run the second tool call is admissible only
    // because the cap was re-armed. Without the reset it would be the run's
    // second tool turn and fail with MaxIterations.
    let provider = ScriptedProvider::new(vec![
        tool_call_script("t1", "noop"),
        text_script("mid"),
        tool_call_script("t2", "noop"),
        text_script("end"),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = Agent::builder()
        .provider(provider)
        .max_tool_iterations(1)
        .tool(Noop {
            calls: calls.clone(),
        })
        .build()
        .expect("build");

    let mut events = agent.subscribe();
    let run = agent.converse("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // First park, after the cap-spending tool turn and its tool-free reply.
    wait_for_idle(&mut events).await;
    handle.steer("more");
    // Second park, after the re-armed tool turn and its tool-free reply.
    wait_for_idle(&mut events).await;
    handle.finish();

    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must complete")
        .expect("run task")
        .expect("a re-armed cap must not fail the run with MaxIterations");
    assert_eq!(reply.text_content(), "end");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "both tool turns must run: the steer re-armed the spent cap"
    );
}

#[tokio::test]
async fn idle_timeout_ends_the_parked_run() {
    let provider = ScriptedProvider::new(vec![text_script("hi")]);
    let agent = Agent::builder()
        .provider(provider)
        .idle_timeout(Some(Duration::from_millis(50)))
        .build()
        .expect("build");

    // No steer, no finish: the idle timeout alone must end the run.
    let reply =
        tokio::time::timeout(Duration::from_secs(5), agent.converse("go"))
            .await
            .expect("the idle timeout must end the parked run")
            .expect("run");
    assert_eq!(reply.text_content(), "hi");
}

#[tokio::test]
async fn prompt_still_terminates_at_a_tool_free_reply() {
    let provider = ScriptedProvider::new(vec![text_script("done")]);
    let agent = Agent::builder().provider(provider).build().expect("build");

    // Subscribe before the run so the watcher sees every event; the broadcast
    // buffers from the subscription point, so it need not race the run's start.
    let watcher = tokio::spawn(parked_before_end(agent.subscribe()));

    // A prompt run resolves at the first tool-free reply, without parking.
    let reply =
        tokio::time::timeout(Duration::from_secs(5), agent.prompt("go"))
            .await
            .expect("prompt must terminate promptly")
            .expect("run");
    assert_eq!(reply.text_content(), "done");
    assert!(
        !watcher.await.expect("watcher task"),
        "a prompt run must never emit Idle"
    );
}

#[tokio::test]
async fn abort_while_parked_terminates_with_cancelled() {
    let provider = ScriptedProvider::new(vec![text_script("hi")]);
    let agent = Agent::builder().provider(provider).build().expect("build");

    let mut events = agent.subscribe();
    let run = agent.converse("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // Once parked in Idle, an abort ends the run with Error::Cancelled rather
    // than resolving it with the parked reply.
    wait_for_idle(&mut events).await;
    handle.abort();

    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("abort must terminate the parked run, not hang")
        .expect("run task");
    let error = result.expect_err("an aborted parked run must fail");
    assert!(
        matches!(*error, Error::Cancelled),
        "an aborted parked run terminates with Error::Cancelled, got {error:?}"
    );
}

#[tokio::test]
async fn finish_mid_run_prevents_the_next_park() {
    // finish() called while a tool batch is in flight — before any park — must
    // keep the run from parking at its next tool-free reply: it ends there with
    // AgentEnd instead of emitting Idle.
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let provider = ScriptedProvider::new(vec![
        tool_call_script("g", "gate"),
        text_script("done"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Gate {
            started: started.clone(),
            release: release.clone(),
        })
        .build()
        .expect("build");

    let watcher = tokio::spawn(parked_before_end(agent.subscribe()));
    let run = agent.converse("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // Request finish mid-turn, while the gate holds the batch open, then release.
    wait_set(&started).await;
    handle.finish();
    release.store(true, Ordering::SeqCst);

    let reply = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("run must end")
        .expect("run task")
        .expect("run");
    assert_eq!(reply.text_content(), "done");
    assert!(
        !watcher.await.expect("watcher task"),
        "a mid-run finish must keep the run from parking in Idle"
    );
}
