// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the concurrency-classed batch scheduler in the run loop.
//! A turn's batch runs its `Safe` calls concurrently and serializes each
//! `Exclusive` call behind a barrier, and results feed back into the next turn's
//! context in the model's order regardless of which call finished first. Fake
//! tools of each class assert the scheduling; everything runs against an in-crate
//! scripted [`Provider`] — no network.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ToolCtx, UpdateSink};
use tapir::{Agent, AgentEvent};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents, Usage,
};

/// A `Provider` that replays one scripted reply per completion, recording the
/// ordered tool-result call-ids each completion saw fed back into its context so
/// a test can prove the assembled order.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
    result_ids: Arc<Mutex<Vec<Vec<String>>>>,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
            result_ids: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle onto the per-completion ordered result-id log.
    fn result_ids(&self) -> Arc<Mutex<Vec<Vec<String>>>> {
        self.result_ids.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        let ids: Vec<String> = ctx
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult(r) => Some(r.tool_call_id.clone()),
                _ => None,
            })
            .collect();
        self.result_ids.lock().unwrap().push(ids);
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

/// A scripted reply requesting a batch of tool calls, in order.
fn tool_calls_script(
    calls: &[(&str, &str, serde_json::Value)],
) -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::MessageStart];
    for (index, (id, name, args)) in calls.iter().enumerate() {
        events.push(StreamEvent::ToolCallStart {
            index,
            id: (*id).to_string(),
            name: (*name).to_string(),
        });
        events.push(StreamEvent::ToolCallDelta {
            index,
            partial_json: args.to_string(),
        });
        events.push(StreamEvent::ToolCallEnd { index });
    }
    events.push(StreamEvent::Done {
        finish_reason: FinishReason::ToolUse,
        usage: Usage::default(),
    });
    events
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

/// Shared observation of how many calls are concurrently inside a tool's body,
/// tracking the peak so a test can tell parallel from serial execution.
#[derive(Clone, Default)]
struct Overlap {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl Overlap {
    /// Record entry, bumping the observed peak, and return a guard that records
    /// the matching exit on drop.
    fn enter(&self) -> OverlapGuard {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        OverlapGuard {
            active: self.active.clone(),
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// How many calls are inside a tool body right now, including the caller.
    fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }
}

struct OverlapGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for OverlapGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A `Safe` tool that meets every sibling call at a shared barrier: two calls in
/// one batch each stall until both have entered, so the run only completes if
/// they truly ran concurrently, and the observed peak reaches 2.
struct Concurrent {
    barrier: Arc<tokio::sync::Barrier>,
    overlap: Overlap,
}

#[async_trait]
impl tapir::tool::Tool for Concurrent {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "concurrent"
    }
    fn description(&self) -> &str {
        "read that waits at a shared barrier"
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
        let _guard = self.overlap.enter();
        self.barrier.wait().await;
        Ok("ok".to_string())
    }
}

/// An `Exclusive` tool that lingers inside its body: if the scheduler let two
/// calls overlap, the peak would reach 2; behind the barrier it stays 1.
struct Serial {
    overlap: Overlap,
}

#[async_trait]
impl tapir::tool::Tool for Serial {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "serial"
    }
    fn description(&self) -> &str {
        "mutation that lingers in its body"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Exclusive
    }

    async fn execute(
        &self,
        _args: NoArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        let _guard = self.overlap.enter();
        tokio::time::sleep(Duration::from_millis(20)).await;
        Ok("ok".to_string())
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DelayArgs {
    /// How long the call sleeps before returning, in milliseconds.
    ms: u64,
}

/// A `Safe` tool that sleeps for the requested time, so a batch can be made to
/// finish out of model order.
struct Sleeper;

#[async_trait]
impl tapir::tool::Tool for Sleeper {
    type Args = DelayArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "sleeper"
    }
    fn description(&self) -> &str {
        "read that sleeps for the requested time"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        args: DelayArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        tokio::time::sleep(Duration::from_millis(args.ms)).await;
        Ok("done".to_string())
    }
}

#[tokio::test]
async fn safe_calls_run_concurrently() {
    // Two `Safe` calls in one batch. The shared barrier only releases once both
    // are inside, so completing at all proves concurrency; the peak confirms it.
    let overlap = Overlap::default();
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("c1", "concurrent", json!({})),
            ("c2", "concurrent", json!({})),
        ]),
        text_script("done"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Concurrent {
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
            overlap: overlap.clone(),
        })
        .build()
        .expect("build");

    // A timeout is the guard against a serial scheduler deadlocking the barrier.
    let reply =
        tokio::time::timeout(Duration::from_secs(5), agent.prompt("go"))
            .await
            .expect("safe calls must not serialize into a barrier deadlock")
            .expect("run");
    assert_eq!(reply.text_content(), "done");
    assert_eq!(overlap.peak(), 2, "both Safe calls must run concurrently");
}

#[tokio::test]
async fn exclusive_calls_serialize_behind_a_barrier() {
    // Two `Exclusive` calls in one batch: each lingers in its body, so any
    // overlap would push the peak to 2. The barrier keeps it at 1.
    let overlap = Overlap::default();
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("c1", "serial", json!({})),
            ("c2", "serial", json!({})),
        ]),
        text_script("done"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Serial {
            overlap: overlap.clone(),
        })
        .build()
        .expect("build");

    let reply = agent.prompt("go").await.expect("run");
    assert_eq!(reply.text_content(), "done");
    assert_eq!(overlap.peak(), 1, "Exclusive calls must not overlap");
}

#[tokio::test]
async fn results_assembled_in_model_order_despite_completion_order() {
    // The first call sleeps far longer than the second, so the second finishes
    // first. Results must still feed back in model order.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("first", "sleeper", json!({ "ms": 60 })),
            ("second", "sleeper", json!({ "ms": 1 })),
        ]),
        text_script("done"),
    ]);
    let result_ids = provider.result_ids();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Sleeper)
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // The completion order was genuinely reversed: `second` ended before `first`.
    let end_order: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionEnd { result, .. } => {
                Some(result.tool_call_id.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        end_order,
        ["second", "first"],
        "the short call should finish before the long one"
    );

    // Yet the second completion saw the results fed back in model order.
    let seen = result_ids.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1],
        ["first", "second"],
        "results must be assembled in model order, not completion order"
    );
}

/// A `Safe` reader for the mixed batch: pairs up at a shared barrier so two
/// readers in one wave must be simultaneously in flight, pushing the peak to 2.
struct MixReader {
    barrier: Arc<tokio::sync::Barrier>,
    overlap: Overlap,
}

#[async_trait]
impl tapir::tool::Tool for MixReader {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "reader"
    }
    fn description(&self) -> &str {
        "safe reader that pairs at a barrier"
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
        let _guard = self.overlap.enter();
        self.barrier.wait().await;
        Ok("r".to_string())
    }
}

/// An `Exclusive` writer for the mixed batch: records if anything else is ever in
/// flight while it runs, so the test can prove the barrier isolates it from the
/// surrounding `Safe` reads.
struct MixWriter {
    overlap: Overlap,
    saw_overlap: Arc<AtomicBool>,
}

#[async_trait]
impl tapir::tool::Tool for MixWriter {
    type Args = NoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "writer"
    }
    fn description(&self) -> &str {
        "exclusive writer that checks it runs alone"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Exclusive
    }

    async fn execute(
        &self,
        _args: NoArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        let _guard = self.overlap.enter();
        // Only this call should be inside a tool body right now.
        if self.overlap.active() != 1 {
            self.saw_overlap.store(true, Ordering::SeqCst);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        if self.overlap.active() != 1 {
            self.saw_overlap.store(true, Ordering::SeqCst);
        }
        Ok("w".to_string())
    }
}

#[tokio::test]
async fn mixed_batch_interleaves_waves_and_isolates_the_barrier() {
    // A [Safe, Safe, Exclusive, Safe, Safe] batch. The scheduler carves it into
    // three waves: two readers concurrently, the writer alone behind a barrier,
    // then two more readers concurrently. Non-adjacent Safe calls must still run
    // (resume after the barrier), the writer must never overlap a read, and every
    // result must feed back in model order.
    let overlap = Overlap::default();
    let saw_overlap = Arc::new(AtomicBool::new(false));
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("r1", "reader", json!({})),
            ("r2", "reader", json!({})),
            ("w1", "writer", json!({})),
            ("r3", "reader", json!({})),
            ("r4", "reader", json!({})),
        ]),
        text_script("done"),
    ]);
    let result_ids = provider.result_ids();
    let agent = Agent::builder()
        .provider(provider)
        // One reader instance shared by all four reader calls; a Barrier(2)
        // releases once per pair, so each reader wave must pair up to proceed.
        .tool(MixReader {
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
            overlap: overlap.clone(),
        })
        .tool(MixWriter {
            overlap: overlap.clone(),
            saw_overlap: saw_overlap.clone(),
        })
        .build()
        .expect("build");

    // A timeout guards against a reader wave failing to run concurrently and
    // deadlocking on its barrier.
    let reply =
        tokio::time::timeout(Duration::from_secs(5), agent.prompt("go"))
            .await
            .expect("reader waves must run concurrently, not deadlock")
            .expect("run");
    assert_eq!(reply.text_content(), "done");

    assert_eq!(
        overlap.peak(),
        2,
        "each Safe reader wave must run its two calls concurrently"
    );
    assert!(
        !saw_overlap.load(Ordering::SeqCst),
        "the Exclusive writer must run alone, never alongside a read"
    );

    let seen = result_ids.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1],
        ["r1", "r2", "w1", "r3", "r4"],
        "results must feed back in model order across all three waves"
    );
}
