// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the tool-call approval gate (`before_tool_call`) and the
//! post-batch observer (`after_tool_call`): the gate sees every call in model
//! order before the batch runs; `Proceed` runs it, `Modify` reruns validation
//! with rewritten args, and `Deny` skips execution while feeding the model a
//! synthetic `is_error` result plus a `ToolCallDenied` event; the observer sees
//! executed results only; and an abort drops a pending gate future. Everything
//! runs against an in-crate scripted [`Provider`] — no network.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ToolCall, ToolCtx, ToolDecision, UpdateSink};
use tapir::{Agent, AgentEvent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents, Usage,
};

/// A `Provider` that replays one scripted reply per completion, advancing a
/// cursor and clamping at the last script. It records how many tool results each
/// completion saw fed back into its context, so a test can prove a synthetic
/// denied result reached the model.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
    tool_results: Arc<Mutex<Vec<usize>>>,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
            tool_results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle onto the per-completion count of fed-back tool results.
    fn tool_results(&self) -> Arc<Mutex<Vec<usize>>> {
        self.tool_results.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        let count = ctx
            .messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult(_)))
            .count();
        self.tool_results.lock().unwrap().push(count);
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
struct EchoArgs {
    /// The value to echo back.
    value: String,
}

/// A `Safe` tool that records every `value` it was invoked with and echoes it,
/// so a test can prove which calls ran and with what (possibly rewritten) args.
struct Echo {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl tapir::tool::Tool for Echo {
    type Args = EchoArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echoes the given value"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        args: EchoArgs,
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        self.seen.lock().unwrap().push(args.value.clone());
        Ok(format!("echo: {}", args.value))
    }
}

/// Spin until `flag` is set, so a test only acts once the gate is genuinely in
/// flight. Callers wrap it in a timeout.
async fn wait_set(flag: &AtomicBool) {
    while !flag.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

#[tokio::test]
async fn gate_sees_every_call_in_model_order_then_proceeds() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("c1", "echo", json!({ "value": "a" })),
            ("c2", "echo", json!({ "value": "b" })),
        ]),
        text_script("done"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let gate_log = Arc::new(Mutex::new(Vec::new()));
    let gl = gate_log.clone();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |call: &ToolCall| {
            let gl = gl.clone();
            let id = call.id().to_string();
            Box::pin(async move {
                gl.lock().unwrap().push(id);
                ToolDecision::Proceed
            })
        })
        .build()
        .expect("build");

    let reply = agent.prompt("go").await.expect("run");
    assert_eq!(reply.text_content(), "done");

    // The gate saw both calls, in model order, before the batch ran.
    assert_eq!(*gate_log.lock().unwrap(), ["c1", "c2"]);
    // Both proceeded and executed with their original args.
    assert_eq!(*seen.lock().unwrap(), ["a", "b"]);
}

#[tokio::test]
async fn deny_skips_execution_and_feeds_synthetic_error() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("c1", "echo", json!({ "value": "x" }))]),
        text_script("ok"),
    ]);
    let tool_results = provider.tool_results();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |_call: &ToolCall| {
            Box::pin(async move {
                ToolDecision::Deny {
                    message: "not allowed".to_string(),
                }
            })
        })
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // The denied call never executed.
    assert!(seen.lock().unwrap().is_empty(), "denied call must not run");

    // A ToolCallDenied event fired with the call id and message; no
    // ToolExecutionStart was emitted for it.
    let denied = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolCallDenied { id, message, turn } => {
                Some((id.clone(), message.clone(), *turn))
            }
            _ => None,
        })
        .expect("a ToolCallDenied event");
    assert_eq!(denied.0, "c1");
    assert_eq!(denied.1, "not allowed");
    assert_eq!(denied.2, 0);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionStart { .. })),
        "a denied call emits no ToolExecutionStart"
    );

    // The synthetic is_error result reached the model on the next turn.
    let counts = tool_results.lock().unwrap().clone();
    assert_eq!(counts.len(), 2);
    assert_eq!(counts[1], 1, "the denied result feeds back to the model");

    // The run recovered to a tool-free reply.
    match events.last() {
        Some(AgentEvent::AgentEnd { message, .. }) => {
            assert_eq!(message.text_content(), "ok");
        }
        other => panic!("expected AgentEnd, got {other:?}"),
    }
}

#[tokio::test]
async fn modify_reruns_validation_with_rewritten_args() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("c1", "echo", json!({ "value": "original" }))]),
        text_script("done"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |_call: &ToolCall| {
            Box::pin(async move {
                ToolDecision::Modify {
                    arguments: json!({ "value": "rewritten" }),
                }
            })
        })
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // The tool ran with the rewritten arguments, not the model's original.
    assert_eq!(*seen.lock().unwrap(), ["rewritten"]);

    // The rewritten call flowed through the normal ToolExecution* events.
    let end = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolExecutionEnd { result, .. } => Some(result),
            _ => None,
        })
        .expect("a tool execution ended");
    assert!(!end.is_error);
    assert_eq!(
        end.content,
        vec![tapir_provider::ContentPart::text("echo: rewritten")]
    );
}

#[tokio::test]
async fn modify_with_invalid_args_rides_back_as_is_error() {
    // A Modify that installs arguments failing validation reruns validation at
    // the dispatch boundary and rides back as an is_error result — not a Deny.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("c1", "echo", json!({ "value": "ok" }))]),
        text_script("recovered"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |_call: &ToolCall| {
            Box::pin(async move {
                ToolDecision::Modify {
                    arguments: json!({ "wrong_field": 1 }),
                }
            })
        })
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // Validation rejected the rewritten args, so the tool body never ran.
    assert!(seen.lock().unwrap().is_empty());
    let end = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolExecutionEnd { result, .. } => Some(result),
            _ => None,
        })
        .expect("a tool execution ended");
    assert!(
        end.is_error,
        "invalid rewritten args are an is_error result"
    );
}

#[tokio::test]
async fn after_tool_call_observes_executed_results_only() {
    // Two calls: c1 proceeds, c2 is denied. The observer sees only c1.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("c1", "echo", json!({ "value": "a" })),
            ("c2", "echo", json!({ "value": "b" })),
        ]),
        text_script("done"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::new(Mutex::new(Vec::new()));
    let ob = observed.clone();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |call: &ToolCall| {
            let deny = call.id() == "c2";
            Box::pin(async move {
                if deny {
                    ToolDecision::Deny {
                        message: "no".to_string(),
                    }
                } else {
                    ToolDecision::Proceed
                }
            })
        })
        .after_tool_call(move |call: &ToolCall, result| {
            let ob = ob.clone();
            let id = call.id().to_string();
            let is_error = result.is_error;
            Box::pin(async move {
                ob.lock().unwrap().push((id, is_error));
            })
        })
        .build()
        .expect("build");

    agent.prompt("go").await.expect("run");

    // Only the executed call was observed — never the gate-denied one.
    assert_eq!(*observed.lock().unwrap(), [("c1".to_string(), false)]);
}

#[tokio::test]
async fn abort_drops_a_pending_gate_future() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("c1", "echo", json!({ "value": "a" }))]),
        text_script("unreachable"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(AtomicBool::new(false));
    let s = started.clone();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .before_tool_call(move |_call: &ToolCall| {
            s.store(true, Ordering::SeqCst);
            Box::pin(
                async move { std::future::pending::<ToolDecision>().await },
            )
        })
        .build()
        .expect("build");

    let run = agent.prompt("go");
    let handle = run.handle();
    let task = tokio::spawn(async move { run.await });

    // Abort only once the gate future is genuinely pending.
    tokio::time::timeout(Duration::from_secs(5), wait_set(&started))
        .await
        .expect("gate must start");
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
        seen.lock().unwrap().is_empty(),
        "the gated call must never execute"
    );
}

#[tokio::test]
async fn hooks_default_to_noop_when_unset() {
    // With neither hook installed, a call runs normally — proof the defaults are
    // no-ops.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("c1", "echo", json!({ "value": "a" }))]),
        text_script("done"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo { seen: seen.clone() })
        .build()
        .expect("build");

    let reply = agent.prompt("go").await.expect("run");
    assert_eq!(reply.text_content(), "done");
    assert_eq!(*seen.lock().unwrap(), ["a"]);
}
