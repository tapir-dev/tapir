// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for sequential tool execution in the run loop: a registered
//! tool is invoked when the model asks for it, its result feeds back, and the
//! run continues to a tool-free reply. Also covers the `ToolExecution*` event
//! order, a bad-arg failure riding back as `ToolResult { is_error }`, and an
//! unknown-tool call yielding a synthetic result in model order. Everything runs
//! against an in-crate scripted [`Provider`] — no network.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ToolCtx, ToolUpdate, UpdateSink};
use tapir::{Agent, AgentEvent};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents, Usage,
};

/// What a scripted completion observed about the context it was handed, so a
/// test can assert the fed-back tool results and the offered tool set.
#[derive(Debug, Clone)]
struct Seen {
    tool_results: usize,
    tools_offered: usize,
}

/// A `Provider` that replays one scripted reply per completion, advancing a
/// cursor and clamping at the last script. It records what each call saw so a
/// test can prove a tool result fed back into the next turn's context.
struct ScriptedProvider {
    scripts: Vec<Vec<StreamEvent>>,
    cursor: AtomicUsize,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl ScriptedProvider {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            scripts,
            cursor: AtomicUsize::new(0),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// A handle onto the per-call observation log, cloned before the provider is
    /// moved into the builder.
    fn seen(&self) -> Arc<Mutex<Vec<Seen>>> {
        self.seen.clone()
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        let tool_results = ctx
            .messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult(_)))
            .count();
        self.seen.lock().unwrap().push(Seen {
            tool_results,
            tools_offered: ctx.tools.len(),
        });
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
struct WeatherArgs {
    /// City name or coordinates.
    city: String,
}

/// The worked example: a read-only weather tool that streams one progress
/// update and returns a fixed report. Hand-written (not `#[tool]`) so it can
/// exercise the `on_update` sink the run loop forwards as `ToolExecutionUpdate`.
struct Weather;

#[async_trait]
impl tapir::tool::Tool for Weather {
    type Args = WeatherArgs;
    type Output = String;
    type Error = String;

    fn name(&self) -> &str {
        "weather"
    }
    fn description(&self) -> &str {
        "Get current weather for a city"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        args: WeatherArgs,
        _ctx: &ToolCtx,
        on_update: &mut UpdateSink<'_>,
    ) -> Result<String, String> {
        on_update(ToolUpdate::Progress(format!("looking up {}", args.city)));
        Ok(format!("It is 18C and clear in {}.", args.city))
    }
}

/// The variant name of an event, for order assertions.
fn name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart { .. } => "AgentStart",
        AgentEvent::TurnStart { .. } => "TurnStart",
        AgentEvent::MessageStart { .. } => "MessageStart",
        AgentEvent::MessageUpdate { .. } => "MessageUpdate",
        AgentEvent::MessageEnd { .. } => "MessageEnd",
        AgentEvent::ToolExecutionStart { .. } => "ToolExecutionStart",
        AgentEvent::ToolExecutionUpdate { .. } => "ToolExecutionUpdate",
        AgentEvent::ToolExecutionEnd { .. } => "ToolExecutionEnd",
        AgentEvent::TurnEnd { .. } => "TurnEnd",
        AgentEvent::Error { .. } => "Error",
        AgentEvent::AgentEnd { .. } => "AgentEnd",
        _ => "Unknown",
    }
}

#[tokio::test]
async fn registered_tool_runs_and_result_feeds_back() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("call_1", "weather", json!({"city": "Paris"}))]),
        text_script("The weather in Paris is clear."),
    ]);
    let seen = provider.seen();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Weather)
        .build()
        .expect("build");

    let reply = agent.prompt("Weather in Paris?").await.expect("run");
    assert_eq!(reply.text_content(), "The weather in Paris is clear.");

    // Two completions ran: the first offered the tool and saw no results; the
    // second saw the one tool result fed back into its context.
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].tools_offered, 1);
    assert_eq!(seen[0].tool_results, 0);
    assert_eq!(seen[1].tool_results, 1);
}

#[tokio::test]
async fn tool_execution_events_in_order() {
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("call_1", "weather", json!({"city": "Paris"}))]),
        text_script("done"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Weather)
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // The tool-requesting turn (turn 0) brackets the call after its MessageEnd.
    let core: Vec<&str> = events
        .iter()
        .map(name)
        .filter(|n| *n != "MessageUpdate")
        .collect();
    assert_eq!(
        core,
        [
            "AgentStart",
            "TurnStart", // turn 0: the tool-requesting reply
            "MessageStart",
            "MessageEnd",
            "ToolExecutionStart",
            "ToolExecutionUpdate",
            "ToolExecutionEnd",
            "TurnEnd",
            "TurnStart", // turn 1: the tool-free reply
            "MessageStart",
            "MessageEnd",
            "TurnEnd",
            "AgentEnd",
        ]
    );

    // Every tool-execution event carries turn 0, and the end result is the
    // successful weather report.
    for event in &events {
        match event {
            AgentEvent::ToolExecutionStart {
                turn,
                call_id,
                name,
            } => {
                assert_eq!(*turn, 0);
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "weather");
            }
            AgentEvent::ToolExecutionUpdate { turn, update, .. } => {
                assert_eq!(*turn, 0);
                assert_eq!(
                    *update,
                    ToolUpdate::Progress("looking up Paris".to_string())
                );
            }
            AgentEvent::ToolExecutionEnd { turn, result } => {
                assert_eq!(*turn, 0);
                assert!(!result.is_error);
                assert_eq!(
                    result.content,
                    vec![tapir_provider::ContentPart::text(
                        "It is 18C and clear in Paris."
                    )]
                );
            }
            _ => {}
        }
    }
}

#[tokio::test]
async fn bad_args_ride_back_as_is_error_and_reprompt() {
    // First reply calls the tool with a bad arg (missing required `city`); the
    // dispatch boundary mints an `is_error` result and the run re-prompts to a
    // tool-free reply within the cap.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[("call_1", "weather", json!({"town": "Paris"}))]),
        text_script("sorry, retried"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Weather)
        .max_tool_iterations(5)
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    // The run recovered to a tool-free reply — no terminal Error.
    match events.last() {
        Some(AgentEvent::AgentEnd { message, .. }) => {
            assert_eq!(message.text_content(), "sorry, retried");
        }
        other => panic!("expected AgentEnd, got {other:?}"),
    }

    let end = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::ToolExecutionEnd { result, .. } => Some(result),
            _ => None,
        })
        .expect("a tool execution ended");
    assert!(end.is_error);
    assert_eq!(end.tool_call_id, "call_1");
    let text = match &end.content[0] {
        tapir_provider::ContentPart::Text(t) => t.as_str(),
        other => panic!("expected text content, got {other:?}"),
    };
    assert!(
        text.contains("invalid arguments for `weather`"),
        "unexpected error text: {text}"
    );
}

#[tokio::test]
async fn unknown_tool_yields_synthetic_result_in_model_order() {
    // A batch of two calls: a known tool then an unknown one. Both results feed
    // back, in the model's order, the unknown one synthetic and flagged.
    let provider = ScriptedProvider::new(vec![
        tool_calls_script(&[
            ("call_1", "weather", json!({"city": "Paris"})),
            ("call_2", "nonesuch", json!({})),
        ]),
        text_script("all set"),
    ]);
    let agent = Agent::builder()
        .provider(provider)
        .tool(Weather)
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("go").collect().await;

    let ends: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolExecutionEnd { result, .. } => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(ends.len(), 2, "both calls in the batch ran");

    // Model order preserved: known tool first, unknown second.
    assert_eq!(ends[0].tool_call_id, "call_1");
    assert!(!ends[0].is_error);

    assert_eq!(ends[1].tool_call_id, "call_2");
    assert_eq!(ends[1].tool_name, "nonesuch");
    assert!(ends[1].is_error);
    assert_eq!(
        ends[1].content,
        vec![tapir_provider::ContentPart::text("unknown tool `nonesuch`")]
    );
}
