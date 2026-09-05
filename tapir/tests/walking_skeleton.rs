// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! End-to-end tests for the walking skeleton: an [`Agent`] over an in-crate
//! fake [`Provider`] (scripted `StreamEvent`s) returns a reply, both as an
//! awaited future and as an event stream, and terminates with
//! `Error::MaxIterations` when a tool-requesting reply runs past the cap.

use async_trait::async_trait;
use futures_util::StreamExt;
use tapir::agent::RunId;
use tapir::{Agent, AgentEvent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Provider, StreamAccumulator, StreamEvent, StreamEvents,
    Usage,
};

/// A `Provider` that replays a fixed script of `StreamEvent`s on every
/// completion. No network, fully deterministic.
struct FakeProvider {
    events: Vec<StreamEvent>,
}

impl FakeProvider {
    fn new(events: Vec<StreamEvent>) -> Self {
        Self { events }
    }
}

#[async_trait]
impl Provider for FakeProvider {
    async fn complete(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Ok(StreamAccumulator::fold(&self.events))
    }

    async fn complete_stream(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        let events = self.events.clone();
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

/// A scripted reply that requests a tool call (so the settled message reports
/// `tool_calls`). Replayed every turn, it drives the loop to the cap.
fn tool_script() -> Vec<StreamEvent> {
    vec![
        StreamEvent::MessageStart,
        StreamEvent::ToolCallStart {
            index: 0,
            id: "call_1".to_string(),
            name: "noop".to_string(),
        },
        StreamEvent::ToolCallDelta {
            index: 0,
            partial_json: "{}".to_string(),
        },
        StreamEvent::ToolCallEnd { index: 0 },
        StreamEvent::Done {
            finish_reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ]
}

/// The variant name of an event, for order assertions.
fn name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart { .. } => "AgentStart",
        AgentEvent::TurnStart { .. } => "TurnStart",
        AgentEvent::MessageStart { .. } => "MessageStart",
        AgentEvent::MessageUpdate { .. } => "MessageUpdate",
        AgentEvent::MessageEnd { .. } => "MessageEnd",
        AgentEvent::TurnEnd { .. } => "TurnEnd",
        AgentEvent::Error { .. } => "Error",
        AgentEvent::AgentEnd { .. } => "AgentEnd",
        _ => "Unknown",
    }
}

#[tokio::test]
async fn prompt_await_returns_reply() {
    let agent = Agent::builder()
        .provider(FakeProvider::new(text_script("Hello, world!")))
        .system("You are helpful.")
        .build()
        .expect("build");

    let reply = agent.prompt("Hi").await.expect("run");
    assert_eq!(reply.text_content(), "Hello, world!");
}

#[tokio::test]
async fn run_stream_emits_core_events_in_order() {
    let agent = Agent::builder()
        .provider(FakeProvider::new(text_script("Hi there")))
        .build()
        .expect("build");

    let events: Vec<AgentEvent> = agent.prompt("Hello").collect().await;

    // The terminal item settles the reply.
    match events.last() {
        Some(AgentEvent::AgentEnd { message, .. }) => {
            assert_eq!(message.text_content(), "Hi there");
        }
        other => panic!("expected AgentEnd terminal, got {other:?}"),
    }

    // At least one streamed delta arrived.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::MessageUpdate { .. }))
    );

    // The core (non-delta) events arrive in the skeleton order.
    let core: Vec<&str> = events
        .iter()
        .map(name)
        .filter(|n| *n != "MessageUpdate")
        .collect();
    assert_eq!(
        core,
        [
            "AgentStart",
            "TurnStart",
            "MessageStart",
            "MessageEnd",
            "TurnEnd",
            "AgentEnd",
        ]
    );
}

#[tokio::test]
async fn max_iterations_terminates_with_error() {
    let agent = Agent::builder()
        .provider(FakeProvider::new(tool_script()))
        .max_tool_iterations(2)
        .build()
        .expect("build");

    // Await-final shortcut maps the terminal error event to `Err`.
    let err = agent.prompt("go").await.expect_err("should exceed cap");
    assert!(matches!(*err, Error::MaxIterations { limit: 2 }));

    // The same failure rides the event stream as a terminal `Error`, with no
    // `AgentEnd`.
    let events: Vec<AgentEvent> = agent.prompt("go again").collect().await;
    match events.last() {
        Some(AgentEvent::Error { error, turn }) => {
            assert!(matches!(**error, Error::MaxIterations { limit: 2 }));
            assert_eq!(*turn, Some(2));
        }
        other => panic!("expected terminal Error, got {other:?}"),
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. }))
    );
}

#[tokio::test]
async fn subscribe_distinguishes_concurrent_runs() {
    let agent = Agent::builder()
        .provider(FakeProvider::new(text_script("ok")))
        .build()
        .expect("build");

    let mut sub = agent.subscribe();

    // Two runs kicked off before either is awaited; RunIds are assigned
    // synchronously at `prompt`.
    let r0 = agent.prompt("a");
    let r1 = agent.prompt("b");
    r0.await.expect("run 0");
    r1.await.expect("run 1");

    // The session-wide subscription observed both runs, told apart by RunId.
    let mut starts = std::collections::HashSet::new();
    while let Ok(event) = sub.try_recv() {
        if let AgentEvent::AgentStart { run } = event {
            starts.insert(run);
        }
    }
    assert_eq!(starts.len(), 2);
    assert!(starts.contains(&RunId(0)));
    assert!(starts.contains(&RunId(1)));
}
