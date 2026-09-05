//! PROTOTYPE - throwaway. Answers wayfinder ticket #6: the `AgentEvent`
//! taxonomy and the `prompt()`/`resume()` streaming API, built on
//! `tapir-provider`'s `StreamEvent` + `StreamAccumulator`.
//!
//! NOT production code. Run:
//!   cargo run --manifest-path prototypes/agentevent/Cargo.toml
//!
//! The verdict lives in the ticket resolution comment; this file is the
//! concrete artifact to react to. It builds against the real `tapir-provider`
//! types and drives a fake streaming provider through a two-turn tool loop so
//! every event variant actually fires.
//!
//! What it decides, made concrete below:
//!   1. AgentEvent: one flat enum, agent/turn/message/tool brackets + error.
//!   2. Delta representation: `MessageUpdate` embeds `StreamEvent` VERBATIM.
//!      The agent folds a `StreamAccumulator` for you; `MessageEnd` hands you
//!      the settled `AssistantMessage`. A TUI reads deltas OR the settled msg.
//!   3. `prompt()`/`resume()` return `Run`: a `Stream<AgentEvent>` whose
//!      terminal item is `AgentEnd { message }`. `Run` is also `IntoFuture`,
//!      so `agent.prompt(..).await?` is the await-the-final-message shortcut.
//!   4. `agent.subscribe()`: a thin session-wide broadcast of every event,
//!      across all runs (for a TUI that renders one agent's whole lifetime).
//!   5. `continue` is a Rust keyword -> the method is `resume()`.

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::json;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Error, ErrorKind,
    FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents, Usage,
};
use tokio::sync::{broadcast, mpsc};

// ===========================================================================
// 1. The event taxonomy - one flat enum
// ===========================================================================

/// A run identifier: one `prompt()`/`resume()` call is one run, spanning N
/// turns. Lets a session-wide subscriber tell concurrent runs apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunId(pub u64);

/// Everything a caller can observe while an agent runs. Single flat enum: a
/// `match` sees the whole vocabulary, and one channel type carries all of it.
///
/// Nesting is by convention, not by type: an `AgentStart` brackets N turns;
/// each `TurnStart..TurnEnd` brackets one message (Start/Update*/End) then its
/// tool batch (ToolExecution Start/Update*/End per call). `turn` on every
/// mid-run event lets a flat consumer re-derive the nesting.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The run began: one prompt/resume call, about to span >=1 turns.
    AgentStart { run: RunId },

    /// A turn began: one provider round-trip plus its tool batch.
    TurnStart { turn: usize },

    /// The assistant message for this turn began streaming.
    MessageStart { turn: usize },

    /// A provider delta for the in-flight message. The delta representation is
    /// tapir-provider's `StreamEvent`, VERBATIM - the agent re-encodes nothing.
    /// A TUI that renders token-by-token reads `TextDelta`/`ThinkingDelta`
    /// here; a consumer that only wants the settled message ignores these and
    /// waits for `MessageEnd`.
    MessageUpdate { turn: usize, delta: StreamEvent },

    /// The assistant message settled. The agent already folded the deltas
    /// through a `StreamAccumulator`, so you never fold them yourself unless
    /// you want to. This message is also appended to the agent's state.
    MessageEnd { turn: usize, message: AssistantMessage },

    /// A tool call began executing.
    ToolExecutionStart {
        turn: usize,
        id: String,
        name: String,
        arguments: serde_json::Value,
    },

    /// Progress from a long-running tool. Forward-compat seam: a tool that
    /// streams (a shell command, a download) reports here; most tools never
    /// emit it and jump straight to `ToolExecutionEnd`.
    ToolExecutionUpdate { turn: usize, id: String, note: String },

    /// A tool call finished. `is_error` mirrors `ToolResultMessage::is_error`.
    ToolExecutionEnd {
        turn: usize,
        id: String,
        output: String,
        is_error: bool,
    },

    /// The turn finished: message settled and any tool batch drained.
    TurnEnd { turn: usize },

    /// The provider stream failed. A first-class EVENT, not a `Result` wrapper
    /// on the stream item, so a session-wide subscriber sees failures too. It
    /// is the last event of a failed run (no `AgentEnd` follows). `Arc` because
    /// `tapir_provider::Error` is not `Clone` (it boxes its source).
    ProviderError { turn: usize, error: Arc<Error> },

    /// The run finished. TERMINAL item; carries the settled final message -
    /// the reply that stopped without asking for a tool.
    AgentEnd { run: RunId, message: AssistantMessage },
}

// ===========================================================================
// 3. The Run handle - a Stream that is also a Future
// ===========================================================================

/// The handle `prompt()`/`resume()` hand back. Two ways to consume it:
///
///   - as a `Stream<Item = AgentEvent>`: `while let Some(ev) = run.next().await`
///   - as a `Future`: `let msg = agent.prompt("hi").await?;` (drains the
///     stream internally, yields the settled final `AssistantMessage`).
///
/// Owns its receiver, so it borrows nothing from the agent: the run drives on
/// a spawned task, and you can drop the `&mut agent` the moment `prompt`
/// returns.
pub struct Run {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl futures::Stream for Run {
    type Item = AgentEvent;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Run holds only a receiver (Unpin), so projecting is a plain get_mut.
        self.get_mut().rx.poll_recv(cx)
    }
}

impl IntoFuture for Run {
    type Output = Result<AssistantMessage, Arc<Error>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    /// The await-the-final-message shortcut: drain events, keep only the
    /// terminal outcome.
    fn into_future(mut self) -> Self::IntoFuture {
        Box::pin(async move {
            while let Some(ev) = self.rx.recv().await {
                match ev {
                    AgentEvent::AgentEnd { message, .. } => return Ok(message),
                    AgentEvent::ProviderError { error, .. } => {
                        return Err(error);
                    }
                    _ => {}
                }
            }
            Err(Arc::new(Error::new(
                ErrorKind::Other,
                "run ended without a terminal event",
            )))
        })
    }
}

// ===========================================================================
// 4. The Agent - stateful, in-memory, with a session-wide broadcast
// ===========================================================================

/// A tool: prototype stand-in for the typed `Tool` trait (ticket #5). Sync,
/// infallible-shaped `Value -> (output, is_error)`.
type ToolFn = Arc<dyn Fn(serde_json::Value) -> (String, bool) + Send + Sync>;

/// A stateful agent that owns its conversation in memory and drives the run
/// loop. State lives behind `Arc<Mutex<..>>` so a spawned run task appends to
/// the same history the agent keeps for the next `prompt`.
pub struct Agent {
    provider: Arc<dyn Provider>,
    system: Option<String>,
    tools: HashMap<String, ToolFn>,
    messages: Arc<Mutex<Vec<Message>>>,
    /// Session-wide event fan-out. Every run's events also flow here, so
    /// `subscribe()` sees the agent's whole lifetime, not just one run.
    events: broadcast::Sender<AgentEvent>,
    next_run: u64,
}

impl Agent {
    pub fn new(provider: Arc<dyn Provider>, system: impl Into<String>) -> Self {
        let (events, _) = broadcast::channel(256);
        Self {
            provider,
            system: Some(system.into()),
            tools: HashMap::new(),
            messages: Arc::new(Mutex::new(Vec::new())),
            events,
            next_run: 0,
        }
    }

    pub fn tool(mut self, name: impl Into<String>, f: ToolFn) -> Self {
        self.tools.insert(name.into(), f);
        self
    }

    /// Append a user turn, then run to completion.
    pub fn prompt(&mut self, input: impl Into<String>) -> Run {
        self.messages.lock().unwrap().push(Message::user(input));
        self.run()
    }

    /// Run again with no new user input - the follow-up / continuation path.
    /// (`continue` is a reserved word, so the method is `resume`.)
    pub fn resume(&mut self) -> Run {
        self.run()
    }

    /// A thin session-wide subscription: every event from every run. Returned
    /// raw here; in real code `tokio_stream::wrappers::BroadcastStream` wraps
    /// it as a `Stream<AgentEvent>`.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    fn run(&mut self) -> Run {
        let run = RunId(self.next_run);
        self.next_run += 1;

        let (tx, rx) = mpsc::unbounded_channel();
        let provider = self.provider.clone();
        let tools = self.tools.clone();
        let messages = self.messages.clone();
        let system = self.system.clone();
        let bcast = self.events.clone();

        tokio::spawn(async move {
            run_loop(run, provider, tools, messages, system, tx, bcast).await;
        });

        Run { rx }
    }
}

/// The run loop: the whole point of the agent. Zero UI, zero persistence -
/// just provider round-trips, tool execution, and events out both channels.
async fn run_loop(
    run: RunId,
    provider: Arc<dyn Provider>,
    tools: HashMap<String, ToolFn>,
    messages: Arc<Mutex<Vec<Message>>>,
    system: Option<String>,
    tx: mpsc::UnboundedSender<AgentEvent>,
    bcast: broadcast::Sender<AgentEvent>,
) {
    // Fan one event out both channels: the per-run stream and the session-wide
    // broadcast. Errors (no receiver) are fine to drop.
    let emit = |ev: AgentEvent| {
        let _ = bcast.send(ev.clone());
        let _ = tx.send(ev);
    };

    emit(AgentEvent::AgentStart { run });

    let mut turn = 0usize;
    let final_message = loop {
        emit(AgentEvent::TurnStart { turn });

        // Snapshot state into a provider Context (one clone per turn: the
        // provider owns its messages - see ticket #4 findings).
        let ctx = {
            let msgs = messages.lock().unwrap().clone();
            let mut ctx = Context::new(msgs);
            if let Some(system) = &system {
                ctx = ctx.with_system(system.clone());
            }
            ctx
        };
        let opts = CompletionOptions::default();

        // Stream the turn, folding deltas into the settled message as they go.
        emit(AgentEvent::MessageStart { turn });
        let mut acc = StreamAccumulator::new();
        let mut events = match provider.complete_stream(&ctx, &opts).await {
            Ok(events) => events,
            Err(error) => {
                emit(AgentEvent::ProviderError {
                    turn,
                    error: Arc::new(error),
                });
                return;
            }
        };
        while let Some(item) = events.next().await {
            match item {
                Ok(delta) => {
                    acc.push(&delta); // agent folds; caller never has to
                    emit(AgentEvent::MessageUpdate { turn, delta });
                }
                Err(error) => {
                    emit(AgentEvent::ProviderError {
                        turn,
                        error: Arc::new(error),
                    });
                    return;
                }
            }
        }
        let message = acc.finish();
        emit(AgentEvent::MessageEnd {
            turn,
            message: message.clone(),
        });
        messages
            .lock()
            .unwrap()
            .push(Message::Assistant(message.clone()));

        // Any tool calls the reply asked for? If none, the run is done.
        let calls: Vec<(String, String, serde_json::Value)> = message
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.clone(), name.clone(), arguments.clone())),
                _ => None,
            })
            .collect();
        if calls.is_empty() {
            emit(AgentEvent::TurnEnd { turn });
            break message;
        }

        // Execute the batch, feeding each result back as a tool-result message.
        for (id, name, arguments) in calls {
            emit(AgentEvent::ToolExecutionStart {
                turn,
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            });
            let (output, is_error) = match tools.get(&name) {
                Some(tool) => tool(arguments),
                None => (format!("no such tool: {name}"), true),
            };
            emit(AgentEvent::ToolExecutionEnd {
                turn,
                id: id.clone(),
                output: output.clone(),
                is_error,
            });
            let mut result = Message::tool_result(id, name, output);
            if is_error {
                if let Message::ToolResult(tr) = &mut result {
                    tr.is_error = true;
                }
            }
            messages.lock().unwrap().push(result);
        }

        emit(AgentEvent::TurnEnd { turn });
        turn += 1;
    };

    emit(AgentEvent::AgentEnd {
        run,
        message: final_message,
    });
}

// ===========================================================================
// A fake streaming provider, so the loop actually runs
// ===========================================================================

/// Canned two-turn provider: turn 1 asks for `get_weather`, turn 2 answers.
/// Stands in for a real `tapir_provider::Provider`; only `complete_stream`
/// matters here.
struct FakeProvider {
    call: AtomicUsize,
}

impl FakeProvider {
    fn new() -> Self {
        Self {
            call: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Provider for FakeProvider {
    async fn complete(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        unimplemented!("prototype streams only")
    }

    async fn complete_stream(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        let events = if self.call.fetch_add(1, Ordering::SeqCst) == 0 {
            // Turn 1: a line of text, then a get_weather tool call.
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextStart { index: 0 },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "Let me check ".into(),
                },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "the weather in Paris.".into(),
                },
                StreamEvent::TextEnd { index: 0 },
                StreamEvent::ToolCallStart {
                    index: 1,
                    id: "call_1".into(),
                    name: "get_weather".into(),
                },
                StreamEvent::ToolCallDelta {
                    index: 1,
                    partial_json: r#"{"city":"Paris"}"#.into(),
                },
                StreamEvent::ToolCallEnd { index: 1 },
                StreamEvent::Done {
                    finish_reason: FinishReason::ToolUse,
                    usage: Usage::default(),
                },
            ]
        } else {
            // Turn 2: the final answer, no tools.
            vec![
                StreamEvent::MessageStart,
                StreamEvent::TextStart { index: 0 },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "It's 21C and sunny ".into(),
                },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "in Paris right now.".into(),
                },
                StreamEvent::TextEnd { index: 0 },
                StreamEvent::Done {
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                },
            ]
        };
        Ok(stream::iter(events.into_iter().map(Ok)).boxed())
    }
}

// ===========================================================================
// 5. The consumer loop a TUI would write
// ===========================================================================

/// Render one run token-by-token, the way a TUI event loop would. Surfaces the
/// full state after each event so you can see exactly what the API delivers.
async fn render_like_a_tui(mut run: Run) {
    println!("\n=== TUI streaming consumer (per-run stream) ===");
    while let Some(event) = run.next().await {
        match event {
            AgentEvent::AgentStart { run } => {
                println!("[run {} started]", run.0);
            }
            AgentEvent::TurnStart { turn } => {
                println!("\n--- turn {turn} ---");
            }
            AgentEvent::MessageStart { .. } => {
                print!("assistant: ");
            }
            // The only branch that touches raw deltas: live text rendering.
            AgentEvent::MessageUpdate { delta, .. } => match delta {
                StreamEvent::TextDelta { text, .. } => {
                    print!("{text}");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
                StreamEvent::ToolCallStart { name, .. } => {
                    print!("\n  (calling {name}...) ");
                }
                _ => {}
            },
            AgentEvent::MessageEnd { message, .. } => {
                // The settled message, folded for us. No accumulator needed.
                println!(
                    "\n  [settled: {} content part(s), finish={:?}]",
                    message.content.len(),
                    message.finish_reason
                );
            }
            AgentEvent::ToolExecutionStart { name, arguments, .. } => {
                println!("  tool {name} <- {arguments}");
            }
            AgentEvent::ToolExecutionUpdate { note, .. } => {
                println!("  ... {note}");
            }
            AgentEvent::ToolExecutionEnd { output, is_error, .. } => {
                println!("  tool -> {output} (error={is_error})");
            }
            AgentEvent::TurnEnd { turn } => {
                println!("--- turn {turn} end ---");
            }
            AgentEvent::ProviderError { error, .. } => {
                println!("  !! provider error: {}", error.message());
            }
            AgentEvent::AgentEnd { message, .. } => {
                println!("\n[run done] final: {:?}", message.text_content());
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let mut agent = Agent::new(Arc::new(FakeProvider::new()), "You are terse.")
        .tool(
            "get_weather",
            Arc::new(|args: serde_json::Value| {
                let city = args
                    .get("city")
                    .and_then(|c| c.as_str())
                    .unwrap_or("?");
                (json!({ "tempC": 21, "sky": "sunny", "city": city }).to_string(), false)
            }),
        );

    // --- Demo A: the full streaming consumer (a TUI) ---
    let run = agent.prompt("What's the weather in Paris?");
    render_like_a_tui(run).await;

    // --- Demo B: a session-wide subscriber watching the NEXT run ---
    println!("\n=== session-wide subscription (agent.subscribe) ===");
    let mut sub = agent.subscribe();
    let watcher = tokio::spawn(async move {
        while let Ok(event) = sub.recv().await {
            // A dashboard cares about brackets, not tokens.
            match event {
                AgentEvent::AgentStart { run } => {
                    println!("[sub] run {} started", run.0)
                }
                AgentEvent::ToolExecutionStart { name, .. } => {
                    println!("[sub] tool: {name}")
                }
                AgentEvent::AgentEnd { run, .. } => {
                    println!("[sub] run {} done", run.0);
                    break;
                }
                _ => {}
            }
        }
    });

    // --- Demo C: the await-the-final-message shortcut ---
    // resume()/prompt() return a Run; `.await` drains it to the final message.
    let reply = agent
        .prompt("And tomorrow?")
        .await
        .expect("run should settle");
    println!(
        "\n=== await shortcut (Run as Future) ===\nfinal reply: {:?}",
        reply.text_content()
    );

    let _ = watcher.await;

    // State check: the agent kept the whole conversation in memory.
    let kept = agent.messages.lock().unwrap().len();
    println!("\n[state] agent retained {kept} messages across both runs");
}
