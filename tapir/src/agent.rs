// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The stateful agent and its run surface. An [`Agent`] owns the in-memory
//! conversation history and drives runs. Each run is a hybrid kernel: an async
//! driver (`run_loop`) owns the IO, while the pure `step` function decides the
//! turn-to-turn transition and is unit-testable with no async or IO. This
//! ticket ships the tool-free spine — the model just replies; tool execution,
//! cancellation, and steering land in later tickets.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use futures_core::Stream;
use futures_util::StreamExt;
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Message, Provider,
    StreamAccumulator,
};
use tokio::sync::{broadcast, mpsc};

use crate::error::Error;
use crate::event::AgentEvent;
use crate::message::{
    AgentMessage, CustomMessage, NoCustom, TransformContext, convert_to_llm,
};

/// Default cap on tool-requesting turns before a run fails with
/// [`Error::MaxIterations`].
const DEFAULT_MAX_TOOL_ITERATIONS: usize = 25;

/// Capacity of the session-wide broadcast channel backing [`Agent::subscribe`].
const BROADCAST_CAPACITY: usize = 256;

/// Disambiguates concurrent runs on the session-wide broadcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub u64);

/// How a steer injection combines with the pending user input.
#[non_exhaustive]
pub enum SteerMode {
    /// Append the steer text as a new user turn.
    Append,
    /// Replace the pending user input with the steer text.
    Replace,
}

/// The cloneable control surface (abort / steer / finish) that outlives the
/// [`Run`]. Its methods land with the cancellation and steering tickets.
pub struct RunHandle;

/// The loop state the pure kernel folds over: how many tool-requesting turns
/// have run, and the configured cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoopState {
    /// Turns so far whose reply requested tools.
    tool_iterations: usize,
    /// The configured cap on tool-requesting turns.
    max_tool_iterations: usize,
}

/// The turn-transition decision returned by [`step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Next {
    /// The reply asked for tools and the cap allows another turn.
    Continue,
    /// The reply answered without tools: finish the run with it.
    Finish,
    /// The reply asked for tools but the cap is spent.
    MaxIterations {
        /// The configured limit that was exceeded.
        limit: usize,
    },
}

/// The pure turn-transition kernel. Given the current loop state and whether the
/// just-settled reply asked for tools, decide what the driver does next. A
/// tool-free reply finishes and never counts against the cap; a tool-requesting
/// reply counts as one iteration, and the cap is checked *after* counting it, so
/// a limit of `n` admits exactly `n` tool-requesting turns.
fn step(state: &LoopState, reply_wants_tools: bool) -> Next {
    if !reply_wants_tools {
        return Next::Finish;
    }
    let used = state.tool_iterations + 1;
    if used > state.max_tool_iterations {
        return Next::MaxIterations {
            limit: state.max_tool_iterations,
        };
    }
    Next::Continue
}

/// The fluent builder for an [`Agent`]. Build with [`Agent::builder`].
pub struct AgentBuilder<M = NoCustom> {
    provider: Option<Arc<dyn Provider>>,
    system: Option<String>,
    max_tool_iterations: usize,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M> Default for AgentBuilder<M> {
    fn default() -> Self {
        Self {
            provider: None,
            system: None,
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M> AgentBuilder<M> {
    /// Start a fresh builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider — the canonical seam. The model rides on the provider
    /// instance, so it is chosen when the provider is constructed.
    #[must_use]
    pub fn provider<P: Provider + 'static>(mut self, provider: P) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the cap on tool-requesting turns (default 25). Overflow fails the
    /// run with [`Error::MaxIterations`].
    #[must_use]
    pub fn max_tool_iterations(mut self, n: usize) -> Self {
        self.max_tool_iterations = n;
        self
    }

    /// Validate the configuration and build the agent. Synchronous: the only
    /// v1 check is that a provider was set. A missing provider is
    /// [`Error::Build`].
    pub fn build(self) -> Result<Agent<M>, Error> {
        let provider = self
            .provider
            .ok_or_else(|| Error::Build("provider is required".to_string()))?;
        let (events, _) = broadcast::channel(BROADCAST_CAPACITY);
        Ok(Agent {
            provider,
            system: self.system,
            max_tool_iterations: self.max_tool_iterations,
            transform: None,
            messages: Arc::new(Mutex::new(Vec::new())),
            events,
            next_run: AtomicU64::new(0),
            _marker: std::marker::PhantomData,
        })
    }
}

/// The stateful object owning the in-memory conversation history and driving
/// runs. Generic over a custom-message type `M`, defaulting to [`NoCustom`].
pub struct Agent<M = NoCustom> {
    provider: Arc<dyn Provider>,
    system: Option<String>,
    max_tool_iterations: usize,
    /// The `transform_context` seam applied to history each turn before
    /// `convert_to_llm`. `None` is the identity (no reshaping); the builder
    /// that installs one lands in a later ticket.
    transform: Option<Arc<TransformContext<M>>>,
    /// History owned behind a mutex so a spawned run appends to the same
    /// history the next `prompt` reads.
    messages: Arc<Mutex<Vec<AgentMessage<M>>>>,
    events: broadcast::Sender<AgentEvent>,
    next_run: AtomicU64,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl Agent<NoCustom> {
    /// Zero-ceremony entry for the common (no custom messages) case.
    #[must_use]
    pub fn builder() -> AgentBuilder<NoCustom> {
        AgentBuilder::new()
    }
}

impl<M: CustomMessage + Send + 'static> Agent<M> {
    /// Append a user turn, then run to completion.
    pub fn prompt(&self, input: impl Into<String>) -> Run {
        self.messages
            .lock()
            .expect("history mutex poisoned")
            .push(AgentMessage::Llm(Message::user(input)));
        self.run()
    }

    /// Subscribe to this session's events across all runs. Each run's events are
    /// tagged with a [`RunId`] so concurrent runs are distinguishable.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Spawn the driver on its own task and hand back the [`Run`].
    fn run(&self) -> Run {
        let run = RunId(self.next_run.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(run_loop(RunLoop {
            run,
            provider: self.provider.clone(),
            system: self.system.clone(),
            max_tool_iterations: self.max_tool_iterations,
            transform: self.transform.clone(),
            messages: self.messages.clone(),
            tx,
            bcast: self.events.clone(),
        }));
        Run { rx }
    }
}

/// One invocation of the agent: both a [`Stream`] of [`AgentEvent`] (terminating
/// with [`AgentEvent::AgentEnd`]) and, via [`IntoFuture`], a future resolving to
/// the settled reply.
pub struct Run {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl Stream for Run {
    type Item = AgentEvent;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.get_mut().rx.poll_recv(cx)
    }
}

impl IntoFuture for Run {
    type Output = Result<AssistantMessage, Arc<Error>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(mut self) -> Self::IntoFuture {
        Box::pin(async move {
            while let Some(ev) = self.rx.recv().await {
                match ev {
                    AgentEvent::AgentEnd { message, .. } => return Ok(message),
                    AgentEvent::Error { error, .. } => return Err(error),
                    _ => {}
                }
            }
            // The driver always emits a terminal event, so this is unreachable
            // in practice; keep the type honest if the task vanishes.
            Err(Arc::new(Error::Provider(tapir_provider::Error::new(
                tapir_provider::ErrorKind::Other,
                "run ended without a terminal event",
            ))))
        })
    }
}

/// Everything one run's driver needs, assembled by [`Agent::run`] and moved onto
/// the spawned task.
struct RunLoop<M> {
    run: RunId,
    provider: Arc<dyn Provider>,
    system: Option<String>,
    max_tool_iterations: usize,
    transform: Option<Arc<TransformContext<M>>>,
    messages: Arc<Mutex<Vec<AgentMessage<M>>>>,
    tx: mpsc::UnboundedSender<AgentEvent>,
    bcast: broadcast::Sender<AgentEvent>,
}

/// The async driver for one run. Owns the IO and event fan-out; defers every
/// turn-transition decision to the pure `step` kernel. One turn is one
/// `complete_stream` folded through a [`StreamAccumulator`]; the provider
/// context is built each turn from `transform` then [`convert_to_llm`] under
/// one lock.
async fn run_loop<M: CustomMessage + Send + 'static>(driver: RunLoop<M>) {
    let RunLoop {
        run,
        provider,
        system,
        max_tool_iterations,
        transform,
        messages,
        tx,
        bcast,
    } = driver;
    // Fan each event out both the per-run stream and the session broadcast.
    let emit = |ev: AgentEvent| {
        let _ = bcast.send(ev.clone());
        let _ = tx.send(ev);
    };

    emit(AgentEvent::AgentStart { run });

    let mut state = LoopState {
        tool_iterations: 0,
        max_tool_iterations,
    };
    let mut turn = 0usize;

    let outcome: Result<AssistantMessage, Error> = loop {
        emit(AgentEvent::TurnStart { turn });

        // Build the provider context under one lock, then drop the guard before
        // any await so the driver future stays `Send`. `transform_context`
        // reshapes history first (identity when unset), then `convert_to_llm`
        // flattens it to provider messages.
        let ctx = {
            let guard = messages.lock().expect("history mutex poisoned");
            let llm_messages = match transform.as_deref() {
                Some(f) => convert_to_llm(&f(guard.as_slice())),
                None => convert_to_llm(guard.as_slice()),
            };
            let mut ctx = Context::new(llm_messages);
            if let Some(system) = &system {
                ctx = ctx.with_system(system.clone());
            }
            ctx
        };
        let opts = CompletionOptions::default();

        emit(AgentEvent::MessageStart { turn });
        let mut acc = StreamAccumulator::new();
        let mut stream = match provider.complete_stream(&ctx, &opts).await {
            Ok(stream) => stream,
            Err(error) => break Err(Error::Provider(error)),
        };

        let mut stream_error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(delta) => {
                    acc.push(&delta);
                    emit(AgentEvent::MessageUpdate { turn, delta });
                }
                Err(error) => {
                    // Discard the partial message: it is never committed to
                    // history, so a later run replays from the last settled
                    // message.
                    stream_error = Some(Error::Provider(error));
                    break;
                }
            }
        }
        if let Some(error) = stream_error {
            break Err(error);
        }

        let message = acc.finish();
        emit(AgentEvent::MessageEnd {
            turn,
            message: message.clone(),
        });
        messages
            .lock()
            .expect("history mutex poisoned")
            .push(AgentMessage::Llm(Message::Assistant(message.clone())));

        let reply_wants_tools = message.tool_calls().next().is_some();
        match step(&state, reply_wants_tools) {
            Next::Finish => {
                emit(AgentEvent::TurnEnd { turn });
                break Ok(message);
            }
            Next::MaxIterations { limit } => {
                break Err(Error::MaxIterations { limit });
            }
            Next::Continue => {
                state.tool_iterations += 1;
            }
        }

        // Tool execution lands in a later ticket. With no tools to run, a
        // tool-requesting reply simply advances to the next turn.
        emit(AgentEvent::TurnEnd { turn });
        turn += 1;
    };

    match outcome {
        Ok(message) => emit(AgentEvent::AgentEnd { run, message }),
        Err(error) => emit(AgentEvent::Error {
            turn: Some(turn),
            error: Arc::new(error),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(tool_iterations: usize, max: usize) -> LoopState {
        LoopState {
            tool_iterations,
            max_tool_iterations: max,
        }
    }

    #[test]
    fn tool_free_reply_finishes() {
        assert_eq!(step(&state(0, 25), false), Next::Finish);
        // A tool-free reply finishes regardless of how many iterations ran.
        assert_eq!(step(&state(25, 25), false), Next::Finish);
    }

    #[test]
    fn tool_reply_under_cap_continues() {
        assert_eq!(step(&state(0, 25), true), Next::Continue);
        // The last admissible tool-requesting turn: used = 25, cap = 25.
        assert_eq!(step(&state(24, 25), true), Next::Continue);
    }

    #[test]
    fn tool_reply_over_cap_hits_max_iterations() {
        // used = 26, cap = 25.
        assert_eq!(
            step(&state(25, 25), true),
            Next::MaxIterations { limit: 25 }
        );
    }

    #[test]
    fn cap_of_zero_admits_no_tool_turns() {
        assert_eq!(step(&state(0, 0), true), Next::MaxIterations { limit: 0 });
        // A tool-free reply still finishes even with a zero cap.
        assert_eq!(step(&state(0, 0), false), Next::Finish);
    }
}
