// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The stateful agent and its run surface. An [`Agent`] owns the in-memory
//! conversation history and drives runs. Each run is a hybrid kernel: an async
//! driver (`run_loop`) owns the IO, while the pure `step` function decides the
//! turn-to-turn transition and is unit-testable with no async or IO. The driver
//! runs a tool-requesting turn's batch through the concurrency-classed scheduler
//! (`execute_batch`): consecutive `Safe` calls run concurrently while each
//! `Exclusive` call serializes behind a barrier, and results are assembled back
//! in model order before feeding history. Steering lands in a later ticket; the
//! cancellation seam is scaffolded here for the abort ticket to drive.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Message,
    Provider, StreamAccumulator, ToolDefinition, ToolResultMessage,
};
use tokio::sync::{broadcast, mpsc};

use crate::cancel::{Cancel, cancel_pair};
use crate::error::Error;
use crate::event::AgentEvent;
use crate::message::{
    AgentMessage, CustomMessage, NoCustom, TransformContext, convert_to_llm,
};
use crate::tool::{Concurrency, ErasedTool, Tool, ToolCtx};

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
    tools: Vec<Arc<dyn ErasedTool>>,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M> Default for AgentBuilder<M> {
    fn default() -> Self {
        Self {
            provider: None,
            system: None,
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            tools: Vec::new(),
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

    /// Register a tool. Chainable and the only way to add tools of differing
    /// types (a literal array of heterogeneous `#[tool]` fns will not compile),
    /// so each `Tool` erases here to an `Arc<dyn ErasedTool>` the run loop
    /// stores uniformly.
    #[must_use]
    pub fn tool<T: Tool>(mut self, tool: T) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Register an already-erased collection of tools in bulk, appending to any
    /// added with [`tool`](Self::tool).
    #[must_use]
    pub fn tools(
        mut self,
        tools: impl IntoIterator<Item = Arc<dyn ErasedTool>>,
    ) -> Self {
        self.tools.extend(tools);
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
            tools: Arc::new(self.tools),
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
    /// The registered tools, erased and shared into every run.
    tools: Arc<Vec<Arc<dyn ErasedTool>>>,
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
            tools: self.tools.clone(),
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
    tools: Arc<Vec<Arc<dyn ErasedTool>>>,
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
        tools,
        transform,
        messages,
        tx,
        bcast,
    } = driver;
    // The provider-ready definitions offered every turn; derived once since the
    // tool set is fixed for the run.
    let tool_defs: Vec<ToolDefinition> =
        tools.iter().map(|t| t.definition()).collect();
    // Fan each event out both the per-run stream and the session broadcast. The
    // `Emitter` is cloneable so a spawned tool task can emit from its own task;
    // `emit` is the driver-body shorthand that borrows it.
    let emitter = Emitter { tx, bcast };
    let emit = |ev: AgentEvent| emitter.emit(ev);

    // One cancellation token spans the run. Held here for the run's lifetime so
    // it outlives every batch; the abort ticket wires the trigger onto the
    // `RunHandle`. Until then the trigger never fires, so the run is effectively
    // un-cancellable and `execute_batch` always completes.
    let (_cancel_trigger, cancel) = cancel_pair();

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
            if !tool_defs.is_empty() {
                ctx = ctx.with_tools(tool_defs.clone());
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

        // Run the batch through the concurrency-classed scheduler: consecutive
        // `Safe` calls run concurrently, each `Exclusive` call serializes behind
        // a barrier, and results come back in the model's order. Each result
        // (success or an `is_error` failure) is appended to history so the next
        // turn re-prompts on it; a bad-arg or author error is a `ToolResult`,
        // never a `tapir::Error`, so the run continues within the cap.
        let batch_ctx = BatchCtx {
            turn,
            emitter: emitter.clone(),
            cancel: cancel.clone(),
        };
        match execute_batch(&tools, &tool_calls_of(&message), &batch_ctx).await
        {
            BatchOutcome::Completed(results) => {
                let mut guard =
                    messages.lock().expect("history mutex poisoned");
                for result in results {
                    guard.push(AgentMessage::Llm(Message::ToolResult(result)));
                }
            }
            BatchOutcome::Cancelled => {
                // The abort ticket owns terminal handling; until it lands the
                // trigger never fires, so this arm is unreachable in a live run.
                break Err(Error::Provider(tapir_provider::Error::new(
                    tapir_provider::ErrorKind::Other,
                    "run cancelled",
                )));
            }
        }

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

/// One model-requested call, lifted out of the reply into owned fields so the
/// batch can run while history is mutated without holding a borrow on the
/// message. `Clone` so each spawned call task owns its copy.
#[derive(Clone)]
struct ToolCall {
    id: String,
    name: String,
    arguments: Value,
}

/// The reply's tool calls in model order, each lifted into an owned
/// [`ToolCall`].
fn tool_calls_of(message: &AssistantMessage) -> Vec<ToolCall> {
    message
        .tool_calls()
        .filter_map(|part| match part {
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Fans a run's events onto both the per-run stream and the session broadcast.
/// Cloneable and `Send` so a spawned tool task emits from its own task.
#[derive(Clone)]
struct Emitter {
    tx: mpsc::UnboundedSender<AgentEvent>,
    bcast: broadcast::Sender<AgentEvent>,
}

impl Emitter {
    /// Send one event to both sinks; a closed receiver is not an error here.
    fn emit(&self, ev: AgentEvent) {
        let _ = self.bcast.send(ev.clone());
        let _ = self.tx.send(ev);
    }
}

/// The outcome of running one turn's tool batch.
enum BatchOutcome {
    /// Every call ran; results are in the model's order.
    Completed(Vec<ToolResultMessage>),
    /// Cancellation fired mid-batch: queued calls were never launched and
    /// in-flight calls were aborted, so no orphaned work remains.
    Cancelled,
}

/// The shared, cloneable context every call in one turn's batch carries: the
/// turn index it belongs to, the event sink, and the cancellation token. Bundled
/// so it threads through the scheduler and into each spawned call as one value.
#[derive(Clone)]
struct BatchCtx {
    turn: usize,
    emitter: Emitter,
    cancel: Cancel,
}

/// Run one turn's tool batch under its concurrency classes and return the
/// results in model order.
///
/// The batch is walked in model order and split into waves: a maximal run of
/// consecutive [`Safe`](Concurrency::Safe) calls forms one wave that executes
/// concurrently, while each [`Exclusive`](Concurrency::Exclusive) call is its own
/// solo wave — a barrier that waits for everything before it and blocks
/// everything after. An unknown-tool call is treated as `Safe`: it runs no author
/// code, only mints a synthetic `is_error` result, so it never needs a barrier.
///
/// Each wave is spawned and joined; if `cancel` fires while a wave is in flight,
/// its tasks are aborted (the cascade) and no later wave is launched, yielding
/// [`BatchOutcome::Cancelled`]. Regardless of completion order, results are
/// placed back at each call's original index.
async fn execute_batch(
    tools: &Arc<Vec<Arc<dyn ErasedTool>>>,
    calls: &[ToolCall],
    ctx: &BatchCtx,
) -> BatchOutcome {
    let mut results: Vec<Option<ToolResultMessage>> = vec![None; calls.len()];

    let mut i = 0;
    while i < calls.len() {
        // A cancel that fired between waves stops us before launching the next.
        if ctx.cancel.is_cancelled() {
            return BatchOutcome::Cancelled;
        }

        // Carve the next wave: consecutive `Safe` calls batch together; anything
        // else (Exclusive, or an unknown tool defaulting to its own wave) runs
        // alone as a barrier.
        let start = i;
        if class_of(tools, &calls[i]) == Concurrency::Safe {
            while i < calls.len()
                && class_of(tools, &calls[i]) == Concurrency::Safe
            {
                i += 1;
            }
        } else {
            i += 1;
        }

        let wave: Vec<usize> = (start..i).collect();
        let handles: Vec<_> = wave
            .iter()
            .map(|&idx| {
                let tool = resolve(tools, &calls[idx].name);
                tokio::spawn(run_call(tool, calls[idx].clone(), ctx.clone()))
            })
            .collect();
        let aborts: Vec<_> = handles.iter().map(|h| h.abort_handle()).collect();

        tokio::select! {
            biased;
            () = ctx.cancel.cancelled() => {
                // Cascade: abort every in-flight call in this wave and launch no
                // more, leaving no orphaned tool work behind.
                for abort in &aborts {
                    abort.abort();
                }
                return BatchOutcome::Cancelled;
            }
            outputs = collect_wave(handles) => {
                for (&idx, output) in wave.iter().zip(outputs) {
                    // A join error here is a panicking tool (aborts are handled on
                    // the cancel arm above): surface it as an `is_error` result so
                    // one bad tool does not sink the whole run.
                    results[idx] = Some(output.unwrap_or_else(|_| {
                        tool_result(
                            &calls[idx],
                            vec![ContentPart::text(format!(
                                "tool `{}` panicked",
                                calls[idx].name
                            ))],
                            true,
                        )
                    }));
                }
            }
        }
    }

    BatchOutcome::Completed(
        results
            .into_iter()
            .map(|r| r.expect("every call ran"))
            .collect(),
    )
}

/// Await a wave's spawned calls, collecting their join results in launch order.
/// The tasks run concurrently (they are already spawned); this only harvests
/// them. Kept separate so the caller's `select!` has one future to join on.
async fn collect_wave(
    handles: Vec<tokio::task::JoinHandle<ToolResultMessage>>,
) -> Vec<Result<ToolResultMessage, tokio::task::JoinError>> {
    let mut outputs = Vec::with_capacity(handles.len());
    for handle in handles {
        outputs.push(handle.await);
    }
    outputs
}

/// The concurrency class of a call's tool, or [`Safe`](Concurrency::Safe) for an
/// unknown tool (which runs no author code).
fn class_of(tools: &[Arc<dyn ErasedTool>], call: &ToolCall) -> Concurrency {
    resolve(tools, &call.name)
        .map_or(Concurrency::Safe, |tool| tool.concurrency())
}

/// Find a registered tool by name, cloning the `Arc` for a spawned task.
fn resolve(
    tools: &[Arc<dyn ErasedTool>],
    name: &str,
) -> Option<Arc<dyn ErasedTool>> {
    tools.iter().find(|t| t.name() == name).cloned()
}

/// Run a single tool call end to end: bracket it with `ToolExecution*` events,
/// dispatch through the erased tool's `invoke` boundary, and shape the outcome
/// into a [`ToolResultMessage`] to feed back to the model.
///
/// Every path yields a result rather than an error: an unknown tool produces a
/// synthetic `is_error` result, and a bad-arg or author failure rides back as an
/// `is_error` result carrying only the model-visible message (operator detail is
/// dropped here; a logging seam lands later).
async fn run_call(
    tool: Option<Arc<dyn ErasedTool>>,
    call: ToolCall,
    ctx: BatchCtx,
) -> ToolResultMessage {
    let BatchCtx {
        turn,
        emitter,
        cancel,
    } = ctx;
    emitter.emit(AgentEvent::ToolExecutionStart {
        turn,
        call_id: call.id.clone(),
        name: call.name.clone(),
    });

    let result = match tool {
        None => tool_result(
            &call,
            vec![ContentPart::text(format!("unknown tool `{}`", call.name))],
            true,
        ),
        Some(tool) => {
            let ctx = ToolCtx::with_cancel(&call.id, cancel);
            let mut sink = |update| {
                emitter.emit(AgentEvent::ToolExecutionUpdate {
                    turn,
                    call_id: call.id.clone(),
                    update,
                });
            };
            match tool.invoke(call.arguments.clone(), &ctx, &mut sink).await {
                Ok(output) => tool_result(&call, output.content, false),
                Err(error) => tool_result(
                    &call,
                    vec![ContentPart::text(error.model_message)],
                    true,
                ),
            }
        }
    };

    emitter.emit(AgentEvent::ToolExecutionEnd {
        turn,
        result: result.clone(),
    });
    result
}

/// Assemble a [`ToolResultMessage`] answering `call`.
fn tool_result(
    call: &ToolCall,
    content: Vec<ContentPart>,
    is_error: bool,
) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: call.id.clone(),
        tool_name: call.name.clone(),
        content,
        is_error,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use async_trait::async_trait;

    use super::*;
    use crate::cancel::cancel_pair;
    use crate::tool::{ToolError, UpdateSink};

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

    /// A `Safe` tool that flags when it starts, then parks forever — so the only
    /// way it ever stops is the executor aborting its task.
    struct Blocker {
        started: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Tool for Blocker {
        type Args = ();
        type Output = ();
        type Error = ToolError;

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
            (): (),
            _ctx: &ToolCtx,
            _on_update: &mut UpdateSink<'_>,
        ) -> Result<(), ToolError> {
            self.started.store(true, Ordering::SeqCst);
            std::future::pending::<()>().await;
            // Unreachable while aborted; flips only if the task were allowed to
            // run to completion.
            self.finished.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// An `Exclusive` tool that records whether it was ever invoked.
    struct Marker {
        ran: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Tool for Marker {
        type Args = ();
        type Output = ();
        type Error = ToolError;

        fn name(&self) -> &str {
            "marker"
        }
        fn description(&self) -> &str {
            "records that it ran"
        }
        fn concurrency(&self) -> Concurrency {
            Concurrency::Exclusive
        }

        async fn execute(
            &self,
            (): (),
            _ctx: &ToolCtx,
            _on_update: &mut UpdateSink<'_>,
        ) -> Result<(), ToolError> {
            self.ran.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: Value::Null,
        }
    }

    fn test_emitter() -> Emitter {
        let (tx, _rx) = mpsc::unbounded_channel();
        let (bcast, _) = broadcast::channel(64);
        Emitter { tx, bcast }
    }

    fn test_ctx(cancel: Cancel) -> BatchCtx {
        BatchCtx {
            turn: 0,
            emitter: test_emitter(),
            cancel,
        }
    }

    #[tokio::test]
    async fn cancel_aborts_inflight_and_skips_queued() {
        let started = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let ran_after = Arc::new(AtomicBool::new(false));

        let tools: Arc<Vec<Arc<dyn ErasedTool>>> = Arc::new(vec![
            Arc::new(Blocker {
                started: started.clone(),
                finished: finished.clone(),
            }),
            Arc::new(Marker {
                ran: ran_after.clone(),
            }),
        ]);
        // A `Safe` call in flight, then an `Exclusive` call queued behind its
        // barrier so it only launches after the blocker's wave settles.
        let calls = vec![call("c1", "blocker"), call("c2", "marker")];

        let (trigger, cancel) = cancel_pair();
        let ctx = test_ctx(cancel);
        let batch =
            tokio::spawn(
                async move { execute_batch(&tools, &calls, &ctx).await },
            );

        // Cancel only once the blocker is genuinely in flight.
        while !started.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        trigger.cancel();

        let outcome = batch.await.expect("batch task");
        assert!(matches!(outcome, BatchOutcome::Cancelled));
        assert!(
            !finished.load(Ordering::SeqCst),
            "the in-flight call must be aborted, not run to completion"
        );
        assert!(
            !ran_after.load(Ordering::SeqCst),
            "the queued call behind the barrier must never launch"
        );
    }

    #[tokio::test]
    async fn completed_results_are_in_model_order() {
        // Two `Exclusive` calls to the same tool: each is its own barrier, yet
        // the results still come back keyed to the model's order.
        let tools: Arc<Vec<Arc<dyn ErasedTool>>> =
            Arc::new(vec![Arc::new(Marker {
                ran: Arc::new(AtomicBool::new(false)),
            })]);
        let calls = vec![call("first", "marker"), call("second", "marker")];

        let (_trigger, cancel) = cancel_pair();
        let ctx = test_ctx(cancel);
        let outcome = execute_batch(&tools, &calls, &ctx).await;

        match outcome {
            BatchOutcome::Completed(results) => {
                let ids: Vec<&str> =
                    results.iter().map(|r| r.tool_call_id.as_str()).collect();
                assert_eq!(ids, ["first", "second"]);
            }
            BatchOutcome::Cancelled => panic!("batch was not cancelled"),
        }
    }
}
