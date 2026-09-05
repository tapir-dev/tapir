//! PROTOTYPE - throwaway. Answers wayfinder ticket #8: the Agent core - run
//! loop, state, and builder - built on `tapir-provider` and on the surfaces
//! locked by #4 (`AgentMessage`), #5 (typed `Tool`), #6 (`AgentEvent` + `Run`)
//! and #7 (error model).
//!
//! NOT production code. Run:
//!   cargo run --manifest-path prototypes/agentcore/Cargo.toml
//!   cargo test --manifest-path prototypes/agentcore/Cargo.toml
//!
//! The verdict lives in the ticket resolution comment; this file is the
//! concrete artifact to react to. It builds against the real `tapir-provider`
//! types and drives a hand-scripted streaming provider through the loop so
//! every decision below actually executes.
//!
//! What it decides, made concrete below:
//!   Q1 HYBRID KERNEL: an imperative async `run_loop` driver (IO: provider
//!      stream, tool execute, emit) whose turn-transition *decisions* are
//!      factored into a pure, synchronous `step()` fn - IO-free and unit-
//!      tested. Steals the state-machine testability without a rewrite.
//!   Q2 `Agent<M = NoCustom>` owns `Vec<AgentMessage<M>>`. Each turn the loop
//!      runs `transform_context` (prune/compact) then `convert_to_llm` to
//!      build the provider `Context`. `Agent` == `Agent<NoCustom>` for the
//!      zero-ceremony common case.
//!   Q3 PLAIN FLUENT builder validating at `.build() -> Result<Agent, Error>`
//!      (the `Build` variant). No typestate.
//!   Q4 `max_tool_iterations` (default 25): one iteration == one turn whose
//!      reply requested >=1 tool call. A tool-free reply finishes the run and
//!      does not count. Overflow ends the run with `Error::MaxIterations`.
//!   Q5 RECOVERY: a provider error discards the partial message (never commits
//!      it) so `resume()` replays cleanly; a bad tool call rides back as a
//!      `ToolResult { is_error }` and re-prompts within `max_tool_iterations`.
//!   Q6/Q7 SKELETON ONLY: the loop fixes the boundary points (cancel checked
//!      at turn-top + tool-batch boundary; steer drained at `TurnEnd`; the
//!      tool batch is one seam a parallel scheduler later replaces) and threads
//!      an inert `CancellationToken` + steer channel. `Run` gains NO new public
//!      surface here; the cancellation/steering/parallel/approval tickets
//!      expose what is already wired.
//!
//! Reconciliation #8 makes (surfaced while wiring terminals): #6 named a
//! terminal `ProviderError` event; #7 fixed the terminal type as
//! `AgentEvent::Error(tapir::Error)`. The loop terminates on provider *and*
//! SDK-native failures (MaxIterations, Cancelled), which are not provider
//! errors, so there is ONE terminal `AgentEvent::Error { error: Arc<Error> }`
//! carrying a `tapir::Error` (its `Provider` variant wraps provider errors) -
//! folding #6's `ProviderError` into #7's unified type.

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Error as ProviderError,
    ErrorKind, FinishReason, Message, Provider, StreamAccumulator, StreamEvent,
    StreamEvents,
};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

// ===========================================================================
// tapir::Error stand-in (ticket #7). Runtime-only; not serde. `Provider`
// wraps the provider error verbatim; the rest are SDK-native.
// ===========================================================================

#[derive(Debug)]
pub enum Error {
    Provider(ProviderError),
    MaxIterations { limit: usize },
    Cancelled,
    Build(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Provider(e) => write!(f, "provider: {e}"),
            Error::MaxIterations { limit } => {
                write!(f, "max tool iterations reached (limit {limit})")
            }
            Error::Cancelled => write!(f, "run cancelled"),
            Error::Build(m) => write!(f, "build: {m}"),
        }
    }
}
impl std::error::Error for Error {}

// ===========================================================================
// AgentMessage<M> (ticket #4). `NoCustom` is uninhabited, so the default
// `Agent<NoCustom>` carries only real provider messages with zero ceremony.
// ===========================================================================

/// The per-effort custom-message seam. `to_llm` IS `convert_to_llm`:
/// `None` means UI-only (never sent to the model).
pub trait CustomMessage {
    fn to_llm(&self) -> Option<Message>;
}

/// Uninhabited: the common case has no custom messages.
pub enum NoCustom {}
impl CustomMessage for NoCustom {
    fn to_llm(&self) -> Option<Message> {
        match *self {}
    }
}

/// A message in the agent's history: either a real LLM message or a custom /
/// UI-only one.
pub enum AgentMessage<M = NoCustom> {
    Llm(Message),
    Custom(M),
}

impl<M: CustomMessage> AgentMessage<M> {
    fn convert_to_llm(&self) -> Option<Message> {
        match self {
            AgentMessage::Llm(m) => Some(m.clone()),
            AgentMessage::Custom(c) => c.to_llm(),
        }
    }
}

/// `transform_context` (#4) folded together with `convert_to_llm`: borrow the
/// history, own out the LLM messages to send. The default prunes nothing; a
/// compacting effort drops/summarises before converting. This is the single
/// per-turn hook the loop calls right before building the `Context`.
type Transform<M> = Arc<dyn Fn(&[AgentMessage<M>]) -> Vec<Message> + Send + Sync>;

fn default_transform<M: CustomMessage>() -> Transform<M> {
    Arc::new(|msgs: &[AgentMessage<M>]| {
        msgs.iter().filter_map(|m| m.convert_to_llm()).collect()
    })
}

// ===========================================================================
// Tools: prototype stand-in for the typed `Tool` / erased `Arc<dyn ErasedTool>`
// of #5. Sync `Value -> (output, is_error)`; a bad call returns is_error=true
// (== the #5 dispatch boundary normalising a validation/author error). The
// batch step below is the seam the parallel-execution ticket replaces.
// ===========================================================================

type ToolFn = Arc<dyn Fn(serde_json::Value) -> (String, bool) + Send + Sync>;

// ===========================================================================
// AgentEvent + Run (ticket #6), with the terminal error unified per #7.
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunId(pub u64);

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart { run: RunId },
    TurnStart { turn: usize },
    MessageStart { turn: usize },
    MessageUpdate { turn: usize, delta: StreamEvent },
    MessageEnd { turn: usize, message: AssistantMessage },
    ToolExecutionStart { turn: usize, id: String, name: String, arguments: serde_json::Value },
    ToolExecutionEnd { turn: usize, id: String, output: String, is_error: bool },
    TurnEnd { turn: usize },
    /// Terminal on failure (no `AgentEnd` follows). Carries `tapir::Error` -
    /// provider OR SDK-native (MaxIterations/Cancelled). `Arc` because the
    /// wrapped provider `Error` is not `Clone`.
    Error { turn: Option<usize>, error: Arc<Error> },
    /// Terminal on success: the reply that stopped without asking for a tool.
    AgentEnd { run: RunId, message: AssistantMessage },
}

/// The handle `prompt()`/`resume()` return. A `Stream<AgentEvent>` that is also
/// a `Future` (await-the-final-message). Public surface is exactly #6's - no
/// abort/steer methods here (Q7: those graduate as their own tickets).
pub struct Run {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl futures::Stream for Run {
    type Item = AgentEvent;
    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
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
            Err(Arc::new(Error::Provider(ProviderError::new(
                ErrorKind::Other,
                "run ended without a terminal event",
            ))))
        })
    }
}

/// The inert control seam (Q7): threaded into every run, exposed by NO public
/// method on `Run`. The cancellation / steering tickets decide the surface;
/// this proves the loop already admits them.
pub struct RunControls {
    pub cancel: CancellationToken,
    pub steer: mpsc::UnboundedSender<String>,
}

// ===========================================================================
// Q1 THE PURE KERNEL: turn-transition decision, IO-free, unit-tested below.
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoopState {
    /// Turns so far whose reply requested tools.
    tool_iterations: usize,
    max_tool_iterations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Next {
    /// Reply asked for tools and the cap allows it: execute the batch, loop.
    Continue,
    /// Reply answered without tools: finish with it.
    Finish,
    /// The reply asked for tools but the cap is spent.
    MaxIterationsReached { limit: usize },
    /// Cancellation observed at this boundary.
    Cancel,
}

/// Given the current state, whether the just-settled reply asked for tools, and
/// whether cancellation is signalled, decide what the driver does next. Pure.
fn step(state: &LoopState, reply_wants_tools: bool, cancelled: bool) -> Next {
    if cancelled {
        return Next::Cancel;
    }
    if !reply_wants_tools {
        return Next::Finish;
    }
    // This reply is a tool iteration; count it and check the cap.
    let used = state.tool_iterations + 1;
    if used > state.max_tool_iterations {
        return Next::MaxIterationsReached { limit: state.max_tool_iterations };
    }
    Next::Continue
}

const DEFAULT_MAX_TOOL_ITERATIONS: usize = 25;

// ===========================================================================
// Q3 THE BUILDER: plain fluent, validate at build().
// ===========================================================================

pub struct AgentBuilder<M: CustomMessage + Send + 'static = NoCustom> {
    provider: Option<Arc<dyn Provider>>,
    system: Option<String>,
    tools: HashMap<String, ToolFn>,
    transform: Option<Transform<M>>,
    max_tool_iterations: usize,
}

impl<M: CustomMessage + Send + 'static> AgentBuilder<M> {
    pub fn new() -> Self {
        Self {
            provider: None,
            system: None,
            tools: HashMap::new(),
            transform: None,
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
        }
    }

    /// Model is chosen when the provider is built (#2: the provider owns the
    /// model), so the builder takes a `provider`, not a `.model(..)`.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
    pub fn tool(mut self, name: impl Into<String>, f: ToolFn) -> Self {
        self.tools.insert(name.into(), f);
        self
    }
    pub fn transform_context(mut self, f: Transform<M>) -> Self {
        self.transform = Some(f);
        self
    }
    pub fn max_tool_iterations(mut self, n: usize) -> Self {
        self.max_tool_iterations = n;
        self
    }

    pub fn build(self) -> Result<Agent<M>, Error> {
        let provider = self
            .provider
            .ok_or_else(|| Error::Build("provider is required".into()))?;
        let (events, _) = broadcast::channel(256);
        Ok(Agent {
            provider,
            system: self.system,
            tools: self.tools,
            transform: self.transform.unwrap_or_else(default_transform::<M>),
            max_tool_iterations: self.max_tool_iterations,
            messages: Arc::new(Mutex::new(Vec::new())),
            events,
            next_run: 0,
        })
    }
}

impl<M: CustomMessage + Send + 'static> Default for AgentBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Q2 THE AGENT: stateful, in-memory over Vec<AgentMessage<M>>.
// ===========================================================================

pub struct Agent<M: CustomMessage + Send + 'static = NoCustom> {
    provider: Arc<dyn Provider>,
    system: Option<String>,
    tools: HashMap<String, ToolFn>,
    transform: Transform<M>,
    max_tool_iterations: usize,
    /// Owned in memory behind a mutex so a spawned run appends to the same
    /// history the next `prompt` reads. This is also the single SessionStore
    /// write-through point (#9, blocked): commits happen where `push` is
    /// called; #9 decides the trait, #8 only marks the seam.
    messages: Arc<Mutex<AgentMessages<M>>>,
    events: broadcast::Sender<AgentEvent>,
    next_run: u64,
}

type AgentMessages<M> = Vec<AgentMessage<M>>;

impl Agent<NoCustom> {
    /// Zero-ceremony entry for the common (no custom messages) case.
    pub fn builder() -> AgentBuilder<NoCustom> {
        AgentBuilder::new()
    }
}

impl<M: CustomMessage + Send + 'static> Agent<M> {
    /// Append a user turn, then run to completion.
    pub fn prompt(&mut self, input: impl Into<String>) -> Run {
        self.messages
            .lock()
            .unwrap()
            .push(AgentMessage::Llm(Message::user(input)));
        self.run().0
    }

    /// Continue with no new user input (`continue` is a keyword -> `resume`).
    pub fn resume(&mut self) -> Run {
        self.run().0
    }

    /// Session-wide event fan-out across all runs.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// The internal run entry that also hands back the inert control seam.
    /// `prompt`/`resume` drop the controls (Q7); a demo / future ticket keeps
    /// them. NOT part of #8's public surface.
    fn run(&mut self) -> (Run, RunControls) {
        let run = RunId(self.next_run);
        self.next_run += 1;

        let (tx, rx) = mpsc::unbounded_channel();
        let (steer_tx, steer_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();

        let params = LoopParams {
            run,
            provider: self.provider.clone(),
            tools: self.tools.clone(),
            transform: self.transform.clone(),
            system: self.system.clone(),
            max_tool_iterations: self.max_tool_iterations,
            messages: self.messages.clone(),
            tx,
            bcast: self.events.clone(),
            cancel: cancel.clone(),
            steer_rx,
        };
        tokio::spawn(run_loop(params));

        (Run { rx }, RunControls { cancel, steer: steer_tx })
    }

    fn history_len(&self) -> usize {
        self.messages.lock().unwrap().len()
    }
}

// ===========================================================================
// THE DRIVER: imperative async loop. IO lives here; decisions live in step().
// ===========================================================================

struct LoopParams<M: CustomMessage + Send + 'static> {
    run: RunId,
    provider: Arc<dyn Provider>,
    tools: HashMap<String, ToolFn>,
    transform: Transform<M>,
    system: Option<String>,
    max_tool_iterations: usize,
    messages: Arc<Mutex<AgentMessages<M>>>,
    tx: mpsc::UnboundedSender<AgentEvent>,
    bcast: broadcast::Sender<AgentEvent>,
    cancel: CancellationToken,
    steer_rx: mpsc::UnboundedReceiver<String>,
}

async fn run_loop<M: CustomMessage + Send + 'static>(mut p: LoopParams<M>) {
    let emit = |ev: AgentEvent| {
        let _ = p.bcast.send(ev.clone());
        let _ = p.tx.send(ev);
    };

    emit(AgentEvent::AgentStart { run: p.run });

    let mut state = LoopState {
        tool_iterations: 0,
        max_tool_iterations: p.max_tool_iterations,
    };
    let mut turn = 0usize;

    let outcome: Result<AssistantMessage, Error> = loop {
        // Cancellation checkpoint #1: turn top (Q6).
        if p.cancel.is_cancelled() {
            break Err(Error::Cancelled);
        }

        emit(AgentEvent::TurnStart { turn });

        // Build the provider Context: transform_context (prune/compact) folded
        // with convert_to_llm, under one lock, no AgentMessage clone (Q2).
        let ctx = {
            let guard = p.messages.lock().unwrap();
            let llm_messages = (p.transform)(&guard);
            let mut ctx = Context::new(llm_messages);
            if let Some(system) = &p.system {
                ctx = ctx.with_system(system.clone());
            }
            ctx
        };
        let opts = CompletionOptions::default();

        // Stream the turn, folding deltas into the settled message.
        emit(AgentEvent::MessageStart { turn });
        let mut acc = StreamAccumulator::new();
        let mut events = match p.provider.complete_stream(&ctx, &opts).await {
            Ok(events) => events,
            Err(error) => break Err(Error::Provider(error)),
        };
        let mut stream_failed = false;
        while let Some(item) = events.next().await {
            match item {
                Ok(delta) => {
                    acc.push(&delta);
                    emit(AgentEvent::MessageUpdate { turn, delta });
                }
                Err(error) => {
                    // Q5: discard the partial - do NOT commit `acc` to state.
                    // resume() will replay from the last committed message.
                    break_stream_error(&emit, turn, error, &mut stream_failed);
                    break;
                }
            }
        }
        if stream_failed {
            // Terminal already emitted with the precise turn; stop the run.
            return;
        }

        let message = acc.finish();
        emit(AgentEvent::MessageEnd { turn, message: message.clone() });
        p.messages
            .lock()
            .unwrap()
            .push(AgentMessage::Llm(Message::Assistant(message.clone())));

        let calls: Vec<(String, String, serde_json::Value)> = message
            .tool_calls()
            .filter_map(|part| match part {
                ContentPart::ToolCall { id, name, arguments } => {
                    Some((id.clone(), name.clone(), arguments.clone()))
                }
                _ => None,
            })
            .collect();

        // Q1: the pure decision. Cancellation checkpoint #2 rides in here too.
        match step(&state, !calls.is_empty(), p.cancel.is_cancelled()) {
            Next::Cancel => break Err(Error::Cancelled),
            Next::Finish => {
                emit(AgentEvent::TurnEnd { turn });
                break Ok(message);
            }
            Next::MaxIterationsReached { limit } => {
                break Err(Error::MaxIterations { limit });
            }
            Next::Continue => {
                state.tool_iterations += 1;
            }
        }

        // Execute the tool batch. One seam: a parallel scheduler (fog ticket)
        // replaces this sequential loop; concurrency classes (#5) decide what
        // may overlap. A bad call rides back as is_error and re-prompts (Q5).
        for (id, name, arguments) in calls {
            emit(AgentEvent::ToolExecutionStart {
                turn,
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            });
            let (output, is_error) = match p.tools.get(&name) {
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
            p.messages.lock().unwrap().push(AgentMessage::Llm(result));
        }

        emit(AgentEvent::TurnEnd { turn });

        // Steering / follow-up boundary (Q6): drain any queued input at
        // TurnEnd and append it before the next turn. Inert when empty.
        while let Ok(msg) = p.steer_rx.try_recv() {
            p.messages
                .lock()
                .unwrap()
                .push(AgentMessage::Llm(Message::user(msg)));
        }

        turn += 1;
    };

    match outcome {
        Ok(message) => emit(AgentEvent::AgentEnd { run: p.run, message }),
        Err(error) => emit(AgentEvent::Error {
            turn: Some(turn),
            error: Arc::new(error),
        }),
    }
}

/// Emit the terminal provider-error event and mark the stream failed. Split out
/// so the driver's stream loop stays readable.
fn break_stream_error<F: Fn(AgentEvent)>(
    emit: &F,
    turn: usize,
    error: ProviderError,
    stream_failed: &mut bool,
) {
    emit(AgentEvent::Error {
        turn: Some(turn),
        error: Arc::new(Error::Provider(error)),
    });
    *stream_failed = true;
}

// ===========================================================================
// A hand-scripted streaming provider, so the loop actually runs.
// ===========================================================================

enum StreamItem {
    Ev(StreamEvent),
    Err(&'static str),
}

enum TurnOutcome {
    Stream(Vec<StreamItem>),
    /// complete_stream returns Err before any event (vs a mid-stream error).
    /// Wired seam; no scenario exercises it in this prototype.
    #[allow(dead_code)]
    OpenFail(&'static str),
}

/// Scripted by a closure over the call index. Stands in for a real
/// `tapir_provider::Provider`; only `complete_stream` matters here.
struct ScriptedProvider {
    script: Box<dyn Fn(usize) -> TurnOutcome + Send + Sync>,
    call: AtomicUsize,
}

impl ScriptedProvider {
    fn new(script: impl Fn(usize) -> TurnOutcome + Send + Sync + 'static) -> Self {
        Self { script: Box::new(script), call: AtomicUsize::new(0) }
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn complete(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Err(ProviderError::new(ErrorKind::Other, "scripted: use complete_stream"))
    }

    async fn complete_stream(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        let n = self.call.fetch_add(1, Ordering::SeqCst);
        match (self.script)(n) {
            TurnOutcome::OpenFail(msg) => Err(ProviderError::new(ErrorKind::Overloaded, msg)),
            TurnOutcome::Stream(items) => {
                let evs: Vec<Result<StreamEvent, ProviderError>> = items
                    .into_iter()
                    .map(|it| match it {
                        StreamItem::Ev(e) => Ok(e),
                        StreamItem::Err(m) => {
                            Err(ProviderError::new(ErrorKind::Transport, m))
                        }
                    })
                    .collect();
                Ok(Box::pin(stream::iter(evs)))
            }
        }
    }
}

// --- scripted turn builders -------------------------------------------------

fn text_turn(text: &str) -> Vec<StreamItem> {
    use StreamEvent::*;
    vec![
        StreamItem::Ev(MessageStart),
        StreamItem::Ev(TextStart { index: 0 }),
        StreamItem::Ev(TextDelta { index: 0, text: text.to_string() }),
        StreamItem::Ev(TextEnd { index: 0 }),
        StreamItem::Ev(Done {
            finish_reason: FinishReason::Stop,
            usage: tapir_provider::Usage::default(),
        }),
    ]
}

fn tool_turn(id: &str, name: &str, args_json: &str) -> Vec<StreamItem> {
    use StreamEvent::*;
    vec![
        StreamItem::Ev(MessageStart),
        StreamItem::Ev(ToolCallStart { index: 0, id: id.to_string(), name: name.to_string() }),
        StreamItem::Ev(ToolCallDelta { index: 0, partial_json: args_json.to_string() }),
        StreamItem::Ev(ToolCallEnd { index: 0 }),
        StreamItem::Ev(Done {
            finish_reason: FinishReason::ToolUse,
            usage: tapir_provider::Usage::default(),
        }),
    ]
}

// ===========================================================================
// Demo: four scenarios exercising every decision.
// ===========================================================================

#[tokio::main]
async fn main() {
    scenario_normal_tool_loop().await;
    scenario_provider_error_discards_partial().await;
    scenario_max_iterations().await;
    scenario_cancelled().await;
    println!("\nall scenarios ran.");
}

fn get_weather_tool() -> ToolFn {
    Arc::new(|args: serde_json::Value| {
        let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("?");
        (format!("{{\"temp_c\":21,\"city\":\"{city}\"}}"), false)
    })
}

async fn scenario_normal_tool_loop() {
    println!("== scenario 1: normal two-turn tool loop ==");
    let provider = Arc::new(ScriptedProvider::new(|n| match n {
        0 => TurnOutcome::Stream(tool_turn("call_1", "get_weather", r#"{"city":"Paris"}"#)),
        _ => TurnOutcome::Stream(text_turn("It's 21C in Paris.")),
    }));
    let mut agent = Agent::builder()
        .provider(provider)
        .system("You are helpful.")
        .tool("get_weather", get_weather_tool())
        .build()
        .expect("provider set");

    let mut run = agent.prompt("weather in Paris?");
    while let Some(ev) = run.next().await {
        print_event(&ev);
    }
}

async fn scenario_provider_error_discards_partial() {
    println!("\n== scenario 2: provider error mid-stream discards the partial (Q5) ==");
    let provider = Arc::new(ScriptedProvider::new(|_n| {
        use StreamEvent::*;
        TurnOutcome::Stream(vec![
            StreamItem::Ev(MessageStart),
            StreamItem::Ev(TextStart { index: 0 }),
            StreamItem::Ev(TextDelta { index: 0, text: "partial...".into() }),
            StreamItem::Err("connection reset"),
        ])
    }));
    let mut agent = Agent::builder().provider(provider).build().unwrap();

    let mut run = agent.prompt("hi");
    while let Some(ev) = run.next().await {
        print_event(&ev);
    }
    // Only the user message is committed; the partial assistant reply is gone.
    println!("   committed history len = {} (user only; partial discarded)", agent.history_len());
}

async fn scenario_max_iterations() {
    println!("\n== scenario 3: max_tool_iterations cap (Q4) ==");
    // Provider asks for a tool forever; cap of 2 stops it.
    let provider = Arc::new(ScriptedProvider::new(|n| {
        TurnOutcome::Stream(tool_turn(&format!("call_{n}"), "noop", "{}"))
    }));
    let mut agent = Agent::builder()
        .provider(provider)
        .tool("noop", Arc::new(|_| ("ok".into(), false)))
        .max_tool_iterations(2)
        .build()
        .unwrap();

    let mut run = agent.prompt("loop forever");
    while let Some(ev) = run.next().await {
        print_event(&ev);
    }
}

async fn scenario_cancelled() {
    println!("\n== scenario 4: cancellation checkpoint (Q6/Q7 inert seam) ==");
    let provider = Arc::new(ScriptedProvider::new(|_| TurnOutcome::Stream(text_turn("hi"))));
    let mut agent = Agent::builder().provider(provider).build().unwrap();

    // Drive via the internal control seam and cancel before the turn top.
    let (mut run, controls) = agent.run();
    controls.cancel.cancel();
    while let Some(ev) = run.next().await {
        print_event(&ev);
    }
}

fn print_event(ev: &AgentEvent) {
    match ev {
        AgentEvent::AgentStart { run } => println!("   AgentStart {run:?}"),
        AgentEvent::TurnStart { turn } => println!("   TurnStart turn={turn}"),
        AgentEvent::MessageStart { turn } => println!("   MessageStart turn={turn}"),
        AgentEvent::MessageUpdate { turn, delta } => {
            println!("   MessageUpdate turn={turn} delta={delta:?}")
        }
        AgentEvent::MessageEnd { turn, message } => {
            println!("   MessageEnd turn={turn} text={:?}", message.text_content())
        }
        AgentEvent::ToolExecutionStart { turn, name, .. } => {
            println!("   ToolExecutionStart turn={turn} name={name}")
        }
        AgentEvent::ToolExecutionEnd { turn, output, is_error, .. } => {
            println!("   ToolExecutionEnd turn={turn} is_error={is_error} output={output}")
        }
        AgentEvent::TurnEnd { turn } => println!("   TurnEnd turn={turn}"),
        AgentEvent::Error { turn, error } => println!("   Error turn={turn:?} error={error}"),
        AgentEvent::AgentEnd { run, message } => {
            println!("   AgentEnd {run:?} final={:?}", message.text_content())
        }
    }
}

// ===========================================================================
// Unit tests for the pure kernel (Q1) + async recovery/cap tests.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn st(iters: usize, max: usize) -> LoopState {
        LoopState { tool_iterations: iters, max_tool_iterations: max }
    }

    #[test]
    fn step_finishes_when_reply_has_no_tools() {
        assert_eq!(step(&st(0, 25), false, false), Next::Finish);
    }

    #[test]
    fn step_continues_within_cap() {
        assert_eq!(step(&st(0, 25), true, false), Next::Continue);
    }

    #[test]
    fn step_allows_exactly_the_cap() {
        // 24 done, this is the 25th tool iteration, max 25 -> allowed.
        assert_eq!(step(&st(24, 25), true, false), Next::Continue);
    }

    #[test]
    fn step_stops_one_past_the_cap() {
        // 25 done, this would be the 26th, max 25 -> stop.
        assert_eq!(step(&st(25, 25), true, false), Next::MaxIterationsReached { limit: 25 });
    }

    #[test]
    fn step_cancel_takes_precedence_over_everything() {
        assert_eq!(step(&st(0, 25), true, true), Next::Cancel);
        assert_eq!(step(&st(0, 25), false, true), Next::Cancel);
    }

    #[tokio::test]
    async fn provider_error_yields_err_and_discards_partial() {
        let provider = Arc::new(ScriptedProvider::new(|_| {
            use StreamEvent::*;
            TurnOutcome::Stream(vec![
                StreamItem::Ev(MessageStart),
                StreamItem::Ev(TextDelta { index: 0, text: "x".into() }),
                StreamItem::Err("boom"),
            ])
        }));
        let mut agent = Agent::builder().provider(provider).build().unwrap();
        let err = agent.prompt("hi").await.unwrap_err();
        assert!(matches!(&*err, Error::Provider(_)));
        // Only the user message committed; partial assistant discarded.
        assert_eq!(agent.history_len(), 1);
    }

    #[tokio::test]
    async fn max_iterations_terminates_with_error() {
        let provider = Arc::new(ScriptedProvider::new(|n| {
            TurnOutcome::Stream(tool_turn(&format!("c{n}"), "noop", "{}"))
        }));
        let mut agent = Agent::builder()
            .provider(provider)
            .tool("noop", Arc::new(|_| ("ok".into(), false)))
            .max_tool_iterations(2)
            .build()
            .unwrap();
        let err = agent.prompt("go").await.unwrap_err();
        assert!(matches!(&*err, Error::MaxIterations { limit: 2 }));
    }

    #[tokio::test]
    async fn cancel_before_turn_top_yields_cancelled() {
        let provider = Arc::new(ScriptedProvider::new(|_| TurnOutcome::Stream(text_turn("hi"))));
        let mut agent = Agent::builder().provider(provider).build().unwrap();
        let (run, controls) = agent.run();
        controls.cancel.cancel();
        let err = run.await.unwrap_err();
        assert!(matches!(&*err, Error::Cancelled));
    }

    #[tokio::test]
    async fn normal_loop_runs_tool_then_answers() {
        let provider = Arc::new(ScriptedProvider::new(|n| match n {
            0 => TurnOutcome::Stream(tool_turn("c1", "get_weather", r#"{"city":"Paris"}"#)),
            _ => TurnOutcome::Stream(text_turn("done")),
        }));
        let mut agent = Agent::builder()
            .provider(provider)
            .tool("get_weather", get_weather_tool())
            .build()
            .unwrap();
        let msg = agent.prompt("weather?").await.unwrap();
        assert_eq!(msg.text_content(), "done");
        // user, assistant(toolcall), toolresult, assistant(text) = 4 committed.
        assert_eq!(agent.history_len(), 4);
    }

    #[test]
    fn builder_requires_provider() {
        let built = AgentBuilder::<NoCustom>::new().build();
        assert!(matches!(built, Err(Error::Build(_))));
    }
}
