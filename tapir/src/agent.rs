// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The stateful agent and its run surface. An [`Agent`] owns the in-memory
//! conversation history and drives runs. Each run is a hybrid kernel: an async
//! driver (`run_loop`) owns the IO, while the pure `step` function decides the
//! turn-to-turn transition and is unit-testable with no async or IO. The driver
//! runs a tool-requesting turn's batch through the concurrency-classed scheduler
//! (`execute_batch`): consecutive `Safe` calls run concurrently while each
//! `Exclusive` call serializes behind a barrier, and results are assembled back
//! in model order before feeding history. A [`RunHandle`] cloned off the run
//! controls it from another task: `abort` fires the run's cancellation token
//! (honored at turn-top and the batch checkpoints), `steer` injects a user turn
//! drained at the next `TurnEnd`, and `finish` ends a parked follow-up run.
//!
//! `prompt`/`resume` runs end at the first tool-free reply. A [`converse`](Agent::converse)
//! run instead *parks* there: it emits [`AgentEvent::Idle`] and awaits more input,
//! waking on a steer (which resets the tool-iteration cap) and ending only on
//! `finish`, an `idle_timeout`, or an abort.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Message,
    Provider, StreamAccumulator, ThinkingLevel, ToolDefinition,
    ToolResultMessage,
};
use tokio::sync::{broadcast, mpsc};

use crate::cancel::{Cancel, cancel_pair};
use crate::error::Error;
use crate::event::AgentEvent;
use crate::message::{
    AgentMessage, CustomMessage, NoCustom, TransformContext, convert_to_llm,
};
use crate::schema::{NonStrict, SchemaProfile};
use crate::store::SessionStore;
use crate::tool::{
    AfterToolCall, BeforeToolCall, Concurrency, ErasedTool, Tool, ToolCall,
    ToolCtx, ToolDecision,
};

/// Default cap on tool-requesting turns before a run fails with
/// [`Error::MaxIterations`].
const DEFAULT_MAX_TOOL_ITERATIONS: usize = 25;

/// Capacity of the session-wide broadcast channel backing [`Agent::subscribe`].
const BROADCAST_CAPACITY: usize = 256;

/// Disambiguates concurrent runs on the session-wide broadcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub u64);

/// How a steer injection combines with input already steered but not yet drained.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SteerMode {
    /// Add the steer text as one more pending user turn.
    Append,
    /// Discard any pending steered input and start over from the steer text.
    Replace,
}

/// One command on a run's ordered control channel. `Input` injects a user turn
/// that the driver drains at the next `TurnEnd`; `Finish` requests a graceful
/// stop.
enum SteerCmd {
    /// Inject `text` as a steered user turn, combined per `mode`.
    Input {
        /// The steer text.
        text: String,
        /// How it combines with any pending steer input.
        mode: SteerMode,
    },
    /// Request the run finish gracefully: end a parked follow-up run, or, on a
    /// run still in its turn loop, prevent the next park so it ends at its next
    /// tool-free reply.
    Finish,
}

/// The cloneable control surface that outlives the [`Run`] the `.await` consumes.
/// Obtained via [`Run::handle`] and shared to another task to abort or steer a
/// live run. Every call is infallible and silent — a call after the run has
/// terminated is a no-op.
#[derive(Clone)]
pub struct RunHandle {
    /// Fires the run's cancellation token; cloned from the run.
    cancel: crate::cancel::CancelTrigger,
    /// The ordered control channel into the driver.
    steer: mpsc::UnboundedSender<SteerCmd>,
}

impl RunHandle {
    /// Abort the run cooperatively. Honored at the next turn-top or tool-batch
    /// checkpoint, where it cascade-cancels any in-flight batch and terminates
    /// the run with [`Error::Cancelled`]. Idempotent, and a no-op once the run
    /// has already terminated.
    pub fn abort(&self) {
        self.cancel.cancel();
    }

    /// Steer the run by appending `text` as a user turn, drained at the next
    /// [`TurnEnd`](AgentEvent::TurnEnd). Shorthand for
    /// [`steer_with`](Self::steer_with) with [`SteerMode::Append`].
    pub fn steer(&self, text: impl Into<String>) {
        self.steer_with(text, SteerMode::Append);
    }

    /// Steer the run with an explicit [`SteerMode`]: `Append` adds `text` as
    /// another pending user turn, `Replace` discards any pending steered input
    /// first. Applied at the next [`TurnEnd`](AgentEvent::TurnEnd). A no-op once
    /// the run has terminated.
    pub fn steer_with(&self, text: impl Into<String>, mode: SteerMode) {
        // A closed channel means the run already ended: drop silently.
        let _ = self.steer.send(SteerCmd::Input {
            text: text.into(),
            mode,
        });
    }

    /// Finish the run gracefully. On a [`converse`](Agent::converse) run parked
    /// in the idle state (having emitted [`AgentEvent::Idle`]) this ends it at
    /// once, resolving the run with [`AgentEnd`](AgentEvent::AgentEnd) carrying
    /// the parked reply. On a run still in its turn loop it is remembered and
    /// prevents the next park, so the run ends at its next tool-free reply. A
    /// no-op once the run has terminated, and on a `prompt`/`resume` run (which
    /// never parks).
    pub fn finish(&self) {
        // A closed channel means the run already ended: drop silently.
        let _ = self.steer.send(SteerCmd::Finish);
    }
}

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
    /// A model id set via [`model`](Self::model), resolved offline at
    /// [`build`](Self::build) into a provider. Mutually exclusive with
    /// `provider`. Only settable when a provider feature is compiled in.
    #[cfg(any(feature = "anthropic", feature = "openai"))]
    model: Option<String>,
    system: Option<String>,
    /// The reasoning-effort level applied to every turn's completion; `None`
    /// leaves the provider default (no thinking budget).
    thinking: Option<ThinkingLevel>,
    max_tool_iterations: usize,
    idle_timeout: Option<Duration>,
    tools: Vec<Arc<dyn ErasedTool>>,
    before_tool_call: Option<BeforeToolCall>,
    after_tool_call: Option<AfterToolCall>,
    /// The send-path tool-schema profile; defaults to the [`NonStrict`]
    /// passthrough.
    schema_profile: Arc<dyn SchemaProfile>,
    store: Option<Arc<dyn SessionStore<M>>>,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M> Default for AgentBuilder<M> {
    fn default() -> Self {
        Self {
            provider: None,
            #[cfg(any(feature = "anthropic", feature = "openai"))]
            model: None,
            system: None,
            thinking: None,
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            idle_timeout: None,
            tools: Vec::new(),
            before_tool_call: None,
            after_tool_call: None,
            schema_profile: Arc::new(NonStrict),
            store: None,
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
    /// instance, so it is chosen when the provider is constructed. Mutually
    /// exclusive with [`model`](Self::model), validated at [`build`](Self::build).
    #[must_use]
    pub fn provider<P: Provider + 'static>(mut self, provider: P) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    /// Name a model by id and let the builder resolve the provider offline — the
    /// convenience path over [`provider`](Self::provider), feature-gated by
    /// `anthropic`/`openai`.
    ///
    /// Resolution runs synchronously at [`build`](Self::build): the compiled-in
    /// catalog is loaded, the id resolved first-match-wins across enabled
    /// providers, and the adapter built over a default reqwest client with
    /// credentials from the provider's env var (e.g. `ANTHROPIC_API_KEY`). For an
    /// explicit key or a custom HTTP client, construct the provider yourself and
    /// pass it to [`provider`](Self::provider) instead.
    ///
    /// Mutually exclusive with [`provider`](Self::provider): setting both (or
    /// neither) fails at `build` with [`Error::Build`]; an unknown id fails there
    /// too, and a missing credential surfaces as [`Error::Provider`].
    #[cfg(any(feature = "anthropic", feature = "openai"))]
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Override the send-path tool-schema profile (default [`NonStrict`], a
    /// passthrough). The profile normalizes each offered tool's argument schema
    /// in place every turn before the provider call.
    #[must_use]
    pub fn schema_profile(
        mut self,
        profile: impl SchemaProfile + 'static,
    ) -> Self {
        self.schema_profile = Arc::new(profile);
        self
    }

    /// Set the system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the reasoning-effort level applied to every turn's completion. Unset
    /// (the default) leaves the provider default, i.e. no thinking budget. The
    /// level rides on each turn's [`CompletionOptions`] via
    /// [`with_thinking`](tapir_provider::CompletionOptions::with_thinking).
    #[must_use]
    pub fn thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking = Some(level);
        self
    }

    /// Set the cap on tool-requesting turns (default 25). Overflow fails the
    /// run with [`Error::MaxIterations`].
    #[must_use]
    pub fn max_tool_iterations(mut self, n: usize) -> Self {
        self.max_tool_iterations = n;
        self
    }

    /// Set how long a [`converse`](Agent::converse) run may sit parked in the
    /// idle state before it ends itself with [`AgentEnd`](AgentEvent::AgentEnd).
    /// `Some(d)` ends a run left idle for `d` with no steer or
    /// [`finish`](RunHandle::finish); `None` (the default) parks indefinitely,
    /// so only `finish` or an abort ends it. No effect on a `prompt`/`resume`
    /// run, which never parks.
    #[must_use]
    pub fn idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.idle_timeout = timeout;
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

    /// Install the pre-batch tool-call approval gate. Awaited once per call in
    /// model order before the turn's batch runs — and so before the concurrency
    /// window opens, so a human wait (the gate awaiting its own UI) holds no
    /// concurrency slot and stalls no barrier. The closure returns a
    /// [`ToolDecision`]: [`Proceed`](ToolDecision::Proceed) runs the call
    /// silently, [`Modify`](ToolDecision::Modify) reruns arg validation with
    /// rewritten arguments, and [`Deny`](ToolDecision::Deny) skips execution,
    /// feeding the model a synthetic `is_error` result plus an
    /// [`AgentEvent::ToolCallDenied`].
    /// Unset (the default) admits every call. An abort drops a pending gate
    /// future with no grace.
    #[must_use]
    pub fn before_tool_call<F>(mut self, gate: F) -> Self
    where
        F: for<'a> Fn(&'a ToolCall) -> crate::tool::BoxFuture<'a, ToolDecision>
            + Send
            + Sync
            + 'static,
    {
        self.before_tool_call = Some(Arc::new(gate));
        self
    }

    /// Install the post-batch, observe-only tool-result hook. Awaited once per
    /// executed result (never for a gate-denied call), in model order, after the
    /// batch settles. It cannot change the result — it only observes. Unset (the
    /// default) is a no-op.
    #[must_use]
    pub fn after_tool_call<F>(mut self, observer: F) -> Self
    where
        F: for<'a> Fn(
                &'a ToolCall,
                &'a tapir_provider::ToolResultMessage,
            ) -> crate::tool::BoxFuture<'a, ()>
            + Send
            + Sync
            + 'static,
    {
        self.after_tool_call = Some(Arc::new(observer));
        self
    }

    /// Attach a [`SessionStore`], making the agent persist history write-through
    /// as a run progresses. Absent (the default) the agent is ephemeral. To
    /// reopen a persisted session, use [`Agent::resume`] instead, which also
    /// seeds history from the store.
    #[must_use]
    pub fn store(mut self, store: Arc<dyn SessionStore<M>>) -> Self {
        self.store = Some(store);
        self
    }

    /// Validate the configuration and build the agent. Synchronous: it resolves
    /// exactly one of [`provider`](Self::provider) or [`model`](Self::model) —
    /// both or neither is [`Error::Build`] — and, on the `model` path, builds the
    /// provider offline from the compiled-in catalog (an unknown id is
    /// `Error::Build`, a missing credential [`Error::Provider`]).
    pub fn build(self) -> Result<Agent<M>, Error> {
        let provider = self.resolve_provider()?;
        let (events, _) = broadcast::channel(BROADCAST_CAPACITY);
        Ok(Agent {
            provider,
            system: self.system,
            thinking: self.thinking,
            max_tool_iterations: self.max_tool_iterations,
            idle_timeout: self.idle_timeout,
            tools: Arc::new(self.tools),
            before_tool_call: self.before_tool_call,
            after_tool_call: self.after_tool_call,
            schema_profile: self.schema_profile,
            transform: None,
            store: self.store,
            messages: Arc::new(Mutex::new(Vec::new())),
            persisted: Arc::new(Mutex::new(0)),
            events,
            next_run: AtomicU64::new(0),
            _marker: std::marker::PhantomData,
        })
    }

    /// Resolve the configured provider seam. With a provider feature compiled in,
    /// [`provider`](Self::provider) and [`model`](Self::model) are mutually
    /// exclusive and exactly one is required; the `model` path resolves offline.
    #[cfg(any(feature = "anthropic", feature = "openai"))]
    fn resolve_provider(&self) -> Result<Arc<dyn Provider>, Error> {
        match (&self.provider, &self.model) {
            (Some(_), Some(_)) => Err(Error::Build(
                "`.model` and `.provider` are mutually exclusive".to_string(),
            )),
            (None, None) => {
                Err(Error::Build("a provider or model is required".to_string()))
            }
            (Some(provider), None) => Ok(provider.clone()),
            (None, Some(model)) => resolve_model(model),
        }
    }

    /// Resolve the configured provider seam. With no provider feature there is no
    /// `model` path, so a provider is required.
    #[cfg(not(any(feature = "anthropic", feature = "openai")))]
    fn resolve_provider(&self) -> Result<Arc<dyn Provider>, Error> {
        self.provider
            .clone()
            .ok_or_else(|| Error::Build("provider is required".to_string()))
    }
}

/// Resolve a `.model("id")` offline into a live provider: load the compiled-in
/// catalog with env-var credentials, find the entry first-match-wins across
/// enabled providers, and build the adapter over a default reqwest client. An
/// unknown id is [`Error::Build`]; a load or construction failure (e.g. a missing
/// credential) is [`Error::Provider`].
#[cfg(any(feature = "anthropic", feature = "openai"))]
fn resolve_model(id: &str) -> Result<Arc<dyn Provider>, Error> {
    use tapir_provider::http::ReqwestClient;
    use tapir_provider::{ModelRegistry, create_provider};

    let registry = ModelRegistry::load(None, None)?;
    let entry = registry
        .find_by_id(id)
        .ok_or_else(|| Error::Build(format!("unknown model id `{id}`")))?;
    Ok(create_provider(entry, ReqwestClient::new())?)
}

/// The stateful object owning the in-memory conversation history and driving
/// runs. Generic over a custom-message type `M`, defaulting to [`NoCustom`].
pub struct Agent<M = NoCustom> {
    provider: Arc<dyn Provider>,
    system: Option<String>,
    /// The reasoning-effort level applied to every run's turns; `None` leaves
    /// the provider default.
    thinking: Option<ThinkingLevel>,
    max_tool_iterations: usize,
    /// How long a `converse` run parks in the idle state before ending itself;
    /// `None` parks indefinitely. Ignored by `prompt`/`resume` runs.
    idle_timeout: Option<Duration>,
    /// The registered tools, erased and shared into every run.
    tools: Arc<Vec<Arc<dyn ErasedTool>>>,
    /// The pre-batch approval gate, shared into every run; `None` admits every
    /// call.
    before_tool_call: Option<BeforeToolCall>,
    /// The post-batch observe-only result hook, shared into every run; `None`
    /// is a no-op.
    after_tool_call: Option<AfterToolCall>,
    /// The send-path tool-schema profile, shared into every run; normalizes each
    /// offered tool's argument schema before the provider call.
    schema_profile: Arc<dyn SchemaProfile>,
    /// The `transform_context` seam applied to history each turn before
    /// `convert_to_llm`. `None` is the identity (no reshaping); the builder
    /// that installs one lands in a later ticket.
    transform: Option<Arc<TransformContext<M>>>,
    /// The persistence seam. `None` (the default) is ephemeral; when set, each
    /// run writes history through it as it progresses.
    store: Option<Arc<dyn SessionStore<M>>>,
    /// History owned behind a mutex so a spawned run appends to the same
    /// history the next `prompt` reads.
    messages: Arc<Mutex<Vec<AgentMessage<M>>>>,
    /// How many leading messages in `messages` are already durable in `store`.
    /// The write-through high-water mark: a run appends only the tail beyond it,
    /// so seeded (resumed) history and messages already written are never
    /// re-appended. Meaningful only when `store` is set.
    persisted: Arc<Mutex<usize>>,
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

impl<M: CustomMessage + Clone + Send + Sync + 'static> Agent<M> {
    /// Reopen a persisted session: build the agent from `builder`, attach
    /// `store`, and seed history from what the store has already recorded. The
    /// returned agent continues the conversation — a following [`prompt`] appends
    /// to the seeded history and persists write-through as usual.
    ///
    /// Async because loading crosses the store's async boundary; a load failure
    /// fails closed as [`Error::Session`]. The seeded history is exactly what
    /// `store` returns — resume is the sole seeding path, so it stays mutually
    /// exclusive with any builder-side message seeding a later ticket adds.
    ///
    /// [`prompt`]: Agent::prompt
    pub async fn resume(
        builder: AgentBuilder<M>,
        store: Arc<dyn SessionStore<M>>,
    ) -> Result<Self, Error> {
        let history = store.load().await?;
        let seeded = history.len();
        let agent = builder.store(store).build()?;
        *agent.messages.lock().expect("history mutex poisoned") = history;
        *agent.persisted.lock().expect("persist mark poisoned") = seeded;
        Ok(agent)
    }

    /// Append a user turn, then run to completion. Terminates at the first
    /// tool-free reply. For an interactive run that parks awaiting more input,
    /// use [`converse`](Self::converse) instead.
    pub fn prompt(&self, input: impl Into<String>) -> Run {
        self.messages
            .lock()
            .expect("history mutex poisoned")
            .push(AgentMessage::Llm(Message::user(input)));
        self.run(false)
    }

    /// Append a user turn, then run as an interactive follow-up: instead of
    /// ending at a tool-free reply, the run parks in the idle state — it emits
    /// [`AgentEvent::Idle`] and awaits more input. A [`steer`](RunHandle::steer)
    /// wakes it (resetting the tool-iteration cap) and the loop resumes;
    /// [`finish`](RunHandle::finish), the builder's
    /// [`idle_timeout`](AgentBuilder::idle_timeout), or an abort ends it. The
    /// run's future resolves only once it ends, carrying the final reply.
    pub fn converse(&self, input: impl Into<String>) -> Run {
        self.messages
            .lock()
            .expect("history mutex poisoned")
            .push(AgentMessage::Llm(Message::user(input)));
        self.run(true)
    }

    /// Subscribe to this session's events across all runs. Each run's events are
    /// tagged with a [`RunId`] so concurrent runs are distinguishable.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Spawn the driver on its own task and hand back the [`Run`]. `follow_up`
    /// selects interactive parking: `true` (via [`converse`](Self::converse))
    /// parks at a tool-free reply, `false` (via [`prompt`](Self::prompt))
    /// terminates there.
    fn run(&self, follow_up: bool) -> Run {
        let run = RunId(self.next_run.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::unbounded_channel();
        // The control surface: one cancellation token and one ordered steer
        // channel. The trigger and steer sender ride on the `Run` so
        // `Run::handle` can clone them onto a `RunHandle` that outlives it; the
        // observer half and the steer receiver move onto the driver task.
        let (cancel_trigger, cancel) = cancel_pair();
        let (steer_tx, steer_rx) = mpsc::unbounded_channel();
        tokio::spawn(run_loop(RunLoop {
            run,
            provider: self.provider.clone(),
            system: self.system.clone(),
            thinking: self.thinking,
            max_tool_iterations: self.max_tool_iterations,
            idle_timeout: self.idle_timeout,
            follow_up,
            tools: self.tools.clone(),
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            schema_profile: self.schema_profile.clone(),
            transform: self.transform.clone(),
            store: self.store.clone(),
            messages: self.messages.clone(),
            persisted: self.persisted.clone(),
            tx,
            bcast: self.events.clone(),
            cancel,
            steer: steer_rx,
        }));
        Run {
            rx,
            cancel: cancel_trigger,
            steer: steer_tx,
        }
    }
}

/// One invocation of the agent: both a [`Stream`] of [`AgentEvent`] (terminating
/// with [`AgentEvent::AgentEnd`]) and, via [`IntoFuture`], a future resolving to
/// the settled reply.
pub struct Run {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
    /// The run's cancellation trigger, cloned onto each [`RunHandle`].
    cancel: crate::cancel::CancelTrigger,
    /// The sending half of the run's ordered control channel.
    steer: mpsc::UnboundedSender<SteerCmd>,
}

impl Run {
    /// Take a [`RunHandle`] onto this run: a cloneable control surface that
    /// outlives the `Run` the `.await` consumes, so another task can abort or
    /// steer the run after this value is gone.
    #[must_use]
    pub fn handle(&self) -> RunHandle {
        RunHandle {
            cancel: self.cancel.clone(),
            steer: self.steer.clone(),
        }
    }
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
    /// The reasoning-effort level applied to every turn's completion; `None`
    /// leaves the provider default.
    thinking: Option<ThinkingLevel>,
    max_tool_iterations: usize,
    /// How long to park in the idle state before ending a follow-up run; `None`
    /// parks indefinitely. Only consulted when `follow_up` is set.
    idle_timeout: Option<Duration>,
    /// Whether a tool-free reply parks (interactive `converse`) or ends the run
    /// (`prompt`/`resume`).
    follow_up: bool,
    tools: Arc<Vec<Arc<dyn ErasedTool>>>,
    /// The pre-batch approval gate for this run; `None` admits every call.
    before_tool_call: Option<BeforeToolCall>,
    /// The post-batch observe-only result hook for this run; `None` is a no-op.
    after_tool_call: Option<AfterToolCall>,
    /// The send-path tool-schema profile for this run.
    schema_profile: Arc<dyn SchemaProfile>,
    transform: Option<Arc<TransformContext<M>>>,
    store: Option<Arc<dyn SessionStore<M>>>,
    messages: Arc<Mutex<Vec<AgentMessage<M>>>>,
    persisted: Arc<Mutex<usize>>,
    tx: mpsc::UnboundedSender<AgentEvent>,
    bcast: broadcast::Sender<AgentEvent>,
    /// The observer half of the run's cancellation token, watched at turn-top and
    /// threaded into each tool batch for the cascade.
    cancel: Cancel,
    /// The receiving half of the run's ordered control channel, drained at each
    /// `TurnEnd`.
    steer: mpsc::UnboundedReceiver<SteerCmd>,
}

/// The async driver for one run. Owns the IO and event fan-out; defers every
/// turn-transition decision to the pure `step` kernel. One turn is one
/// `complete_stream` folded through a [`StreamAccumulator`]; the provider
/// context is built each turn from `transform` then [`convert_to_llm`] under
/// one lock.
async fn run_loop<M: CustomMessage + Clone + Send + Sync + 'static>(
    driver: RunLoop<M>,
) {
    let RunLoop {
        run,
        provider,
        system,
        thinking,
        max_tool_iterations,
        idle_timeout,
        follow_up,
        tools,
        before_tool_call,
        after_tool_call,
        schema_profile,
        transform,
        store,
        messages,
        persisted,
        tx,
        bcast,
        cancel,
        mut steer,
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

    emit(AgentEvent::AgentStart { run });

    let mut state = LoopState {
        tool_iterations: 0,
        max_tool_iterations,
    };
    let mut turn = 0usize;
    // Sticky: a `finish()` seen mid-run keeps a follow-up run from parking, so it
    // ends at its next tool-free reply.
    let mut finish_requested = false;

    let outcome: Result<AssistantMessage, Error> = loop {
        // Turn-top abort checkpoint: an abort fired between turns terminates the
        // run here, before any provider work, with `Error::Cancelled`.
        if cancel.is_cancelled() {
            break Err(Error::Cancelled);
        }

        emit(AgentEvent::TurnStart { turn });

        // Write-through the pending tail before the provider call — on the first
        // turn that is the user message this run is answering, so the request is
        // itself durable. A store error fails the run closed.
        if let Err(error) = persist_tail(&store, &messages, &persisted).await {
            break Err(error);
        }

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
                // Normalize each offered tool's schema on the send path. The
                // default `NonStrict` profile is a passthrough; a custom profile
                // reshapes the portable schema for its provider here.
                let mut defs = tool_defs.clone();
                for def in &mut defs {
                    schema_profile.normalize(&mut def.input_schema);
                }
                ctx = ctx.with_tools(defs);
            }
            ctx
        };
        let opts = match thinking {
            Some(level) => CompletionOptions::default().with_thinking(level),
            None => CompletionOptions::default(),
        };

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

        // Persist the settled reply — an artifact after the call — before the
        // transition is decided, so it is durable whether the run finishes,
        // hits the cap, or continues. Fail closed on a store error.
        if let Err(error) = persist_tail(&store, &messages, &persisted).await {
            break Err(error);
        }

        let reply_wants_tools = message.tool_calls().next().is_some();
        match step(&state, reply_wants_tools) {
            Next::Finish => {
                emit(AgentEvent::TurnEnd { turn });
                // A `prompt`/`resume` run ends here, and so does a follow-up run
                // once `finish` has been requested. Otherwise the follow-up run
                // parks: it emits `Idle` and awaits the next control signal.
                if !follow_up || finish_requested {
                    break Ok(message);
                }
                emit(AgentEvent::Idle { run });
                match park(&mut steer, &cancel, idle_timeout).await {
                    Park::Steered(steered) => {
                        // A steer woke the run: inject the folded user turns and
                        // re-arm the tool budget so the continuation gets a fresh
                        // cap, then resume with the next turn.
                        inject_user_turns(&messages, steered);
                        state.tool_iterations = 0;
                        turn += 1;
                        continue;
                    }
                    // `finish` or the idle timeout ends the run with the parked
                    // reply.
                    Park::Finished | Park::TimedOut => break Ok(message),
                    // An abort fired while parked: terminate the run.
                    Park::Cancelled => break Err(Error::Cancelled),
                }
            }
            Next::MaxIterations { limit } => {
                break Err(Error::MaxIterations { limit });
            }
            Next::Continue => {
                state.tool_iterations += 1;
            }
        }

        // Gate the batch before it runs: every call passes through
        // `before_tool_call` sequentially in model order, ahead of the
        // concurrency window, so a human wait never holds a slot. A denied call
        // yields a synthetic result in its model-order slot and never executes.
        let calls = tool_calls_of(&message);
        let gated = match gate_batch(
            &before_tool_call,
            &calls,
            turn,
            &emitter,
            &cancel,
        )
        .await
        {
            GateOutcome::Decided(gated) => gated,
            // An abort dropped a pending gate future: terminate the run.
            GateOutcome::Cancelled => break Err(Error::Cancelled),
        };

        // Split the gated calls: denied ones already carry a synthetic result;
        // the rest run through the batch, each remembering its model-order slot.
        let mut results: Vec<Option<ToolResultMessage>> =
            vec![None; calls.len()];
        let mut runnable: Vec<(usize, ToolCall)> = Vec::new();
        for (idx, verdict) in gated.into_iter().enumerate() {
            match verdict {
                GatedCall::Denied(result) => results[idx] = Some(result),
                GatedCall::Run(call) => runnable.push((idx, call)),
            }
        }

        // Run the admitted calls through the concurrency-classed scheduler:
        // consecutive `Safe` calls run concurrently, each `Exclusive` call
        // serializes behind a barrier, and results come back in the runnable
        // slice's order. Each result (success or an `is_error` failure) is
        // appended to history so the next turn re-prompts on it; a bad-arg or
        // author error is a `ToolResult`, never a `tapir::Error`, so the run
        // continues within the cap.
        let batch_ctx = BatchCtx {
            turn,
            emitter: emitter.clone(),
            cancel: cancel.clone(),
        };
        let runnable_calls: Vec<ToolCall> =
            runnable.iter().map(|(_, call)| call.clone()).collect();
        match execute_batch(&tools, &runnable_calls, &batch_ctx).await {
            BatchOutcome::Completed(executed) => {
                // Observe each executed result (never a gate-denied one) in
                // model order, then place it back in its model-order slot.
                for ((idx, call), result) in runnable.into_iter().zip(executed)
                {
                    observe_result(&after_tool_call, &call, &result).await;
                    results[idx] = Some(result);
                }
                let mut guard =
                    messages.lock().expect("history mutex poisoned");
                for result in results {
                    let result = result.expect("every call gated or executed");
                    guard.push(AgentMessage::Llm(Message::ToolResult(result)));
                }
            }
            BatchOutcome::Cancelled => {
                // An abort fired mid-batch: the cascade already aborted in-flight
                // calls and launched no more, so terminate the run here.
                break Err(Error::Cancelled);
            }
        }

        // Persist the batch's tool results — artifacts after the call — before
        // the next turn re-prompts on them. Fail closed on a store error.
        if let Err(error) = persist_tail(&store, &messages, &persisted).await {
            break Err(error);
        }

        // TurnEnd steer drain: fold every queued control command, then inject each
        // steered user turn into history so the next turn's provider call picks
        // them up at the turn boundary. A queued `finish` is remembered so a
        // follow-up run ends at its next tool-free reply instead of parking.
        let (steered, finish) = drain_steer(&mut steer);
        finish_requested |= finish;
        inject_user_turns(&messages, steered);

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

/// Flush any unpersisted tail of history — everything past the `persisted`
/// high-water mark — to the store, one `append().await` per message, bumping the
/// mark as each resolves durable. A no-op when the agent is ephemeral (no store).
///
/// Called before the provider call (persisting the user turn) and after each
/// artifact push, so history is durable message by message. A store error
/// normalizes to [`Error::Session`] for the caller to surface as the terminal
/// error; the just-pushed message stays only in memory, never seen as durable.
async fn persist_tail<M: CustomMessage + Clone + Send + Sync + 'static>(
    store: &Option<Arc<dyn SessionStore<M>>>,
    messages: &Arc<Mutex<Vec<AgentMessage<M>>>>,
    persisted: &Arc<Mutex<usize>>,
) -> Result<(), Error> {
    let Some(store) = store else {
        return Ok(());
    };
    loop {
        // Snapshot the next unpersisted message and drop both locks before the
        // await — the std mutexes never cross an await point.
        let next = {
            let guard = messages.lock().expect("history mutex poisoned");
            let mark = *persisted.lock().expect("persist mark poisoned");
            guard.get(mark).cloned()
        };
        match next {
            None => return Ok(()),
            Some(message) => {
                store.append(&message).await?;
                *persisted.lock().expect("persist mark poisoned") += 1;
            }
        }
    }
}

/// Drain every command queued on the control channel and fold the steer inputs
/// into the user turns to inject, in order, reporting whether a
/// [`Finish`](SteerCmd::Finish) was among them. The channel is ordered, so
/// folding in receive order honors the caller's sequence:
/// [`Append`](SteerMode::Append) adds its text as one more pending turn, while
/// [`Replace`](SteerMode::Replace) discards whatever is pending and starts over
/// from its text. Purely non-blocking — it takes only what is already queued and
/// never awaits.
fn drain_steer(
    steer: &mut mpsc::UnboundedReceiver<SteerCmd>,
) -> (Vec<String>, bool) {
    let mut pending: Vec<String> = Vec::new();
    let mut finish = false;
    while let Ok(cmd) = steer.try_recv() {
        match cmd {
            SteerCmd::Input { text, mode } => {
                fold_steer_input(&mut pending, text, mode);
            }
            SteerCmd::Finish => finish = true,
        }
    }
    (pending, finish)
}

/// Fold one steered input into the pending user turns per its [`SteerMode`]:
/// [`Append`](SteerMode::Append) adds another turn,
/// [`Replace`](SteerMode::Replace) discards the pending turns and starts over
/// from this text. Shared by the `TurnEnd` drain and the idle-park wake so both
/// honor the same `Append`/`Replace` semantics.
fn fold_steer_input(pending: &mut Vec<String>, text: String, mode: SteerMode) {
    match mode {
        SteerMode::Append => pending.push(text),
        SteerMode::Replace => {
            pending.clear();
            pending.push(text);
        }
    }
}

/// Push each steered text into history as its own user turn, under one lock; a
/// no-op on an empty batch. Shared by the `TurnEnd` drain and the idle-park wake,
/// which inject folded steer input the same way.
fn inject_user_turns<M>(
    messages: &Arc<Mutex<Vec<AgentMessage<M>>>>,
    turns: Vec<String>,
) {
    if turns.is_empty() {
        return;
    }
    let mut guard = messages.lock().expect("history mutex poisoned");
    for text in turns {
        guard.push(AgentMessage::Llm(Message::user(text)));
    }
}

/// The outcome of parking a follow-up run at a tool-free reply.
enum Park {
    /// A steer arrived: inject these folded user turns and resume with a fresh
    /// tool-iteration cap.
    Steered(Vec<String>),
    /// [`finish`](RunHandle::finish) was requested, or every control sender
    /// dropped: end the run with [`AgentEnd`](AgentEvent::AgentEnd).
    Finished,
    /// The idle timeout elapsed with no input: end the run with `AgentEnd`.
    TimedOut,
    /// An abort fired while parked: terminate the run with [`Error::Cancelled`].
    Cancelled,
}

/// Park a follow-up run in the idle state, awaiting the next control signal.
///
/// Resolves when a steer wakes it — folding the waking command with anything
/// already queued behind it through the shared [`fold_steer_input`], so the
/// caller's `Append`/`Replace` sequence is honored — or when
/// [`finish`](RunHandle::finish),
/// the `idle_timeout`, or an abort ends it. A [`Finish`](SteerCmd::Finish) seen
/// anywhere in the woken batch ends the run (finish wins over queued input). With
/// `idle_timeout` `None` the run parks indefinitely, so only `finish` or an abort
/// ends it. Blocks on [`recv`](mpsc::UnboundedReceiver::recv) rather than the
/// non-blocking drain the turn loop uses, since a parked run has nothing else to
/// do until a signal arrives.
async fn park(
    steer: &mut mpsc::UnboundedReceiver<SteerCmd>,
    cancel: &Cancel,
    idle_timeout: Option<Duration>,
) -> Park {
    let sleep = async {
        match idle_timeout {
            Some(d) => tokio::time::sleep(d).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(sleep);

    let first = tokio::select! {
        biased;
        () = cancel.cancelled() => return Park::Cancelled,
        cmd = steer.recv() => cmd,
        () = &mut sleep => return Park::TimedOut,
    };
    // Every control sender dropped: no one can steer or finish, so end gracefully
    // rather than hang the task.
    let Some(first) = first else {
        return Park::Finished;
    };

    // Fold the waking command with everything already queued behind it.
    let mut cmds = vec![first];
    while let Ok(cmd) = steer.try_recv() {
        cmds.push(cmd);
    }
    let mut steered: Vec<String> = Vec::new();
    for cmd in cmds {
        match cmd {
            SteerCmd::Finish => return Park::Finished,
            SteerCmd::Input { text, mode } => {
                fold_steer_input(&mut steered, text, mode);
            }
        }
    }
    Park::Steered(steered)
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

/// One call's verdict from the pre-batch gate.
enum GatedCall {
    /// The gate denied the call: this synthetic `is_error` result stands in for
    /// it in model order, and the call never executes.
    Denied(ToolResultMessage),
    /// The gate admitted the call (a [`Proceed`](ToolDecision::Proceed), or a
    /// [`Modify`](ToolDecision::Modify) whose rewritten arguments are already
    /// folded in): run it through the batch.
    Run(ToolCall),
}

/// The outcome of gating a turn's batch.
enum GateOutcome {
    /// Every call was decided; verdicts are in model order.
    Decided(Vec<GatedCall>),
    /// An abort fired while a gate future was pending: it was dropped with no
    /// grace and no call executed.
    Cancelled,
}

/// Run every call through the `before_tool_call` gate sequentially in model
/// order, ahead of the concurrency window. `None` admits every call without
/// awaiting. Each future is raced against `cancel`: an abort drops the pending
/// gate future with no grace and yields [`GateOutcome::Cancelled`]. A
/// [`Deny`](ToolDecision::Deny) emits [`AgentEvent::ToolCallDenied`] and mints a
/// synthetic `is_error` result in the call's slot; a
/// [`Modify`](ToolDecision::Modify) folds the rewritten arguments into the call
/// so the batch reruns validation on them.
async fn gate_batch(
    before: &Option<BeforeToolCall>,
    calls: &[ToolCall],
    turn: usize,
    emitter: &Emitter,
    cancel: &Cancel,
) -> GateOutcome {
    let mut gated = Vec::with_capacity(calls.len());
    for call in calls {
        let decision = match before {
            None => ToolDecision::Proceed,
            Some(gate) => tokio::select! {
                biased;
                () = cancel.cancelled() => return GateOutcome::Cancelled,
                decision = gate(call) => decision,
            },
        };
        match decision {
            ToolDecision::Proceed => gated.push(GatedCall::Run(call.clone())),
            ToolDecision::Modify { arguments } => {
                let mut modified = call.clone();
                modified.arguments = arguments;
                gated.push(GatedCall::Run(modified));
            }
            ToolDecision::Deny { message } => {
                emitter.emit(AgentEvent::ToolCallDenied {
                    turn,
                    id: call.id.clone(),
                    message: message.clone(),
                });
                gated.push(GatedCall::Denied(tool_result(
                    call,
                    vec![ContentPart::text(message)],
                    true,
                )));
            }
        }
    }
    GateOutcome::Decided(gated)
}

/// Hand one executed call and its result to the `after_tool_call` observer, if
/// one is installed. A no-op when unset; the hook only observes and cannot
/// change the result.
async fn observe_result(
    after: &Option<AfterToolCall>,
    call: &ToolCall,
    result: &ToolResultMessage,
) {
    if let Some(observer) = after {
        observer(call, result).await;
    }
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
    use serde_json::Value;

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
