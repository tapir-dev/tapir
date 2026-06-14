//! Provider-neutral chat types and the shared system prompt.
//!
//! The wire protocols live in [`crate::providers`] (one module per API shape:
//! OpenAI Responses for Copilot, OpenAI Chat Completions for OpenAI / DeepSeek /
//! OpenRouter, Anthropic Messages, and Google Gemini). This module only holds
//! the types those clients share — the conversation message, images, token
//! usage, a tool call, the round-error wrapper, the streamed `TurnEvent` — plus
//! `system_prompt` and the reasoning-level mapping, identical across providers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// The provider-neutral message vocabulary and the round contract moved to
// `tapir-ai` (the LLM layer). Re-exported so this crate's own `crate::agent::*`
// paths and existing dependents (`tapir_core::agent::TurnEvent`, …) keep
// resolving unchanged.
pub use tapir_ai::message::{
    ChatMessage, Image, Input, ModelRef, Role, RoundError, RoundRunner,
    ToolCall, TurnEvent, Usage, reasoning_effort,
};

/// The system prompt, assembled as: intro, the available-tools list,
/// behavioral guidelines, the prompt-template / skill authoring notes, an
/// optional `append` block (user's `system.md`), the project/global `context`,
/// then the local date/time and working directory last so the model is grounded.
/// Each provider client places this text where its API expects the system
/// instructions; the text itself is identical across providers.
pub fn system_prompt(
    cwd: &std::path::Path,
    context: &str,
    skills: &str,
    append: &str,
    active_tools: &[String],
    custom: Option<&str>,
) -> String {
    // `--system-prompt` replaces the default body (intro + tools + guidelines);
    // context, skills, the append block, and the date/cwd tail are still added.
    let mut prompt = match custom {
        Some(text) => text.to_string(),
        None => default_body(active_tools),
    };
    if !append.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(append);
    }
    if !context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(context);
    }
    // Skills require the `read` tool to load their files, so only list them when
    // it's available.
    if !skills.is_empty() && active_tools.iter().any(|t| t == "read") {
        prompt.push_str("\n\n");
        prompt.push_str(skills);
    }
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %:z");
    prompt.push_str(&format!(
        "\n\nCurrent date and time: {now}\nCurrent working directory: {}",
        cwd.display()
    ));
    prompt
}

/// The default system-prompt body (intro, available-tools list, guidelines, and
/// the skill/prompt authoring notes), used when `--system-prompt` isn't given.
fn default_body(active_tools: &[String]) -> String {
    let tools = crate::tools::snippets_for(active_tools)
        .iter()
        .map(|(name, snippet)| format!("- {name}: {snippet}"))
        .collect::<Vec<_>>()
        .join("\n");
    // With tools disabled (`--no-tools`), say so instead of an empty list.
    let tools_section = if active_tools.is_empty() {
        "You have no tools available this session.".to_string()
    } else {
        format!(
            "Available tools:\n{tools}\n\n\
             In addition to the tools above, you may have access to other custom \
             tools depending on the project."
        )
    };

    let mut guidelines = crate::tools::guidelines_for(active_tools);
    guidelines.push("Be concise in your responses");
    guidelines.push("Show file paths clearly when working with files");
    let guidelines = guidelines
        .iter()
        .map(|g| format!("- {g}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are tapir, an interactive terminal coding agent. You help users by \
         reading files, running commands, editing code, and writing new files.\n\n\
         {tools_section}\n\n\
         Guidelines:\n{guidelines}\n\n{}\n\n{}",
        crate::skills::HELP_FOR_SKILL,
        crate::prompts::HELP_FOR_PROMPT,
    )
}

/// Tokens kept free below the context window before auto-compaction fires.
pub const COMPACTION_RESERVE: u64 = 16_384;

/// Structured-summary prompt for auto-compaction.
pub const SUMMARY_PROMPT: &str = "The messages above are a conversation to summarize. \
Create a structured context checkpoint another LLM will use to continue the work.\n\n\
Use this format:\n\n\
## Goal\n[What the user is trying to accomplish.]\n\n\
## Constraints & Preferences\n- [Constraints/preferences, or (none)]\n\n\
## Progress\n### Done\n- [Completed work]\n### In Progress\n- [Current work]\n### Blocked\n- [Blockers, if any]\n\n\
## Key Decisions\n- [Decision]: [rationale]\n\n\
## Next Steps\n1. [What should happen next]\n\n\
## Critical Context\n- [Data/examples/references needed, or (none)]";

/// Upper bound on rounds in a single turn, so a misbehaving model that keeps
/// requesting tools can't spin forever. Steering / follow-up legitimately extend
/// a turn, so the cap is generous.
const MAX_ROUNDS: usize = 256;

/// Recorded in the durable history when the user interrupts a turn, so the next
/// turn's context shows the interruption and the model doesn't silently redo the
/// work. Kept here (the engine owns the conversation) rather than in a frontend.
pub const INTERRUPTED: &str = "[The user interrupted this action before it completed. Do not redo it unless the user asks again.]";

/// A cooperative cancellation handle shared between a frontend and a running
/// turn. Cloning shares the same flag, so aborting any clone aborts them all.
/// The turn loop checks it at each round boundary: an abort lets the in-flight
/// round finish, then stops the turn (a frontend wanting a hard stop also drops
/// the task running it).
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Signal cancellation to every clone of this handle.
    pub fn abort(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been signalled.
    pub fn is_aborted(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Clear a prior cancellation so the handle can be reused for the next turn.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// One in-flight conversation: the neutral history plus the loop that drives a
/// turn to completion. Frontends (the TUI, headless print, future adapters)
/// construct an `Agent` and drive it; the loop emits [`TurnEvent`]s and never
/// touches a terminal.
///
/// Tapir's [`crate::session::Session`] is the on-disk transcript — a separate,
/// persistence concern.
pub struct Agent {
    /// The durable conversation, shared between the held Session handle and a
    /// running turn task — the same pattern as `queue` and `cancel` below.
    /// Locked briefly per operation; never across an await (rounds receive a
    /// snapshot).
    history: History,
    /// The working directory tool calls run in.
    cwd: std::path::PathBuf,
    /// Steering / follow-up messages the frontend queues mid-turn; drained
    /// between rounds. See [`crate::queue`].
    queue: crate::queue::Shared,
    steering_mode: crate::queue::Mode,
    follow_up_mode: crate::queue::Mode,
    /// Cooperative cancellation for the in-flight turn; checked between rounds.
    cancel: CancelToken,
    /// Whether a turn is in flight, as a shared atomic — the engine owns it
    /// (it spawns the turn via [`run_with`](Self::run_with)), and a turn task holding a
    /// clone can clear it when its stream ends.
    running: Arc<AtomicBool>,
    /// The in-flight turn's task, so the engine can hard-abort it. `None` when
    /// idle. Aborting goes through [`abort`](Self::abort) alongside the
    /// cooperative [`CancelToken`].
    run_handle: Option<tokio::task::JoinHandle<()>>,
    /// Read-only consumers notified of every event (registered on the Runtime).
    /// Empty for a bare `Agent`; [`run_with`](Self::run_with) relays through them when set.
    observers: Observers,
    /// Tools the turn loop dispatches through (registered on the Runtime). Tools
    /// not found here fall back to the built-in name dispatch.
    tools: crate::tools::Tools,
    /// The session's opaque metadata, surfaced to each tool's execution context.
    metadata: std::sync::Arc<crate::tools::tool::Metadata>,
    /// Interception hooks (registered on the Runtime) the turn loop consults
    /// around each tool call. Empty for a bare `Agent`.
    hooks: Hooks,
    /// The exec operations `bash` performs its process execution through,
    /// surfaced to each tool's execution context. Local spawn by default; a
    /// sandbox backend injects its own through the Runtime.
    exec_ops: std::sync::Arc<dyn crate::tools::exec::ExecOps>,
    /// The filesystem operations the file tools perform their raw I/O
    /// through, surfaced to each tool's execution context. Local `std::fs`
    /// by default; a sandbox backend injects its own through the Runtime.
    fs_ops: std::sync::Arc<dyn crate::tools::fs::FsOps>,
    /// The optional workspace boundary installed on each tool's execution
    /// context: every model path is translated guest→host and jailed to it.
    /// `None` (the default) keeps today's unrestricted local resolution.
    boundary: Option<crate::tools::jail::PathBoundary>,
    /// The model this session is pointed at, if selected (see [`ModelRef`]).
    model: Option<ModelRef>,
    /// The Runtime this Session was spawned from — what [`run`](Self::run)
    /// resolves the Provider, credentials, and config through. `None` for a
    /// bare `Agent` (tests); a turn without it errors in-channel.
    runtime: Option<crate::runtime::Runtime>,
    /// How [`run`](Self::run) assembles the system prompt each turn.
    prompt: crate::runtime::PromptSpec,
    /// The session's thinking level ("off" … "xhigh"); `None` = medium. Mapped
    /// to the provider effort each turn (see [`reasoning_effort`]).
    thinking: Option<String>,
}

/// A shared, immutable list of [`Observer`](crate::observer::Observer)s — cloned
/// (one `Arc` bump) into the relay task when a turn is spawned.
pub type Observers = Arc<[Arc<dyn crate::observer::Observer>]>;

/// The Session's durable conversation as a shared handle — clones observe each
/// other's writes, so a spawned turn task and the held Session write to the
/// same history (no snapshot-and-mirror). Lock briefly; never across an await.
pub(crate) type History = Arc<std::sync::Mutex<Vec<crate::providers::Step>>>;

/// A shared, immutable list of [`Hook`](crate::hook::Hook)s — cloned (one `Arc`
/// bump) into the turn loop, which consults them around each tool call.
pub type Hooks = Arc<[Arc<dyn crate::hook::Hook>]>;

impl Agent {
    pub fn new(
        history: Vec<crate::providers::Step>,
        cwd: std::path::PathBuf,
        queue: crate::queue::Shared,
        steering_mode: crate::queue::Mode,
        follow_up_mode: crate::queue::Mode,
    ) -> Self {
        Self::with_handles(
            history,
            cwd,
            queue,
            steering_mode,
            follow_up_mode,
            CancelToken::default(),
        )
    }

    /// Like [`new`](Self::new) but sharing an existing [`CancelToken`] — used when
    /// a frontend holds the conversation's cancel handle and spawns the turn on a
    /// separate task that must observe the same aborts.
    pub fn with_handles(
        history: Vec<crate::providers::Step>,
        cwd: std::path::PathBuf,
        queue: crate::queue::Shared,
        steering_mode: crate::queue::Mode,
        follow_up_mode: crate::queue::Mode,
        cancel: CancelToken,
    ) -> Self {
        Self {
            history: Arc::new(std::sync::Mutex::new(history)),
            cwd,
            queue,
            steering_mode,
            follow_up_mode,
            cancel,
            running: Arc::new(AtomicBool::new(false)),
            run_handle: None,
            observers: Arc::from(Vec::new()),
            tools: Arc::from(Vec::new()),
            metadata: Arc::new(crate::tools::tool::Metadata::new()),
            hooks: Arc::from(Vec::new()),
            exec_ops: Arc::new(crate::tools::exec::LocalExecOps),
            fs_ops: Arc::new(crate::tools::fs::LocalFsOps),
            boundary: None,
            model: None,
            runtime: None,
            prompt: crate::runtime::PromptSpec::default(),
            thinking: None,
        }
    }

    /// Set the session's thinking level (the model-slot treatment: a frontend
    /// sets it when the user changes it; [`run`](Self::run) reads it per turn).
    pub fn set_thinking(&mut self, level: impl Into<String>) {
        self.thinking = Some(level.into());
    }

    /// The session's thinking level, if one has been set (`None` = medium).
    pub fn thinking(&self) -> Option<&str> {
        self.thinking.as_deref()
    }

    /// Set how [`run`](Self::run) assembles the system prompt — the Runtime
    /// sets it when spawning the session (from the session options), and an
    /// adapter may replace it between turns (a bot announces a recreated
    /// sandbox environment in the next turn's prompt).
    pub fn set_prompt(&mut self, prompt: crate::runtime::PromptSpec) {
        self.prompt = prompt;
    }

    /// Attach the Runtime this Session resolves turns through (set by the
    /// Runtime when spawning the session).
    pub(crate) fn set_runtime(&mut self, runtime: crate::runtime::Runtime) {
        self.runtime = Some(runtime);
    }

    /// A twin of this Session for the turn task: every conversation handle is
    /// shared (history, queue, cancel, running, tools, hooks, metadata), so the
    /// twin's turn writes the same conversation the held Session reads — the
    /// same Session, in another task. (Not `Clone`: the twin deliberately
    /// carries no task handle, and observers stay with the holder's relay.)
    fn task_twin(&self) -> Agent {
        Agent {
            history: self.history_handle(),
            cwd: self.cwd.clone(),
            queue: self.queue.clone(),
            steering_mode: self.steering_mode,
            follow_up_mode: self.follow_up_mode,
            cancel: self.cancel.clone(),
            running: self.running.clone(),
            run_handle: None,
            observers: Arc::from(Vec::new()),
            tools: self.tools.clone(),
            metadata: self.metadata.clone(),
            hooks: self.hooks.clone(),
            exec_ops: self.exec_ops.clone(),
            fs_ops: self.fs_ops.clone(),
            boundary: self.boundary.clone(),
            model: self.model.clone(),
            runtime: self.runtime.clone(),
            prompt: self.prompt.clone(),
            thinking: self.thinking.clone(),
        }
    }

    /// Send a Turn into the Session: append `input` to the conversation and
    /// drive a full agent turn — Provider and credentials resolved through the
    /// Runtime, rounds streamed, tools dispatched (through Hooks), steering and
    /// follow-ups delivered — emitting the lifecycle Events on the returned
    /// stream. The held Session's history accumulates the turn as it runs; when
    /// the stream ends the Session is idle again. Every failure, setup or
    /// mid-turn, arrives in-channel as an [`Error`](TurnEvent::Error) event.
    pub fn run(
        &mut self,
        input: Input,
    ) -> tokio::sync::mpsc::Receiver<TurnEvent> {
        self.submit(input);
        let mut twin = self.task_twin();
        self.spawn_turn_task(move |tx| async move {
            let result: anyhow::Result<()> = async {
                let runtime = twin.runtime.clone().ok_or_else(|| {
                    anyhow::anyhow!("this session has no runtime")
                })?;
                let model = twin.model.clone().ok_or_else(|| {
                    anyhow::anyhow!("no model selected for this session")
                })?;
                let provider = runtime
                    .find_provider(&model.provider)
                    .ok_or_else(|| {
                        anyhow::anyhow!("unknown provider: {}", model.provider)
                    })?;
                let config = runtime.config();
                let base = crate::providers::base_client(
                    config.http_timeout_secs.unwrap_or(120),
                );
                let creds = provider
                    .creds(&base, runtime.credentials().as_ref())
                    .await?;
                let client = crate::providers::retrying_client(
                    base,
                    config.http_retries.unwrap_or(2),
                );
                let defs = twin.tool_definitions();
                let active: Vec<String> =
                    defs.iter().map(|d| d.name.to_string()).collect();
                // Assemble the prompt from the session's knobs — context files
                // and skills are re-read every turn, matching the TUI.
                let spec = &twin.prompt;
                let config_dir = crate::config::dir();
                let context = if spec.context {
                    crate::context::format_for_prompt(
                        &crate::context::load_with(
                            twin.cwd(),
                            config_dir.as_deref(),
                            spec.trust_project,
                        ),
                    )
                } else {
                    String::new()
                };
                let skills = if spec.skills {
                    crate::skills::format_for_prompt(&crate::skills::load_with(
                        twin.cwd(),
                        config_dir.as_deref(),
                        &spec.skill_paths,
                        true,
                        spec.trust_project,
                    ))
                } else {
                    String::new()
                };
                let append = match &spec.append {
                    Some(lines) => crate::config::merge_append(lines),
                    None => String::new(),
                };
                let instructions = system_prompt(
                    twin.cwd(),
                    &context,
                    &skills,
                    &append,
                    &active,
                    spec.custom.as_deref(),
                );
                let effort = reasoning_effort(
                    twin.thinking.as_deref().unwrap_or("medium"),
                )
                .map(str::to_string);
                let runner = crate::providers::LiveRounds {
                    client,
                    provider,
                    creds,
                    model: model.id,
                    instructions,
                    tools: defs,
                    effort,
                };
                twin.run_turn(&runner, &tx).await;
                Ok(())
            }
            .await;
            if let Err(e) = result {
                let _ =
                    tx.send(TurnEvent::Error { message: e.to_string() }).await;
            }
            // Idle before the sender drops: when the stream ends, the Session
            // already reads as not running — no frontend idle-marking. (The
            // legacy run_with path keeps the frontend-owned set_idle until its
            // callers move onto run(input).)
            twin.running.store(false, Ordering::SeqCst);
        })
    }

    /// The model this session is pointed at, if one has been selected.
    // Session-state surface for the command registry / adapters; the TUI keeps
    // its own selection until it consumes the registry.
    pub fn model(&self) -> Option<&ModelRef> {
        self.model.as_ref()
    }

    /// Point this session at `model` (or clear the selection with `None`).
    pub fn set_model(&mut self, model: Option<ModelRef>) {
        self.model = model;
    }

    /// Attach the session's opaque metadata (set by the Runtime when spawning).
    /// Set the exec operations this session's `bash` calls execute through
    /// (set by the Runtime when spawning the session; defaults to the local
    /// spawn path).
    pub fn set_exec_ops(
        &mut self,
        ops: std::sync::Arc<dyn crate::tools::exec::ExecOps>,
    ) {
        self.exec_ops = ops;
    }

    /// Set the filesystem operations this session's file tools perform
    /// their raw I/O through (set by the Runtime when spawning the session;
    /// defaults to the local `std::fs` path).
    pub fn set_fs_ops(
        &mut self,
        ops: std::sync::Arc<dyn crate::tools::fs::FsOps>,
    ) {
        self.fs_ops = ops;
    }

    /// Install (or clear) the workspace [`PathBoundary`] every tool call of
    /// this session resolves its paths through — the sandbox adapters' jail
    /// (a bot sets the channel's guest↔host workspace mapping here). `None`,
    /// the default, keeps the unrestricted local resolution.
    ///
    /// [`PathBoundary`]: crate::tools::jail::PathBoundary
    pub fn set_boundary(
        &mut self,
        boundary: Option<crate::tools::jail::PathBoundary>,
    ) {
        self.boundary = boundary;
    }

    pub fn set_metadata(
        &mut self,
        metadata: std::sync::Arc<crate::tools::tool::Metadata>,
    ) {
        self.metadata = metadata;
    }

    /// A clone of the metadata handle, for handing to an ephemeral turn agent.
    pub fn metadata_handle(
        &self,
    ) -> std::sync::Arc<crate::tools::tool::Metadata> {
        self.metadata.clone()
    }

    /// The session's working directory — where its tool calls run.
    pub fn cwd(&self) -> &std::path::Path {
        &self.cwd
    }

    /// Attach the observers a turn's events are relayed through. The Runtime sets
    /// these when it spawns the session; a bare `Agent` has none.
    pub fn set_observers(&mut self, observers: Observers) {
        self.observers = observers;
    }

    /// A clone of the observer registry handle (mirrors [`hooks_handle`]) — lets
    /// a builder read the Runtime's observers to extend rather than replace them.
    ///
    /// [`hooks_handle`]: Agent::hooks_handle
    pub fn observers_handle(&self) -> Observers {
        self.observers.clone()
    }

    /// Attach the tool registry the turn loop dispatches through. The Runtime
    /// sets this when it spawns the session.
    pub fn set_tools(&mut self, tools: crate::tools::Tools) {
        self.tools = tools;
    }

    /// The registered tool with this name, if any — how the turn loop dispatches
    /// a call through the [`Tool`](crate::tools::tool::Tool) trait.
    pub fn find_tool(
        &self,
        name: &str,
    ) -> Option<&std::sync::Arc<dyn crate::tools::tool::Tool>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    /// A clone of the tool registry handle, for handing to an ephemeral turn
    /// agent that shares this conversation's wiring (as `spawn_chat` does).
    pub fn tools_handle(&self) -> crate::tools::Tools {
        self.tools.clone()
    }

    /// Attach the interception hooks the turn loop consults around each tool
    /// call. The Runtime sets this when it spawns the session.
    pub fn set_hooks(&mut self, hooks: Hooks) {
        self.hooks = hooks;
    }

    /// A clone of the hook registry handle, for handing to an ephemeral turn
    /// agent that shares this conversation's wiring (as `spawn_chat` does).
    pub fn hooks_handle(&self) -> Hooks {
        self.hooks.clone()
    }

    /// The model-facing definitions of this session's tools — scoped to its
    /// selection, so an adapter advertises only what the session can run. (The
    /// TUI advertises via the CLI tool flags; adapters use this — see
    /// [`crate::runtime::ToolSelection`].)
    pub fn tool_definitions(&self) -> Vec<crate::tools::ToolDef> {
        self.tools
            .iter()
            .map(|t| crate::tools::ToolDef {
                name: t.name(),
                description: t.description(),
                parameters: t.parameters(),
            })
            .collect()
    }

    /// A snapshot of the canonical conversation so far — what a turn is sent
    /// with. Frontends read this instead of keeping their own copy. (A snapshot,
    /// not a borrow: the history is shared with a possibly-running turn task.)
    pub fn history(&self) -> Vec<crate::providers::Step> {
        self.history.lock().unwrap().clone()
    }

    /// The shared conversation handle itself — clones observe each other's
    /// writes (see [`History`]). The seam a spawned turn task writes through.
    pub(crate) fn history_handle(&self) -> History {
        self.history.clone()
    }

    /// Begin (or extend) the user side of the conversation: append `input` as a
    /// user message, merging into the previous step when it is also a user
    /// message — the durable history must alternate roles (Anthropic / Gemini
    /// reject two user messages in a row), and consecutive user inputs (a `!`
    /// command's output, a skill block, a follow-up) coalesce just as they did
    /// when the history was rebuilt from the transcript.
    pub fn submit(&mut self, input: Input) {
        self.push_user(input.model_text, input.images);
    }

    fn push_user(&mut self, text: String, images: Vec<crate::agent::Image>) {
        // Coalesce only when the incoming message carries no images (an image
        // message must stay its own block), matching the old transcript merge.
        let mut history = self.history.lock().unwrap();
        if images.is_empty()
            && let Some(crate::providers::Step::User { text: prev, .. }) =
                history.last_mut()
        {
            prev.push_str("\n\n");
            prev.push_str(&text);
        } else {
            history.push(crate::providers::Step::User { text, images });
        }
    }

    /// Record a finished assistant reply segment in the durable history, merging
    /// into the previous step when it is also assistant text — segments split by
    /// tool calls (which never enter the durable history) coalesce into one
    /// message, exactly as the transcript-derived history did.
    pub fn push_assistant(&mut self, text: &str) {
        let mut history = self.history.lock().unwrap();
        if let Some(crate::providers::Step::Assistant { text: prev, .. }) =
            history.last_mut()
        {
            prev.push_str("\n\n");
            prev.push_str(text);
        } else {
            history.push(crate::providers::Step::Assistant {
                text: text.to_string(),
                thinking: String::new(),
                tool_calls: Vec::new(),
                raw: None,
            });
        }
    }

    /// Clear the conversation history (a new session, or a reset). The queue and
    /// cancel handle belong to the live process, not the conversation, so they
    /// are left untouched.
    pub fn reset_history(&mut self) {
        self.history.lock().unwrap().clear();
    }

    /// Replace the conversation with a single summary user message — the
    /// auto-compaction checkpoint, where everything so far is represented by the
    /// summary and later turns accumulate after it.
    pub fn compact(&mut self, summary: &str) {
        *self.history.lock().unwrap() = vec![crate::providers::Step::User {
            text: format!("Summary of the conversation so far:\n\n{summary}"),
            images: Vec::new(),
        }];
    }

    /// Record that the user interrupted the turn, so the next turn's context
    /// carries the interruption (and the model doesn't silently redo the work).
    pub fn interrupted(&mut self) {
        self.push_assistant(INTERRUPTED);
    }

    /// Queue a steering message: delivered into the running turn after the
    /// current round's tool calls, before the next model round. Takes `&self`
    /// (it only touches the shared queue) so a frontend can steer while a turn
    /// is in flight.
    pub fn steer(&self, input: Input) {
        self.enqueue(crate::queue::Kind::Steer, input);
    }

    /// Queue a follow-up message: delivered once the turn goes idle with no
    /// steering pending, then run as its own round. Like [`steer`](Self::steer),
    /// it only touches the shared queue, so it takes `&self`.
    pub fn follow_up(&self, input: Input) {
        self.enqueue(crate::queue::Kind::FollowUp, input);
    }

    /// A clone of the shared steering / follow-up queue handle, for a frontend
    /// that inspects or drains it (display, restore-to-editor, race-window flush)
    /// outside the engine's own enqueue methods.
    pub fn queue_handle(&self) -> crate::queue::Shared {
        self.queue.clone()
    }

    /// A cloneable handle to cancel this turn from elsewhere — e.g. a frontend
    /// that has moved the Agent into a task and needs to abort it from the UI
    /// thread. Aborting the handle stops the turn at the next round boundary.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Spawn the in-flight turn. `setup` receives the event [`Sender`](tokio::sync::mpsc::Sender)
    /// and does the actual work (resolve credentials, build the provider runner,
    /// drive the round loop); the engine owns the resulting task — its abort
    /// handle and the "running" state — and returns the
    /// [`Receiver`](tokio::sync::mpsc::Receiver) the frontend streams from. This
    /// is the single entry point that starts a turn, so "is a turn running?" and
    /// "abort the turn" become the engine's to answer, not the frontend's.
    // Test-only surface: adapters send Turns with [`run`](Self::run); this
    // low-level spawn entry remains for scripted tests (the engine's own and,
    // via the `test-util` feature, the frontends').
    #[cfg(any(test, feature = "test-util"))]
    pub fn run_with<F, Fut>(
        &mut self,
        setup: F,
    ) -> tokio::sync::mpsc::Receiver<TurnEvent>
    where
        F: FnOnce(tokio::sync::mpsc::Sender<TurnEvent>) -> Fut,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.spawn_turn_task(setup)
    }

    /// Spawn the in-flight turn's task: `setup` receives the event sender and
    /// does the work; the engine owns the task (abort handle, running flag)
    /// and returns the receiver. [`run`](Self::run) is the public face.
    fn spawn_turn_task<F, Fut>(
        &mut self,
        setup: F,
    ) -> tokio::sync::mpsc::Receiver<TurnEvent>
    where
        F: FnOnce(tokio::sync::mpsc::Sender<TurnEvent>) -> Fut,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.cancel.reset();
        self.running.store(true, Ordering::SeqCst);
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        self.run_handle = Some(tokio::spawn(setup(tx)));
        if self.observers.is_empty() {
            // No observers: hand the frontend the engine's stream directly, so a
            // bare session (today's TUI) is exactly as before — no extra hop.
            return rx;
        }
        // Observers registered: relay every event through them, in order, before
        // the frontend sees it. The turn task remains the abort handle; this relay
        // is auxiliary and ends when the turn's sender drops.
        let observers = self.observers.clone();
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                for o in observers.iter() {
                    o.on_event(&ev);
                }
                if out_tx.send(ev).await.is_err() {
                    break;
                }
            }
        });
        out_rx
    }

    /// Whether a turn is in flight.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Mark the turn finished — the frontend calls this when it sees the terminal
    /// `Done` / `Error` (or right after an abort) — dropping the task handle.
    pub fn set_idle(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.run_handle = None;
    }

    /// Abort the in-flight turn: cooperatively (it stops at the next round
    /// boundary) **and** hard (the task is cancelled immediately, mid-round). A
    /// frontend that only holds this engine gets both for free. Safe when idle.
    pub fn abort(&self) {
        self.cancel.abort();
        if let Some(handle) = &self.run_handle {
            handle.abort();
        }
        // A hard-killed task never reaches its own flag-clear.
        self.running.store(false, Ordering::SeqCst);
    }

    /// Test-only: pretend a turn is running (with no real task) so a frontend's
    /// turn handling can be exercised without a live round loop. Exposed to
    /// frontends' tests via the `test-util` feature.
    #[cfg(any(test, feature = "test-util"))]
    pub fn mark_running(&mut self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Push an input onto the shared queue under `kind`.
    fn enqueue(&self, kind: crate::queue::Kind, input: Input) {
        self.queue.lock().unwrap().push(crate::queue::Queued {
            display: input.display,
            model_text: input.model_text,
            images: input.images,
            kind,
        });
    }

    /// Run a turn to completion: stream rounds via `runner`, append the
    /// assistant step each round, and finish when the model stops requesting
    /// tools. Emits `Usage` per round and a terminal `Done`.
    pub async fn run_turn<R: RoundRunner>(
        &mut self,
        runner: &R,
        tx: &tokio::sync::mpsc::Sender<TurnEvent>,
    ) {
        let _ = tx.send(TurnEvent::AgentStart).await;
        // Whether the turn ever produced anything visible (text, thinking, or a
        // tool call). A turn that ends having produced nothing is surfaced as an
        // error instead of a blank reply.
        let mut produced_output = false;
        for _ in 0..MAX_ROUNDS {
            // An abort stops the turn at the round boundary; the in-flight round
            // (if any) has already finished by the time we re-check here.
            if self.cancel.is_aborted() {
                break;
            }
            let _ = tx.send(TurnEvent::TurnStart).await;
            let _ = tx.send(TurnEvent::MessageStart).await;
            let snapshot = self.history();
            let outcome = match runner.run_round(&snapshot, tx).await {
                Ok(outcome) => outcome,
                Err(e) => {
                    let _ =
                        tx.send(TurnEvent::Error { message: e.message }).await;
                    return;
                }
            };
            let _ = tx.send(TurnEvent::MessageEnd).await;
            let _ = tx.send(TurnEvent::Usage { usage: outcome.usage }).await;
            let tool_calls = outcome.tool_calls;
            if !produced_output {
                let said_something = matches!(
                    &outcome.assistant,
                    crate::providers::Step::Assistant { text, thinking, .. }
                        if !text.trim().is_empty() || !thinking.trim().is_empty()
                );
                produced_output = said_something || !tool_calls.is_empty();
            }
            self.history.lock().unwrap().push(outcome.assistant);
            if tool_calls.is_empty() {
                // Idle: deliver steering first, then follow-up; if neither is
                // queued, the turn is truly done.
                let mut delivered = crate::queue::drain(
                    &self.queue,
                    crate::queue::Kind::Steer,
                    self.steering_mode,
                );
                if delivered.is_empty() {
                    delivered = crate::queue::drain(
                        &self.queue,
                        crate::queue::Kind::FollowUp,
                        self.follow_up_mode,
                    );
                }
                if delivered.is_empty() {
                    if !produced_output {
                        let _ = tx
                            .send(TurnEvent::Error {
                                message: "the model returned an empty response (it may have reached the output token limit)".to_string(),
                            })
                            .await;
                        return;
                    }
                    let _ = tx.send(TurnEvent::TurnEnd).await;
                    break;
                }
                self.deliver(tx, delivered).await;
                let _ = tx.send(TurnEvent::TurnEnd).await;
                continue;
            }
            // Cloned once so the hook context and tool context can borrow them
            // while the loop also pushes onto `self.history`.
            let metadata = self.metadata.clone();
            let hooks = self.hooks.clone();
            for call in tool_calls {
                let call_id = call.call_id.clone();
                // Consult the before-tool hooks in registration order. Each sees
                // the model's original call; the last ModifyArgs wins (its args are
                // the ones that run) and the first Deny blocks the call — it does
                // not run, the denial becomes its result. The same seam a bot uses
                // for policy and a TUI for interactive approval.
                let hook_ctx = crate::hook::HookCtx::new(&metadata);
                let mut denied: Option<String> = None;
                let mut effective_args = call.args.clone();
                for h in hooks.iter() {
                    match h.before_tool(&call, &hook_ctx).await {
                        crate::hook::ToolDecision::Allow => {}
                        crate::hook::ToolDecision::ModifyArgs(args) => {
                            effective_args = args
                        }
                        crate::hook::ToolDecision::Deny { reason } => {
                            denied = Some(reason);
                            break;
                        }
                    }
                }
                // The registered Tool provides its own title; an unknown name
                // (a model hallucination) just titles itself. The title reflects
                // the effective (post-hook) arguments — what actually runs.
                let title = match self.find_tool(&call.name) {
                    Some(t) => t.title(&effective_args),
                    None => call.name.clone(),
                };
                let _ = tx
                    .send(TurnEvent::ToolStart {
                        call_id: call_id.clone(),
                        name: call.name.clone(),
                        title,
                    })
                    .await;
                if let Some(reason) = denied {
                    // A denied call never runs; the denial surfaces as its error
                    // result and the (correlated) ToolEnd the frontend sees.
                    let _ = tx
                        .send(TurnEvent::ToolEnd {
                            call_id,
                            output: reason.clone(),
                            is_error: true,
                            took_ms: 0,
                        })
                        .await;
                    self.history.lock().unwrap().push(
                        crate::providers::Step::ToolResult {
                            call_id: call.call_id,
                            name: call.name,
                            output: reason,
                            is_error: true,
                        },
                    );
                    continue;
                }
                let started = std::time::Instant::now();
                // Forward a streaming tool's output as ToolUpdate, tagged with the
                // call's id, while it runs. The sink drops when the tool returns,
                // ending the forwarder; we join it before ToolEnd so every update
                // is delivered first.
                let (progress, mut updates) =
                    tokio::sync::mpsc::channel::<String>(64);
                let forward = {
                    let tx = tx.clone();
                    let call_id = call_id.clone();
                    tokio::spawn(async move {
                        while let Some(delta) = updates.recv().await {
                            let _ = tx
                                .send(TurnEvent::ToolUpdate {
                                    call_id: call_id.clone(),
                                    delta,
                                })
                                .await;
                        }
                    })
                };
                // Dispatch through the registered Tool trait, falling back to the
                // built-in name dispatch for tools not yet converted.
                let mut ctx = crate::tools::tool::ToolCtx::new(
                    self.cwd.clone(),
                    self.cancel.clone(),
                    Some(progress),
                )
                .with_metadata(metadata.clone())
                .with_exec_ops(self.exec_ops.clone())
                .with_fs_ops(self.fs_ops.clone());
                // The session's workspace boundary, when set, jails every
                // path this call resolves (sandboxed adapters).
                if let Some(boundary) = &self.boundary {
                    ctx = ctx.with_boundary(boundary.clone());
                }
                let (model_text, display, is_error) =
                    match crate::tools::dispatch(
                        &self.tools,
                        &call.name,
                        &effective_args,
                        &ctx,
                    )
                    .await
                    {
                        Ok(r) => (r.model_text, r.display, false),
                        Err(e) => (e.to_string(), e.to_string(), true),
                    };
                // Drop the context (and its progress sender) so the forwarder
                // sees the channel close and ends — otherwise the await below
                // would hang waiting on a sender we still hold.
                drop(ctx);
                let _ = forward.await;
                let took_ms = started.elapsed().as_millis() as u64;
                // Consult the after-tool hooks; the last Some replaces the result
                // the model sees (and the transcript shows). The call's error state
                // is preserved across an override. Lets a hook redact or
                // post-process tool output.
                let mut result =
                    crate::tools::ToolResult { model_text, display };
                for h in hooks.iter() {
                    if let Some(replacement) =
                        h.after_tool(&call, &result, &hook_ctx).await
                    {
                        result = replacement;
                    }
                }
                let crate::tools::ToolResult { model_text, display } = result;
                let _ = tx
                    .send(TurnEvent::ToolEnd {
                        call_id,
                        output: display,
                        is_error,
                        took_ms,
                    })
                    .await;
                self.history.lock().unwrap().push(
                    crate::providers::Step::ToolResult {
                        call_id: call.call_id,
                        name: call.name,
                        output: model_text,
                        is_error,
                    },
                );
            }
            // Steering messages join the turn after its tool calls.
            let steers = crate::queue::drain(
                &self.queue,
                crate::queue::Kind::Steer,
                self.steering_mode,
            );
            self.deliver(tx, steers).await;
            let _ = tx.send(TurnEvent::TurnEnd).await;
        }
        let _ = tx.send(TurnEvent::Done).await;
        let _ = tx.send(TurnEvent::AgentEnd).await;
    }

    /// Emit each queued message as a `QueuedDelivered` event (so the frontend
    /// shows it as a user turn) and append it to the history so the next round
    /// sees it.
    async fn deliver(
        &mut self,
        tx: &tokio::sync::mpsc::Sender<TurnEvent>,
        messages: Vec<crate::queue::Queued>,
    ) {
        for m in messages {
            let _ = tx
                .send(TurnEvent::QueuedDelivered {
                    display: m.display,
                    model_text: m.model_text.clone(),
                    images: m.images.clone(),
                })
                .await;
            self.history.lock().unwrap().push(crate::providers::Step::User {
                text: m.model_text,
                images: m.images,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_assembles_all_sections_in_order() {
        let all: Vec<String> =
            crate::tools::names().iter().map(|s| s.to_string()).collect();
        let p = system_prompt(
            std::path::Path::new("/tmp/proj"),
            "<project_context>X</project_context>",
            "<available_skills>S</available_skills>",
            "APPENDED RULES",
            &all,
            None,
        );
        assert!(p.contains("<available_skills>S</available_skills>"));
        // Tools list + guidelines from the tool registry.
        assert!(p.contains("Available tools:"));
        assert!(p.contains("- read: Read file contents"));
        assert!(p.contains("Guidelines:"));
        assert!(p.contains("Be concise in your responses"));
        assert!(p.contains("Use read to examine files instead of cat or sed."));
        // Append, context, and the date/cwd tail.
        assert!(p.contains("APPENDED RULES"));
        assert!(p.contains("<project_context>X</project_context>"));
        assert!(p.contains("/tmp/proj"));
        // Order: tools → append → context → date/cwd tail.
        let tools_at = p.find("Available tools:").unwrap();
        let append_at = p.find("APPENDED RULES").unwrap();
        let ctx_at = p.find("<project_context>").unwrap();
        let date_at = p.find("Current date and time:").unwrap();
        assert!(tools_at < append_at && append_at < ctx_at && ctx_at < date_at);
    }

    #[test]
    fn custom_prompt_replaces_body_but_keeps_context_and_skills() {
        let all: Vec<String> =
            crate::tools::names().iter().map(|s| s.to_string()).collect();
        let p = system_prompt(
            std::path::Path::new("/tmp/proj"),
            "<project_context>X</project_context>",
            "<available_skills>S</available_skills>",
            "EXTRA",
            &all,
            Some("CUSTOM ROOT PROMPT"),
        );
        assert!(p.starts_with("CUSTOM ROOT PROMPT"));
        // The default body is gone…
        assert!(!p.contains("You are tapir"));
        assert!(!p.contains("Available tools:"));
        // …but the append, context, skills, and date/cwd tail remain.
        assert!(p.contains("EXTRA"));
        assert!(p.contains("<project_context>X</project_context>"));
        assert!(p.contains("<available_skills>S</available_skills>"));
        assert!(p.contains("Current date and time:"));
    }

    #[test]
    fn skills_are_omitted_when_read_tool_is_disabled() {
        // Without `read`, a skill can't be loaded, so it isn't listed.
        let p = system_prompt(
            std::path::Path::new("/p"),
            "",
            "<available_skills>S</available_skills>",
            "",
            &["bash".to_string()],
            None,
        );
        assert!(!p.contains("<available_skills>"));
    }

    #[test]
    fn effort_maps_levels() {
        assert_eq!(reasoning_effort("off"), None);
        assert_eq!(reasoning_effort("minimal"), Some("low"));
        assert_eq!(reasoning_effort("medium"), Some("medium"));
        assert_eq!(reasoning_effort("xhigh"), Some("xhigh"));
    }

    // --- Agent turn loop ---------------------------------------------------

    use crate::providers::{RoundOutcome, Step};
    use tokio::sync::mpsc;

    /// A scripted round: text deltas plus the outcome to return, or a terminal
    /// error. Lets a turn be driven with no network.
    enum Scripted {
        Ok { deltas: Vec<String>, outcome: RoundOutcome },
        Err(RoundError),
    }

    /// A [`RoundRunner`] that replays pre-scripted rounds in order — the test
    /// seam standing in for a real provider.
    struct ScriptedRunner {
        rounds: std::sync::Mutex<std::collections::VecDeque<Scripted>>,
    }

    impl ScriptedRunner {
        fn new(rounds: Vec<Scripted>) -> Self {
            Self { rounds: std::sync::Mutex::new(rounds.into()) }
        }
    }

    impl RoundRunner for ScriptedRunner {
        async fn run_round(
            &self,
            _history: &[Step],
            tx: &mpsc::Sender<TurnEvent>,
        ) -> Result<RoundOutcome, RoundError> {
            // Pop before any await so the lock guard isn't held across it.
            let next = self
                .rounds
                .lock()
                .unwrap()
                .pop_front()
                .expect("ScriptedRunner ran out of rounds");
            match next {
                Scripted::Ok { deltas, outcome } => {
                    for d in deltas {
                        let _ = tx.send(TurnEvent::Text { delta: d }).await;
                    }
                    Ok(outcome)
                }
                Scripted::Err(e) => Err(e),
            }
        }
    }

    fn text_round(text: &str) -> RoundOutcome {
        RoundOutcome {
            usage: Usage::default(),
            tool_calls: Vec::new(),
            assistant: Step::Assistant {
                text: text.to_string(),
                thinking: String::new(),
                tool_calls: Vec::new(),
                raw: None,
            },
        }
    }

    /// Collect every event of a turn. Bounded by a low per-event timeout: a
    /// stuck turn (e.g. a channel that never closes — a deadlock) ends the drain
    /// instead of hanging `cargo test` forever, so the test fails fast and names
    /// itself. Scripted turns emit events in microseconds, so 10s never trips on
    /// a healthy run.
    async fn drain(rx: &mut mpsc::Receiver<TurnEvent>) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        let cap = std::time::Duration::from_secs(10);
        while let Ok(Some(ev)) = tokio::time::timeout(cap, rx.recv()).await {
            out.push(ev);
        }
        out
    }

    /// An Agent with an empty queue and default (one-at-a-time) delivery, plus
    /// the built-in tool registry a real session carries — the setup for tests
    /// that don't exercise steering / follow-up.
    fn plain_agent(history: Vec<Step>, cwd: std::path::PathBuf) -> Agent {
        let mut agent = Agent::new(
            history,
            cwd,
            crate::queue::new(),
            crate::queue::Mode::OneAtATime,
            crate::queue::Mode::OneAtATime,
        );
        agent.set_tools(std::sync::Arc::from(crate::tools::builtin_tools()));
        agent
    }

    #[tokio::test]
    async fn text_only_round_emits_text_then_usage_then_done() {
        let runner = ScriptedRunner::new(vec![Scripted::Ok {
            deltas: vec!["hi".into()],
            outcome: text_round("hi"),
        }]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "hello".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(
                |e| matches!(e, TurnEvent::Text { delta } if delta == "hi")
            ),
            "the streamed text is emitted",
        );
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Usage { .. })),
            "the round's usage should be emitted",
        );
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "the turn should end with Done",
        );
        // The assistant turn is appended to the conversation.
        assert_eq!(agent.history().len(), 2);
    }

    #[tokio::test]
    async fn a_run_is_bracketed_by_agent_start_and_end() {
        // The lifecycle vocabulary brackets the whole run.
        let runner = ScriptedRunner::new(vec![Scripted::Ok {
            deltas: vec!["hi".into()],
            outcome: text_round("hi"),
        }]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "hello".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            matches!(events.first(), Some(TurnEvent::AgentStart)),
            "a run opens with AgentStart",
        );
        assert!(
            matches!(events.last(), Some(TurnEvent::AgentEnd)),
            "a run closes with AgentEnd",
        );
    }

    #[tokio::test]
    async fn round_error_is_emitted_and_stops_the_turn() {
        let runner = ScriptedRunner::new(vec![Scripted::Err(RoundError {
            message: "boom".into(),
        })]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "hi".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            matches!(events.last(), Some(TurnEvent::Error { message }) if message == "boom"),
            "a failed round should end the turn with its Error message",
        );
        assert!(
            !events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "no Done should follow an error",
        );
    }

    fn tool_round(name: &str, args: serde_json::Value) -> RoundOutcome {
        RoundOutcome {
            usage: Usage::default(),
            tool_calls: vec![ToolCall {
                call_id: "call-1".into(),
                name: name.into(),
                args,
            }],
            assistant: Step::Assistant {
                text: String::new(),
                thinking: String::new(),
                tool_calls: Vec::new(),
                raw: None,
            },
        }
    }

    #[tokio::test]
    async fn tool_call_round_runs_the_tool_then_records_its_result() {
        use serde_json::json;
        // A real directory for `ls` to list, so the loop exercises a real tool.
        let dir = std::env::temp_dir()
            .join(format!("tapir-agent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.txt"), "x").unwrap();

        // Round 1 asks to `ls`; round 2 replies with text and ends the turn.
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("ls", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["done".into()],
                outcome: text_round("done"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "list".into(), images: Vec::new() }],
            dir.clone(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::ToolStart { name, .. } if name == "ls")),
            "the loop should announce the tool call",
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TurnEvent::ToolEnd { is_error, .. } if !is_error)),
            "the tool should run and report a (non-error) end",
        );
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "the turn ends once the model stops requesting tools",
        );
        // The tool result is recorded in the conversation for the next round.
        assert!(
            agent.history().iter().any(|s| matches!(
                s,
                Step::ToolResult { name, is_error, .. } if name == "ls" && !is_error
            )),
            "a ToolResult step should be appended",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn tool_start_and_end_share_a_call_id() {
        use serde_json::json;
        let dir = std::env::temp_dir()
            .join(format!("tapir-callid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("ls", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["done".into()],
                outcome: text_round("done"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "list".into(), images: Vec::new() }],
            dir.clone(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let start_id = events.iter().find_map(|e| match e {
            TurnEvent::ToolStart { call_id, .. } => Some(call_id.clone()),
            _ => None,
        });
        let end_id = events.iter().find_map(|e| match e {
            TurnEvent::ToolEnd { call_id, .. } => Some(call_id.clone()),
            _ => None,
        });
        assert!(
            start_id.is_some(),
            "the tool call announced a start with a call_id"
        );
        assert_eq!(
            start_id, end_id,
            "a tool's start and end carry the same call_id"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn each_round_is_bracketed_by_turn_start_and_end() {
        use serde_json::json;
        // A turn is one assistant response plus any tool calls:
        // here two rounds — a tool call, then a text reply — so two Turns.
        let dir = std::env::temp_dir()
            .join(format!("tapir-turn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("ls", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["done".into()],
                outcome: text_round("done"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "list".into(), images: Vec::new() }],
            dir.clone(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let starts =
            events.iter().filter(|e| matches!(e, TurnEvent::TurnStart)).count();
        let ends =
            events.iter().filter(|e| matches!(e, TurnEvent::TurnEnd)).count();
        assert_eq!(starts, 2, "one TurnStart per round — two rounds here");
        assert_eq!(ends, 2, "one TurnEnd per round");

        // The Turns nest inside the run: the first TurnStart follows AgentStart,
        // the last TurnEnd precedes AgentEnd.
        let first_turn = events
            .iter()
            .position(|e| matches!(e, TurnEvent::TurnStart))
            .unwrap();
        let last_end = events
            .iter()
            .rposition(|e| matches!(e, TurnEvent::TurnEnd))
            .unwrap();
        assert!(
            matches!(events.first(), Some(TurnEvent::AgentStart))
                && first_turn > 0
        );
        assert!(
            matches!(events.last(), Some(TurnEvent::AgentEnd))
                && last_end < events.len() - 1
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn assistant_deltas_are_bracketed_by_message_start_and_end() {
        // The streamed text/thinking deltas of a round are the assistant's
        // message; MessageStart/MessageEnd bracket them.
        let runner = ScriptedRunner::new(vec![Scripted::Ok {
            deltas: vec!["hel".into(), "lo".into()],
            outcome: text_round("hello"),
        }]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "hi".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let start = events
            .iter()
            .position(|e| matches!(e, TurnEvent::MessageStart))
            .expect("the assistant message opens with MessageStart");
        let end = events
            .iter()
            .position(|e| matches!(e, TurnEvent::MessageEnd))
            .expect("the assistant message closes with MessageEnd");
        let first_text = events
            .iter()
            .position(|e| matches!(e, TurnEvent::Text { .. }))
            .expect("a text delta");
        let last_text = events
            .iter()
            .rposition(|e| matches!(e, TurnEvent::Text { .. }))
            .unwrap();
        assert!(
            start < first_text,
            "MessageStart precedes the first text delta"
        );
        assert!(last_text < end, "MessageEnd follows the last text delta");
    }

    #[tokio::test]
    async fn the_seam_streams_an_ordered_event_sequence_with_call_ids() {
        // The whole pipeline end-to-end: a Runtime with an Observer registered,
        // spawning a Session whose turn is driven by a mock Provider (the scripted
        // runner). The Observer must see the full lifecycle vocabulary in order,
        // with each tool event tagged by its correlation id — the contract a Hook
        // or bot adapter relies on.
        use crate::runtime::Runtime;
        use serde_json::json;

        #[derive(Default)]
        struct SeamRecorder(std::sync::Mutex<Vec<String>>);
        impl crate::observer::Observer for SeamRecorder {
            fn on_event(&self, event: &TurnEvent) {
                self.0.lock().unwrap().push(label(event));
            }
        }
        fn label(e: &TurnEvent) -> String {
            match e {
                TurnEvent::AgentStart => "agent_start".into(),
                TurnEvent::AgentEnd => "agent_end".into(),
                TurnEvent::TurnStart => "turn_start".into(),
                TurnEvent::TurnEnd => "turn_end".into(),
                TurnEvent::MessageStart => "message_start".into(),
                TurnEvent::MessageEnd => "message_end".into(),
                TurnEvent::Text { delta } => format!("text:{delta}"),
                TurnEvent::Thinking { .. } => "thinking".into(),
                TurnEvent::ToolStart { call_id, name, .. } => {
                    format!("tool_start:{name}:{call_id}")
                }
                TurnEvent::ToolUpdate { call_id, .. } => {
                    format!("tool_update:{call_id}")
                }
                TurnEvent::ToolEnd { call_id, is_error, .. } => {
                    format!(
                        "tool_end:{call_id}:{}",
                        if *is_error { "err" } else { "ok" }
                    )
                }
                TurnEvent::Usage { .. } => "usage".into(),
                TurnEvent::QueuedDelivered { .. } => "queued".into(),
                TurnEvent::Done => "done".into(),
                TurnEvent::Compacted { .. } => "compacted".into(),
                TurnEvent::Error { .. } => "error".into(),
            }
        }

        let dir = std::env::temp_dir()
            .join(format!("tapir-seam-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let recorder = std::sync::Arc::new(SeamRecorder::default());
        let rt = Runtime::builder().observer(recorder.clone()).build();
        let mut session = rt.session(dir.clone());

        // The mock Provider: a tool round (ls), then a text reply that ends it.
        let cwd = dir.clone();
        let mut rx = session.run_with(move |tx| async move {
            let runner = ScriptedRunner::new(vec![
                Scripted::Ok {
                    deltas: vec![],
                    outcome: tool_round("ls", json!({})),
                },
                Scripted::Ok {
                    deltas: vec!["done".into()],
                    outcome: text_round("done"),
                },
            ]);
            let mut agent = plain_agent(
                vec![Step::User { text: "list".into(), images: Vec::new() }],
                cwd,
            );
            agent.run_turn(&runner, &tx).await;
        });
        let _ = drain(&mut rx).await;

        assert_eq!(
            recorder.0.lock().unwrap().clone(),
            vec![
                "agent_start",
                // Round 1 — the tool call, tagged with its call_id.
                "turn_start",
                "message_start",
                "message_end",
                "usage",
                "tool_start:ls:call-1",
                "tool_end:call-1:ok",
                "turn_end",
                // Round 2 — the text reply.
                "turn_start",
                "message_start",
                "text:done",
                "message_end",
                "usage",
                "turn_end",
                "done",
                "agent_end",
            ],
            "the seam streams the full lifecycle in order, tools tagged by call_id",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_streaming_tool_emits_tool_update_deltas() {
        use serde_json::json;
        // `bash` streams its output as it runs: the loop forwards each chunk as a
        // ToolUpdate tagged with the call's id, before the terminal ToolEnd.
        let dir = std::env::temp_dir()
            .join(format!("tapir-stream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round(
                    "bash",
                    json!({ "command": "printf 'one\\ntwo\\nthree\\n'" }),
                ),
            },
            Scripted::Ok {
                deltas: vec!["done".into()],
                outcome: text_round("done"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "run it".into(), images: Vec::new() }],
            dir.clone(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let updates: Vec<(String, String)> = events
            .iter()
            .filter_map(|e| match e {
                TurnEvent::ToolUpdate { call_id, delta } => {
                    Some((call_id.clone(), delta.clone()))
                }
                _ => None,
            })
            .collect();
        assert!(
            !updates.is_empty(),
            "the streaming tool emitted at least one update"
        );
        assert!(
            updates.iter().all(|(id, _)| id == "call-1"),
            "every update is tagged with the tool call's id",
        );
        let streamed: String =
            updates.iter().map(|(_, d)| d.as_str()).collect();
        assert!(
            streamed.contains("one")
                && streamed.contains("two")
                && streamed.contains("three"),
            "the streamed deltas reconstruct the command output, got: {streamed:?}",
        );
        // Updates precede the tool's end.
        let first_update = events
            .iter()
            .position(|e| matches!(e, TurnEvent::ToolUpdate { .. }))
            .unwrap();
        let tool_end = events
            .iter()
            .position(|e| matches!(e, TurnEvent::ToolEnd { .. }))
            .unwrap();
        assert!(first_update < tool_end, "updates stream before the ToolEnd");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_tool_sees_the_session_metadata_via_its_context() {
        use serde_json::{Value, json};
        // A tool that echoes a value from the session metadata its context carries.
        struct MetaEcho;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for MetaEcho {
            fn name(&self) -> &'static str {
                "meta"
            }
            fn description(&self) -> &'static str {
                "echoes metadata"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                ctx: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text(
                    ctx.metadata().get("user").cloned().unwrap_or_default(),
                ))
            }
        }

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(MetaEcho))
            .build();
        let mut agent = rt.session_with(
            crate::runtime::SessionOptions::new(".".into())
                .metadata([("user", "u42")]),
        );
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("meta", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let out = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the tool ran");
        assert_eq!(
            out, "u42",
            "the tool read the session metadata through ctx.metadata()"
        );
    }

    #[tokio::test]
    async fn run_turn_dispatches_read_through_the_registered_tool() {
        use serde_json::json;
        // There is no name-dispatch fallback: without registration, even a
        // built-in name is unknown (distinct from a read that fails on a file).
        let empty: crate::tools::Tools = std::sync::Arc::from(Vec::new());
        let bare = crate::tools::tool::ToolCtx::new(
            ".".into(),
            CancelToken::default(),
            None,
        );
        let err = crate::tools::dispatch(
            &empty,
            "read",
            &json!({ "path": "x" }),
            &bare,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("unknown tool"),
            "an unregistered tool must be unknown, got: {err}",
        );

        // ...yet a Runtime session still runs a read call — through the Tool trait.
        let dir = std::env::temp_dir()
            .join(format!("tapir-readdispatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("note.txt"), "trace bullet").unwrap();

        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("read", json!({ "path": "note.txt" })),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let mut agent =
            crate::runtime::Runtime::builder().build().session(dir.clone());
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let start_id = events.iter().find_map(|e| match e {
            TurnEvent::ToolStart { call_id, .. } => Some(call_id.clone()),
            _ => None,
        });
        let end = events.iter().find_map(|e| match e {
            TurnEvent::ToolEnd { call_id, output, .. } => {
                Some((call_id.clone(), output.clone()))
            }
            _ => None,
        });
        let (end_id, output) = end.expect("the read call ended");
        assert_eq!(
            start_id.as_deref(),
            Some(end_id.as_str()),
            "start and end share a call_id"
        );
        assert!(
            output.contains("trace bullet"),
            "read ran via the trait and returned the file"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_trait_tool_streams_partials_as_tool_updates() {
        use serde_json::{Value, json};
        // A tool that streams two partials through the context's update sink; the
        // turn loop surfaces them as ToolUpdate events tagged with the call id.
        struct Streamer;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for Streamer {
            fn name(&self) -> &'static str {
                "streamer"
            }
            fn description(&self) -> &'static str {
                "streams"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                ctx: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                ctx.update("chunk-a").await;
                ctx.update("chunk-b").await;
                Ok(crate::tools::ToolResult::text("final"))
            }
        }

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(Streamer))
            .build();
        let mut agent = rt.session(".".into());
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("streamer", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let updates: Vec<(String, String)> = events
            .iter()
            .filter_map(|e| match e {
                TurnEvent::ToolUpdate { call_id, delta } => {
                    Some((call_id.clone(), delta.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            updates,
            vec![
                ("call-1".into(), "chunk-a".into()),
                ("call-1".into(), "chunk-b".into())
            ],
            "the partials surface as ToolUpdate, tagged with the call id",
        );
    }

    #[tokio::test]
    async fn the_read_tool_seam_runs_through_a_session_with_correlated_events()
    {
        use serde_json::json;
        // The full seam, production-shaped: a Runtime session whose registry is
        // carried onto the ephemeral agent that runs the turn (as spawn_chat
        // does). A mock Provider requests read; the Observer must see the tool
        // run with start/end events sharing a call id.
        #[derive(Default)]
        struct ToolRecorder(std::sync::Mutex<Vec<String>>);
        impl crate::observer::Observer for ToolRecorder {
            fn on_event(&self, e: &TurnEvent) {
                let label = match e {
                    TurnEvent::ToolStart { call_id, name, .. } => {
                        format!("start:{name}:{call_id}")
                    }
                    TurnEvent::ToolEnd { call_id, output, .. } => {
                        format!("end:{call_id}:{output}")
                    }
                    _ => return,
                };
                self.0.lock().unwrap().push(label);
            }
        }

        let dir = std::env::temp_dir()
            .join(format!("tapir-readseam-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("doc.txt"), "seam content").unwrap();

        let recorder = std::sync::Arc::new(ToolRecorder::default());
        let rt = crate::runtime::Runtime::builder()
            .observer(recorder.clone())
            .build();
        let mut session = rt.session(dir.clone());
        // Carry the held session's tool registry onto the ephemeral turn agent.
        let tools = session.tools_handle();
        let cwd = dir.clone();
        let mut rx = session.run_with(move |tx| async move {
            let runner = ScriptedRunner::new(vec![
                Scripted::Ok {
                    deltas: vec![],
                    outcome: tool_round("read", json!({ "path": "doc.txt" })),
                },
                Scripted::Ok {
                    deltas: vec!["ok".into()],
                    outcome: text_round("ok"),
                },
            ]);
            let mut agent = plain_agent(
                vec![Step::User { text: "read it".into(), images: Vec::new() }],
                cwd,
            );
            agent.set_tools(tools);
            agent.run_turn(&runner, &tx).await;
        });
        let _ = drain(&mut rx).await;

        let seen = recorder.0.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "exactly a start and an end, got: {seen:?}");
        assert_eq!(seen[0], "start:read:call-1", "the read tool call started");
        assert_eq!(
            seen[1], "end:call-1:seam content",
            "read ran through the trait and returned the file, ended with the same call id",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn the_ephemeral_turn_agent_inherits_the_session_metadata() {
        use serde_json::{Value, json};
        // Mirror spawn_chat: the turn runs on an ephemeral agent built from the
        // held session's cwd, tools, and metadata. A tool must see the metadata.
        struct MetaEcho;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for MetaEcho {
            fn name(&self) -> &'static str {
                "meta"
            }
            fn description(&self) -> &'static str {
                "echoes metadata"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                ctx: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text(
                    ctx.metadata().get("user").cloned().unwrap_or_default(),
                ))
            }
        }
        #[derive(Default)]
        struct Rec(std::sync::Mutex<Vec<String>>);
        impl crate::observer::Observer for Rec {
            fn on_event(&self, e: &TurnEvent) {
                if let TurnEvent::ToolEnd { output, .. } = e {
                    self.0.lock().unwrap().push(output.clone());
                }
            }
        }

        let dir = std::env::temp_dir()
            .join(format!("tapir-metaseam-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let rec = std::sync::Arc::new(Rec::default());
        let rt = crate::runtime::Runtime::builder()
            .observer(rec.clone())
            .tool(std::sync::Arc::new(MetaEcho))
            .build();
        let mut session = rt.session_with(
            crate::runtime::SessionOptions::new(dir.clone())
                .metadata([("user", "u42")]),
        );
        // Carry the session's cwd + tools + metadata onto the ephemeral.
        let cwd = session.cwd().to_path_buf();
        let tools = session.tools_handle();
        let metadata = session.metadata_handle();
        let mut rx = session.run_with(move |tx| async move {
            let runner = ScriptedRunner::new(vec![
                Scripted::Ok {
                    deltas: vec![],
                    outcome: tool_round("meta", json!({})),
                },
                Scripted::Ok {
                    deltas: vec!["ok".into()],
                    outcome: text_round("ok"),
                },
            ]);
            let mut agent = plain_agent(
                vec![Step::User { text: "go".into(), images: Vec::new() }],
                cwd,
            );
            agent.set_tools(tools);
            agent.set_metadata(metadata);
            agent.run_turn(&runner, &tx).await;
        });
        let _ = drain(&mut rx).await;

        assert_eq!(
            rec.0.lock().unwrap().clone(),
            vec!["u42".to_string()],
            "the ephemeral turn agent inherited the session metadata",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_before_tool_hook_can_deny_a_call_so_it_never_runs() {
        use serde_json::{Value, json};
        use std::sync::atomic::{AtomicUsize, Ordering};
        // A recording tool that counts how often it runs, and a hook that denies
        // every call. The denied call must not reach the tool, and the denial must
        // surface as the tool's error result and ToolEnd event.
        struct RecordingTool {
            ran: std::sync::Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for RecordingTool {
            fn name(&self) -> &'static str {
                "rec"
            }
            fn description(&self) -> &'static str {
                "records that it ran"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                _: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                self.ran.fetch_add(1, Ordering::SeqCst);
                Ok(crate::tools::ToolResult::text("ran"))
            }
        }

        struct DenyHook;
        #[async_trait::async_trait]
        impl crate::hook::Hook for DenyHook {
            async fn before_tool(
                &self,
                _: &ToolCall,
                _: &crate::hook::HookCtx<'_>,
            ) -> crate::hook::ToolDecision {
                crate::hook::ToolDecision::Deny {
                    reason: "blocked by policy".into(),
                }
            }
        }

        let ran = std::sync::Arc::new(AtomicUsize::new(0));
        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(RecordingTool { ran: ran.clone() }))
            .hook(std::sync::Arc::new(DenyHook))
            .build();
        let mut agent = rt.session(".".into());
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("rec", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "a denied tool call never runs"
        );
        let end = events.iter().find_map(|e| match e {
            TurnEvent::ToolEnd { output, is_error, .. } => {
                Some((output.clone(), *is_error))
            }
            _ => None,
        });
        let (output, is_error) =
            end.expect("a denied call still emits a ToolEnd");
        assert!(is_error, "the denial is an error result");
        assert!(
            output.contains("blocked by policy"),
            "the denial reason surfaces, got: {output}"
        );
        // The model sees the denial too — it's recorded as the tool's result step.
        assert!(
            agent.history().iter().any(|s| matches!(
                s,
                Step::ToolResult { name, is_error, output, .. }
                    if name == "rec" && *is_error && output.contains("blocked by policy")
            )),
            "the denial is recorded as the tool result the model sees",
        );
    }

    #[tokio::test]
    async fn a_before_tool_hook_can_modify_the_arguments_actually_run() {
        use serde_json::{Value, json};
        // A tool that echoes the arguments it was handed, and a hook that rewrites
        // them. The echoed result must reflect the modified arguments, proving the
        // modified ones are what actually executed.
        struct ArgEcho;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for ArgEcho {
            fn name(&self) -> &'static str {
                "echo"
            }
            fn description(&self) -> &'static str {
                "echoes its arguments"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                args: &Value,
                _: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text(args.to_string()))
            }
        }

        struct ModifyHook;
        #[async_trait::async_trait]
        impl crate::hook::Hook for ModifyHook {
            async fn before_tool(
                &self,
                _: &ToolCall,
                _: &crate::hook::HookCtx<'_>,
            ) -> crate::hook::ToolDecision {
                crate::hook::ToolDecision::ModifyArgs(
                    json!({ "v": "modified" }),
                )
            }
        }

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(ArgEcho))
            .hook(std::sync::Arc::new(ModifyHook))
            .build();
        let mut agent = rt.session(".".into());
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("echo", json!({ "v": "original" })),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let output = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the tool ran");
        assert!(
            output.contains("modified"),
            "the modified arguments are executed, got: {output}"
        );
        assert!(
            !output.contains("original"),
            "the original arguments are not executed, got: {output}"
        );
    }

    #[tokio::test]
    async fn an_after_tool_hook_can_replace_the_result_the_model_sees() {
        use serde_json::{Value, json};
        // A tool produces raw output; an after-tool hook replaces it. Both the
        // ToolEnd event and the recorded ToolResult (what the model sees next
        // round) must carry the replacement, not the raw output.
        struct RawTool;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for RawTool {
            fn name(&self) -> &'static str {
                "raw"
            }
            fn description(&self) -> &'static str {
                "produces raw output"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                _: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text("raw secret output"))
            }
        }

        struct RedactHook;
        #[async_trait::async_trait]
        impl crate::hook::Hook for RedactHook {
            async fn after_tool(
                &self,
                _: &ToolCall,
                _: &crate::tools::ToolResult,
                _: &crate::hook::HookCtx<'_>,
            ) -> Option<crate::tools::ToolResult> {
                Some(crate::tools::ToolResult::text("[redacted]"))
            }
        }

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(RawTool))
            .hook(std::sync::Arc::new(RedactHook))
            .build();
        let mut agent = rt.session(".".into());
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("raw", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let output = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the tool ran");
        assert_eq!(
            output, "[redacted]",
            "the ToolEnd shows the replacement, not the raw output"
        );
        let recorded = agent
            .history()
            .iter()
            .find_map(|s| match s {
                Step::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the tool result was recorded");
        assert_eq!(
            recorded, "[redacted]",
            "the model sees the replacement next round"
        );
        assert!(
            !recorded.contains("raw secret"),
            "the raw output never reaches the model, got: {recorded}",
        );
    }

    #[tokio::test]
    async fn a_hook_reads_the_session_metadata_to_decide() {
        use serde_json::{Value, json};
        // A policy hook that keys its verdict off the session metadata: it denies
        // and names the conversation's user. The denial text proves the hook read
        // the per-session metadata through its HookCtx.
        struct NoopTool;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for NoopTool {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn description(&self) -> &'static str {
                "does nothing"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                _: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text("ran"))
            }
        }

        struct MetaGate;
        #[async_trait::async_trait]
        impl crate::hook::Hook for MetaGate {
            async fn before_tool(
                &self,
                _: &ToolCall,
                ctx: &crate::hook::HookCtx<'_>,
            ) -> crate::hook::ToolDecision {
                let who =
                    ctx.metadata().get("user").cloned().unwrap_or_default();
                crate::hook::ToolDecision::Deny {
                    reason: format!("denied for {who}"),
                }
            }
        }

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(NoopTool))
            .hook(std::sync::Arc::new(MetaGate))
            .build();
        let mut agent = rt.session_with(
            crate::runtime::SessionOptions::new(".".into())
                .metadata([("user", "u42")]),
        );
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("noop", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let output = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("a denied call still ends");
        assert_eq!(
            output, "denied for u42",
            "the hook read the session metadata through its HookCtx to decide",
        );
    }

    #[tokio::test]
    async fn an_all_default_hook_is_transparent_to_tool_results() {
        use serde_json::{Value, json};
        // A registered hook that overrides nothing (both methods default: Allow,
        // no replacement) must not disturb a tool call — it runs and its real
        // result reaches the model unchanged. Guards the seam against silently
        // clobbering output when a hook declines to act.
        struct RealTool;
        #[async_trait::async_trait]
        impl crate::tools::tool::Tool for RealTool {
            fn name(&self) -> &'static str {
                "real"
            }
            fn description(&self) -> &'static str {
                "returns its real output"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            async fn run(
                &self,
                _: &Value,
                _: &crate::tools::tool::ToolCtx,
            ) -> anyhow::Result<crate::tools::ToolResult> {
                Ok(crate::tools::ToolResult::text("the real result"))
            }
        }

        // All-default Hook: allows every call, replaces no result.
        struct PassThrough;
        impl crate::hook::Hook for PassThrough {}

        let rt = crate::runtime::Runtime::builder()
            .tool(std::sync::Arc::new(RealTool))
            .hook(std::sync::Arc::new(PassThrough))
            .build();
        let mut agent = rt.session(".".into());
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("real", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let (output, is_error) = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, is_error, .. } => {
                    Some((output.clone(), *is_error))
                }
                _ => None,
            })
            .expect("the tool ran");
        assert!(!is_error, "an allowed call runs normally");
        assert_eq!(
            output, "the real result",
            "the tool's real result passes through unchanged"
        );
    }

    #[tokio::test]
    async fn two_sessions_run_tools_in_their_own_working_directory() {
        use serde_json::json;
        // Working-directory isolation: two sessions from one Runtime read a
        // relative path and each gets its own dir's file — no cross-talk.
        let base = std::env::temp_dir()
            .join(format!("tapir-cwdiso-{}", std::process::id()));
        let dir_a = base.join("a");
        let dir_b = base.join("b");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        std::fs::write(dir_a.join("f.txt"), "alpha").unwrap();
        std::fs::write(dir_b.join("f.txt"), "beta").unwrap();

        let rt = crate::runtime::Runtime::builder().build();
        let read_f = || {
            ScriptedRunner::new(vec![
                Scripted::Ok {
                    deltas: vec![],
                    outcome: tool_round("read", json!({ "path": "f.txt" })),
                },
                Scripted::Ok {
                    deltas: vec!["ok".into()],
                    outcome: text_round("ok"),
                },
            ])
        };
        async fn run_and_read(
            agent: &mut Agent,
            runner: &ScriptedRunner,
        ) -> String {
            let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
            agent.run_turn(runner, &tx).await;
            drop(tx);
            drain(&mut rx)
                .await
                .iter()
                .find_map(|e| match e {
                    TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                    _ => None,
                })
                .expect("the read ran")
        }

        let mut a = rt.session(dir_a.clone());
        let mut b = rt.session(dir_b.clone());
        let out_a = run_and_read(&mut a, &read_f()).await;
        let out_b = run_and_read(&mut b, &read_f()).await;

        assert_eq!(out_a, "alpha", "session A read its own working directory");
        assert_eq!(
            out_b, "beta",
            "session B read its own working directory — no cross-talk"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn a_session_built_with_custom_exec_ops_runs_bash_through_them() {
        use serde_json::json;
        // The injection plumbing every sandbox backend reuses: exec operations
        // configured on the builder reach a session's bash calls — no process
        // is spawned, the fake's outcome is the tool's result.
        let fake =
            std::sync::Arc::new(crate::tools::exec::testing::FakeExecOps::ok(
                "from the sandbox\n",
                0,
            ));
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("bash", json!({ "command": "echo hi" })),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let mut agent = crate::runtime::Runtime::builder()
            .exec_ops(fake.clone())
            .build()
            .session(".".into());
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let output = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the bash call ran");
        assert!(
            output.contains("from the sandbox"),
            "the ops' output came back: {output}"
        );
        let calls = fake.calls();
        assert_eq!(
            calls.len(),
            1,
            "the session routed exec through the injected ops"
        );
        assert_eq!(calls[0].command, "echo hi");
    }

    #[tokio::test]
    async fn a_session_built_with_custom_fs_ops_routes_file_tools_through_them()
    {
        use serde_json::json;
        // The injection plumbing every sandbox backend reuses: filesystem
        // operations configured on the builder reach a session's file tools —
        // the read serves the fake's tree, not the host's.
        let fake = std::sync::Arc::new(
            crate::tools::fs::testing::FakeFsOps::new()
                .file("/nowhere/real/f.txt", "from the sandbox tree"),
        );
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("read", json!({ "path": "f.txt" })),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let mut agent = crate::runtime::Runtime::builder()
            .fs_ops(fake.clone())
            .build()
            .session("/nowhere/real".into());
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        let output = events
            .iter()
            .find_map(|e| match e {
                TurnEvent::ToolEnd { output, .. } => Some(output.clone()),
                _ => None,
            })
            .expect("the read call ran");
        assert!(
            output.contains("from the sandbox tree"),
            "the ops' tree came back: {output}"
        );
        assert_eq!(
            fake.reads(),
            [std::path::PathBuf::from("/nowhere/real/f.txt")],
            "the session routed the read through the injected ops"
        );
    }

    #[tokio::test]
    async fn dispatch_refuses_to_run_a_tool_on_an_aborted_turn() {
        use serde_json::json;
        // The execution context carries the abort signal; dispatch honours it and
        // does not start the tool (here read would otherwise just fail on a file).
        let registry: crate::tools::Tools =
            std::sync::Arc::from(crate::tools::builtin_tools());
        let cancel = CancelToken::default();
        cancel.abort();
        let ctx = crate::tools::tool::ToolCtx::new(".".into(), cancel, None);
        let err = crate::tools::dispatch(
            &registry,
            "read",
            &json!({ "path": "x" }),
            &ctx,
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("aborted"),
            "an aborted turn refuses the tool, got: {err}",
        );
    }

    #[tokio::test]
    async fn empty_round_with_nothing_queued_is_an_error() {
        // A round that produces no text, no thinking, and no tool calls — and an
        // empty queue — is a dead end, surfaced as an error rather than a blank
        // reply.
        let runner = ScriptedRunner::new(vec![Scripted::Ok {
            deltas: vec![],
            outcome: text_round(""),
        }]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(
            vec![Step::User { text: "hi".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            matches!(events.last(), Some(TurnEvent::Error { message }) if message.contains("empty")),
            "an empty response should end the turn with an error",
        );
        assert!(
            !events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "an empty response is not a successful Done",
        );
    }

    fn queued(kind: crate::queue::Kind, text: &str) -> crate::queue::Queued {
        crate::queue::Queued {
            display: text.into(),
            model_text: text.into(),
            images: Vec::new(),
            kind,
        }
    }

    #[tokio::test]
    async fn steering_message_is_delivered_after_a_tool_round() {
        use serde_json::json;
        let dir = std::env::temp_dir()
            .join(format!("tapir-steer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // The frontend queues a steering message while the agent works.
        let queue = crate::queue::new();
        queue
            .lock()
            .unwrap()
            .push(queued(crate::queue::Kind::Steer, "keep going"));

        // Round 1 calls a tool; the steering message should be delivered after
        // it; round 2 replies and ends the turn.
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec![],
                outcome: tool_round("ls", json!({})),
            },
            Scripted::Ok {
                deltas: vec!["ok".into()],
                outcome: text_round("ok"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = Agent::new(
            vec![Step::User { text: "go".into(), images: Vec::new() }],
            dir.clone(),
            queue.clone(),
            crate::queue::Mode::OneAtATime,
            crate::queue::Mode::OneAtATime,
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(
                |e| matches!(e, TurnEvent::QueuedDelivered { display, .. } if display == "keep going")
            ),
            "the queued steering message should be delivered into the running turn",
        );
        assert!(
            agent.history().iter().any(
                |s| matches!(s, Step::User { text, .. } if text == "keep going")
            ),
            "the steering message becomes a user step in the conversation",
        );
        assert!(crate::queue::is_empty(&queue), "the queue is drained");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn follow_up_is_delivered_when_idle_with_no_steering() {
        let queue = crate::queue::new();
        queue
            .lock()
            .unwrap()
            .push(queued(crate::queue::Kind::FollowUp, "and then this"));

        // Round 1 replies (idle, no tools); with a follow-up queued the turn
        // continues; round 2 replies and, the queue now empty, the turn ends.
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec!["first".into()],
                outcome: text_round("first"),
            },
            Scripted::Ok {
                deltas: vec!["second".into()],
                outcome: text_round("second"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = Agent::new(
            vec![Step::User { text: "go".into(), images: Vec::new() }],
            ".".into(),
            queue.clone(),
            crate::queue::Mode::OneAtATime,
            crate::queue::Mode::OneAtATime,
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(
                |e| matches!(e, TurnEvent::QueuedDelivered { display, .. } if display == "and then this")
            ),
            "the queued follow-up should be delivered once the agent is idle",
        );
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "the turn ends after the follow-up is answered",
        );
        assert!(crate::queue::is_empty(&queue), "the queue is drained");
    }

    #[tokio::test]
    async fn turn_stops_after_the_round_cap_even_if_the_model_never_idles() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A runner that never goes idle — it requests a tool every round.
        struct Looping {
            calls: AtomicUsize,
        }
        impl RoundRunner for Looping {
            async fn run_round(
                &self,
                _history: &[Step],
                _tx: &mpsc::Sender<TurnEvent>,
            ) -> Result<RoundOutcome, RoundError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(tool_round("ls", serde_json::json!({})))
            }
        }

        let runner = Looping { calls: AtomicUsize::new(0) };
        // Big buffer: the loop runs unattended, so the channel must hold a whole
        // capped turn's worth of events without a concurrent receiver.
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(4096);
        let mut agent = plain_agent(
            vec![Step::User { text: "go".into(), images: Vec::new() }],
            ".".into(),
        );

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert_eq!(
            runner.calls.load(Ordering::SeqCst),
            MAX_ROUNDS,
            "the turn is bounded to the round cap",
        );
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "hitting the cap ends the turn",
        );
    }

    // --- The Agent as a held conversation ----------------------------------

    /// The text a step carries, for asserting what the provider was handed.
    fn step_text(step: &Step) -> String {
        match step {
            Step::User { text, .. } => text.clone(),
            Step::Assistant { text, .. } => text.clone(),
            Step::ToolResult { output, .. } => output.clone(),
        }
    }

    /// A [`RoundRunner`] that records the history it is handed on each round, so
    /// a test can assert what the provider actually sees — including turns from
    /// earlier in the same Agent's life.
    struct RecordingRunner {
        rounds: std::sync::Mutex<std::collections::VecDeque<RoundOutcome>>,
        seen: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl RecordingRunner {
        fn new(rounds: Vec<RoundOutcome>) -> Self {
            Self {
                rounds: std::sync::Mutex::new(rounds.into()),
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl RoundRunner for RecordingRunner {
        async fn run_round(
            &self,
            history: &[Step],
            _tx: &mpsc::Sender<TurnEvent>,
        ) -> Result<RoundOutcome, RoundError> {
            self.seen
                .lock()
                .unwrap()
                .push(history.iter().map(step_text).collect());
            Ok(self
                .rounds
                .lock()
                .unwrap()
                .pop_front()
                .expect("ran out of rounds"))
        }
    }

    #[tokio::test]
    async fn history_accumulates_across_sequential_turns() {
        // The Agent is held across turns: a second turn must be sent with the
        // first turn's user message and the model's reply, not a blank history.
        let runner =
            RecordingRunner::new(vec![text_round("one"), text_round("two")]);
        let (tx, _rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(Vec::new(), ".".into());

        agent.submit(Input::text("first"));
        agent.run_turn(&runner, &tx).await;

        agent.submit(Input::text("second"));
        agent.run_turn(&runner, &tx).await;

        let seen = runner.seen.lock().unwrap();
        assert_eq!(
            seen.last().expect("the second turn ran a round"),
            &["first", "one", "second"],
            "the second turn should send the whole accumulated conversation",
        );
    }

    #[tokio::test]
    async fn steer_enqueues_a_message_delivered_into_the_running_turn() {
        // Steering goes through the engine method, not the raw shared queue.
        // Round 1 is idle, so the steering message is delivered before round 2.
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok { deltas: vec![], outcome: text_round("working") },
            Scripted::Ok {
                deltas: vec!["done".into()],
                outcome: text_round("done"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("go"));
        agent.steer(Input::text("keep going"));

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(|e| matches!(
                e, TurnEvent::QueuedDelivered { display, .. } if display == "keep going"
            )),
            "steer() should enqueue a steering message delivered into the turn",
        );
    }

    #[tokio::test]
    async fn follow_up_enqueues_a_message_delivered_when_idle() {
        // A follow-up waits until the agent is idle with no steering pending,
        // then runs as its own round.
        let runner = ScriptedRunner::new(vec![
            Scripted::Ok {
                deltas: vec!["first".into()],
                outcome: text_round("first"),
            },
            Scripted::Ok {
                deltas: vec!["second".into()],
                outcome: text_round("second"),
            },
        ]);
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("go"));
        agent.follow_up(Input::text("and then this"));

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(|e| matches!(
                e, TurnEvent::QueuedDelivered { display, .. } if display == "and then this"
            )),
            "follow_up() should enqueue a follow-up delivered once the agent is idle",
        );
    }

    #[tokio::test]
    async fn abort_stops_the_turn_between_rounds() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // A real directory so the `ls` tool runs; the runner asks for a tool
        // every round (the turn would never idle on its own) and aborts after
        // the first round via the agent's cancel handle.
        let dir = std::env::temp_dir()
            .join(format!("tapir-abort-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        struct AbortAfterFirst {
            cancel: CancelToken,
            calls: AtomicUsize,
        }
        impl RoundRunner for AbortAfterFirst {
            async fn run_round(
                &self,
                _history: &[Step],
                _tx: &mpsc::Sender<TurnEvent>,
            ) -> Result<RoundOutcome, RoundError> {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.cancel.abort();
                }
                Ok(tool_round("ls", serde_json::json!({})))
            }
        }

        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(Vec::new(), dir.clone());
        agent.submit(Input::text("go"));
        let runner = AbortAfterFirst {
            cancel: agent.cancel_token(),
            calls: AtomicUsize::new(0),
        };

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        assert_eq!(
            runner.calls.load(Ordering::SeqCst),
            1,
            "an abort should stop the loop after the in-flight round",
        );
        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "an aborted turn still ends cleanly with Done",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn abort_method_short_circuits_the_next_run() {
        // `abort()` is exposed directly on the engine; once set, the next run
        // does no rounds at all (the runner — scripted with none — would panic
        // if a round were attempted).
        let runner = ScriptedRunner::new(Vec::new());
        let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("go"));
        agent.abort();

        agent.run_turn(&runner, &tx).await;
        drop(tx);

        let events = drain(&mut rx).await;
        assert!(
            events.iter().any(|e| matches!(e, TurnEvent::Done)),
            "an agent aborted before running ends immediately with Done",
        );
    }

    #[test]
    fn agent_is_send_and_sync() {
        // Slice #01: the engine must be ownable inside an async task and
        // shareable across threads. A non-`Send`/`Sync` field added later fails
        // to compile here, at the contract, rather than at a distant spawn site.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Agent>();
    }

    // --- Durable history maintenance (replacing the transcript-derived one) ---

    #[test]
    fn the_conversation_is_shared_between_session_handles() {
        // The same move the queue and cancel token already make: the durable
        // history is a shared handle, so a turn task holding it and the held
        // Session observe each other's writes — the seam that lets run(input)
        // retire the ephemeral-agent + event-mirroring pattern (slice #02).
        let mut agent = plain_agent(Vec::new(), ".".into());
        let history = agent.history_handle();

        // A write through the Session is visible through the handle…
        agent.submit(Input::text("from the session"));
        assert_eq!(
            history.lock().unwrap().len(),
            1,
            "the handle sees the session's write"
        );

        // …and a write through the handle is visible in the Session's snapshot.
        history.lock().unwrap().push(Step::Assistant {
            text: "from the handle".into(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            raw: None,
        });
        assert!(
            matches!(
                agent.history().last(),
                Some(Step::Assistant { text, .. }) if text == "from the handle"
            ),
            "the session's snapshot sees the handle's write",
        );
    }

    #[test]
    fn submit_coalesces_consecutive_user_messages() {
        // Two user inputs with no assistant reply between them (a `!` output, a
        // skill block, a follow-up) merge into one message — the wire format
        // must alternate roles.
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("first"));
        agent.submit(Input::text("second"));
        let h = agent.history();
        assert_eq!(
            h.len(),
            1,
            "consecutive user inputs merge into one message"
        );
        assert!(
            matches!(&h[0], Step::User { text, .. } if text == "first\n\nsecond")
        );
    }

    #[test]
    fn submit_keeps_an_image_message_separate() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("look"));
        agent.submit(Input {
            display: "img".into(),
            model_text: "img".into(),
            images: vec![Image { mime: "image/png".into(), data: "x".into() }],
        });
        assert_eq!(
            agent.history().len(),
            2,
            "an image message stays its own block"
        );
    }

    #[test]
    fn push_assistant_coalesces_segments_split_by_tools() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("q"));
        agent.push_assistant("part one");
        agent.push_assistant("part two");
        let h = agent.history();
        assert_eq!(
            h.len(),
            2,
            "assistant segments coalesce after the user message"
        );
        assert!(
            matches!(&h[1], Step::Assistant { text, .. } if text == "part one\n\npart two")
        );
    }

    #[test]
    fn compact_replaces_history_with_a_summary_message() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("old question"));
        agent.push_assistant("old answer");
        agent.compact("S");
        let h = agent.history();
        assert_eq!(h.len(), 1);
        assert!(matches!(&h[0], Step::User { text, .. }
            if text == "Summary of the conversation so far:\n\nS"));
    }

    #[test]
    fn interrupted_marks_the_turn_in_the_durable_history() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        agent.submit(Input::text("do it"));
        agent.push_assistant("working");
        agent.interrupted();
        // The marker coalesces into the assistant segment (like the old path).
        assert!(matches!(h_last(&agent), Step::Assistant { text, .. }
            if text == format!("working\n\n{INTERRUPTED}")));
    }

    fn h_last(agent: &Agent) -> Step {
        agent.history().last().cloned().expect("history is non-empty")
    }

    #[tokio::test]
    async fn one_runtime_drives_two_independent_sessions() {
        // The SDK front door: one Runtime spawns many sessions, each its own
        // conversation, with no cross-talk between them.
        let rt = crate::runtime::Runtime::builder().build();
        let mut a = rt.session(".".into());
        let mut b = rt.session(".".into());

        a.submit(Input::text("ask a"));
        b.submit(Input::text("ask b"));

        let runner_a = RecordingRunner::new(vec![text_round("reply a")]);
        let runner_b = RecordingRunner::new(vec![text_round("reply b")]);
        let (tx, _rx) = mpsc::channel::<TurnEvent>(64);
        a.run_turn(&runner_a, &tx).await;
        b.run_turn(&runner_b, &tx).await;

        // Each session's provider saw only its own conversation…
        assert_eq!(runner_a.seen.lock().unwrap().last().unwrap(), &["ask a"]);
        assert_eq!(runner_b.seen.lock().unwrap().last().unwrap(), &["ask b"]);
        // …each accumulated only its own reply…
        assert!(
            matches!(h_last(&a), Step::Assistant { text, .. } if text == "reply a")
        );
        assert!(
            matches!(h_last(&b), Step::Assistant { text, .. } if text == "reply b")
        );
        // …and their steering queues don't bleed into each other.
        a.steer(Input::text("only a"));
        assert!(!crate::queue::is_empty(&a.queue_handle()));
        assert!(crate::queue::is_empty(&b.queue_handle()));
    }

    // --- The engine owns the turn lifecycle (run / is_running / abort) --------

    #[tokio::test]
    async fn run_spawns_the_turn_reports_running_and_streams_events() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        assert!(!agent.is_running(), "idle before run()");

        // The engine spawns the task and hands back the event receiver.
        let mut rx = agent.run_with(|tx| async move {
            let _ = tx.send(TurnEvent::Text { delta: "hello".into() }).await;
            let _ = tx.send(TurnEvent::Done).await;
        });
        assert!(agent.is_running(), "running once the turn is spawned");

        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        assert!(
            matches!(events.first(), Some(TurnEvent::Text { delta }) if delta == "hello")
        );
        assert!(events.iter().any(|e| matches!(e, TurnEvent::Done)));

        // It stays flagged running until the frontend marks idle on the terminal
        // event — so "is a turn running?" is the engine's to answer.
        assert!(agent.is_running(), "still running until set_idle");
        agent.set_idle();
        assert!(!agent.is_running(), "idle after set_idle");
    }

    #[tokio::test]
    async fn abort_hard_kills_the_running_turn() {
        let mut agent = plain_agent(Vec::new(), ".".into());
        // A turn that emits one event then hangs forever — only a hard abort
        // (cancelling the task) can end it, since it never checks the token.
        let mut rx = agent.run_with(|tx| async move {
            let _ = tx.send(TurnEvent::Text { delta: "working".into() }).await;
            std::future::pending::<()>().await;
        });
        assert!(matches!(rx.recv().await, Some(TurnEvent::Text { .. })));

        // The engine owns the task handle, so abort() cancels it outright; the
        // task is dropped, its sender with it, and the channel closes.
        agent.abort();
        assert!(rx.recv().await.is_none(), "abort closes the turn's channel");
    }
}
