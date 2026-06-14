//! The SDK front door: a [`Runtime`] built by a [`RuntimeBuilder`], from which
//! conversation sessions (the engine's [`Agent`]) are spawned.
//!
//! The Runtime holds process-wide wiring — today just the [`Config`]; pluggable
//! pieces (credentials, storage, tools, providers, hooks, observers, commands)
//! slot onto the same builder in their own slices. It is cheaply cloneable
//! (`Arc`-backed) and `Send + Sync`, so one process can hold many sessions: the
//! TUI holds one, a bot would hold many, all driving the same wiring.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::Agent;
use crate::config::Config;
use crate::observer::Observer;

/// Builds a [`Runtime`]. Today it only carries the [`Config`]; the pluggable
/// registries (credentials, storage, tools, providers, hooks, observers,
/// commands) attach here as their slices land.
pub struct RuntimeBuilder {
    config: Config,
    observers: Vec<Arc<dyn Observer>>,
    tools: Vec<Arc<dyn crate::tools::tool::Tool>>,
    hooks: Vec<Arc<dyn crate::hook::Hook>>,
    providers: Vec<Arc<dyn crate::providers::Provider>>,
    credentials: Arc<dyn crate::credentials::CredentialProvider>,
    store: Arc<dyn crate::store::SessionStore>,
    commands: Vec<Arc<dyn crate::command::Command>>,
    exec_ops: Arc<dyn crate::tools::exec::ExecOps>,
    fs_ops: Arc<dyn crate::tools::fs::FsOps>,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        // Seed the registry with the built-in tools already on the trait; the
        // rest still run through the name dispatch until they're converted.
        Self {
            config: Config::default(),
            observers: Vec::new(),
            tools: crate::tools::builtin_tools(),
            hooks: Vec::new(),
            providers: crate::providers::builtin_providers(),
            credentials: Arc::new(crate::credentials::FileCreds::new()),
            store: Arc::new(crate::store::InMemoryStore::default()),
            commands: crate::command::builtin_commands(),
            exec_ops: Arc::new(crate::tools::exec::LocalExecOps),
            fs_ops: Arc::new(crate::tools::fs::LocalFsOps),
        }
    }

    /// Set the process-wide configuration.
    // Part of the builder surface; the TUI wires real config through here in the
    // config-injection slice (it builds with defaults for now).
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Register a read-only [`Observer`], notified of every event of every
    /// session this Runtime spawns. Call repeatedly to add several.
    // Part of the builder surface; adapters (a bot, a logger) register here. The
    // TUI registers none yet; a logging/auditing adapter hangs off it.
    pub fn observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Register a custom [`Tool`](crate::tools::tool::Tool); the turn loop
    /// dispatches matching calls through it. Built-in tools are already present.
    // Part of the builder surface — adapters / embedders add tools here; the TUI
    // adds none beyond the built-ins yet.
    pub fn tool(mut self, tool: Arc<dyn crate::tools::tool::Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Register an interception [`Hook`](crate::hook::Hook); the turn loop
    /// consults it around each tool call (deny / modify before, override after).
    /// Call repeatedly to add several — they run in registration order.
    // Part of the builder surface — adapters register a policy gate here; the TUI
    // registers none, so it's exercised only by the seam tests in this binary.
    pub fn hook(mut self, hook: Arc<dyn crate::hook::Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Register a [`Provider`](crate::providers::Provider) — a model wire
    /// protocol dispatched by id. Call repeatedly to add several.
    // Part of the builder surface — adapters / embedders add providers here; the
    // TUI uses the built-ins.
    pub fn provider(
        mut self,
        provider: Arc<dyn crate::providers::Provider>,
    ) -> Self {
        self.providers.push(provider);
        self
    }

    /// Inject the [`CredentialProvider`](crate::credentials::CredentialProvider)
    /// every Provider's default credential path consults. Defaults to
    /// [`FileCreds`](crate::credentials::FileCreds) — the existing `auth.toml`
    /// chain — so the TUI behaves as before; a headless adapter injects e.g.
    /// [`EnvCreds`](crate::credentials::EnvCreds) or its own secret store.
    // Part of the builder surface — adapters inject here; the TUI keeps the
    // default.
    pub fn credentials(
        mut self,
        resolver: Arc<dyn crate::credentials::CredentialProvider>,
    ) -> Self {
        self.credentials = resolver;
        self
    }

    /// Inject the [`SessionStore`](crate::store::SessionStore) conversations
    /// persist to and resume from. Defaults to the in-memory store; a frontend
    /// that wants durable conversations registers the file store (or a bot its
    /// database) here — the TUI and headless print both inject
    /// [`FileStore::in_layout`](crate::store::FileStore::in_layout).
    pub fn store(mut self, store: Arc<dyn crate::store::SessionStore>) -> Self {
        self.store = store;
        self
    }

    /// Register a [`Command`](crate::command::Command) — a named conversation
    /// operation dispatched through [`Runtime::run_command`]. Call repeatedly
    /// to add several; plugins and adapters extend the set the same way.
    // Part of the builder surface — see the note on the trait; exercised by the
    // seam tests.
    pub fn command(
        mut self,
        command: Arc<dyn crate::command::Command>,
    ) -> Self {
        self.commands.push(command);
        self
    }

    /// Inject the [`ExecOps`](crate::tools::exec::ExecOps) every session's
    /// `bash` calls execute through. Defaults to
    /// [`LocalExecOps`](crate::tools::exec::LocalExecOps) — the host spawn
    /// path — so the TUI and headless modes behave as before; a sandbox
    /// backend injects its own.
    pub fn exec_ops(
        mut self,
        ops: Arc<dyn crate::tools::exec::ExecOps>,
    ) -> Self {
        self.exec_ops = ops;
        self
    }

    /// Inject the [`FsOps`](crate::tools::fs::FsOps) every session's file
    /// tools (read, write, edit, ls, grep, find) perform their raw I/O
    /// through. Defaults to [`LocalFsOps`](crate::tools::fs::LocalFsOps) —
    /// the host `std::fs` path — so the TUI and headless modes behave as
    /// before; a sandbox backend injects its own.
    pub fn fs_ops(mut self, ops: Arc<dyn crate::tools::fs::FsOps>) -> Self {
        self.fs_ops = ops;
        self
    }

    /// Finish building. The result is cheaply cloneable and shareable across
    /// tasks (clones share the same wiring).
    pub fn build(self) -> Runtime {
        Runtime(Arc::new(Inner {
            config: self.config,
            observers: Arc::from(self.observers),
            tools: Arc::from(self.tools),
            hooks: Arc::from(self.hooks),
            providers: Arc::from(self.providers),
            credentials: self.credentials,
            store: self.store,
            commands: Arc::from(self.commands),
            exec_ops: self.exec_ops,
            fs_ops: self.fs_ops,
        }))
    }
}

/// The SDK runtime: shared, process-wide wiring from which sessions are spawned.
/// Cloning is an `Arc` bump, and clones share the same wiring.
#[derive(Clone)]
pub struct Runtime(Arc<Inner>);

struct Inner {
    config: Config,
    observers: crate::agent::Observers,
    tools: crate::tools::Tools,
    hooks: crate::agent::Hooks,
    providers: crate::providers::Providers,
    credentials: Arc<dyn crate::credentials::CredentialProvider>,
    store: Arc<dyn crate::store::SessionStore>,
    commands: Arc<[Arc<dyn crate::command::Command>]>,
    exec_ops: Arc<dyn crate::tools::exec::ExecOps>,
    fs_ops: Arc<dyn crate::tools::fs::FsOps>,
}

/// How a session selects from the Runtime's registered tools.
//
// `Subset` (and the `only`/`metadata` builders, and `Agent::tool_definitions`)
// are the per-session restriction API for adapters (a bot scopes a session by
// construction). The TUI spawns a full-tool session, so they're exercised only
// by the RPC adapter (per-session tool subsets) and the seam tests.
#[derive(Clone, Default)]
pub enum ToolSelection {
    /// Every tool registered on the Runtime.
    #[default]
    All,
    /// Only the named tools; the rest are excluded by construction.
    Subset(Vec<String>),
}

/// How a Session's system prompt is assembled, per turn (context files are
/// re-read each turn, so edits land without restarting). The default is the
/// minimal prompt — no context files, no skills, no append — which existing
/// callers get when they set nothing; a frontend opts into the full assembly.
#[derive(Clone, Default)]
pub struct PromptSpec {
    /// Load the project/global context files (AGENTS.md / CLAUDE.md / …).
    pub context: bool,
    /// List the available skills (requires the `read` tool to be active).
    pub skills: bool,
    /// Extra skill paths to scan (the `--skill` flags), additive.
    pub skill_paths: Vec<std::path::PathBuf>,
    /// Trust the project's own context files and skills (the trust gate).
    pub trust_project: bool,
    /// When set, merge the config dir's `system.md` plus these extra lines
    /// into the prompt (the `--append-system-prompt` flags). `None` appends
    /// nothing at all.
    pub append: Option<Vec<String>>,
    /// Replace the default prompt body entirely (`--system-prompt`).
    pub custom: Option<String>,
}

/// Per-session options: where the session runs, which of the Runtime's tools it
/// may use, an opaque metadata map carried to each tool's execution context,
/// and how its system prompt is assembled.
#[derive(Clone)]
pub struct SessionOptions {
    pub cwd: PathBuf,
    pub tools: ToolSelection,
    pub metadata: crate::tools::tool::Metadata,
    pub prompt: PromptSpec,
}

impl SessionOptions {
    /// Options for `cwd` with the full registered tool set, no metadata, and
    /// the minimal prompt.
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            cwd,
            tools: ToolSelection::All,
            metadata: Default::default(),
            prompt: PromptSpec::default(),
        }
    }

    /// Assemble the session's system prompt from `spec` each turn.
    pub fn prompt(mut self, spec: PromptSpec) -> Self {
        self.prompt = spec;
        self
    }

    /// Restrict the session to only the named tools. (Adapter API — see
    /// [`ToolSelection`].)
    pub fn only<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tools =
            ToolSelection::Subset(names.into_iter().map(Into::into).collect());
        self
    }

    /// Set the session's opaque metadata from key/value pairs. (Adapter API.)
    pub fn metadata<I, K, V>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata =
            entries.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }
}

/// A fluent builder for a session ([`Agent`]), started by [`Runtime::agent`].
///
/// Collects per-session configuration and produces a ready-to-drive [`Agent`]
/// on [`build`](AgentBuilder::build) — sugar over [`Runtime::session_with`] plus
/// the `Agent` setters. Hooks and observers are *additive* over the Runtime's
/// defaults; the ops and boundary *replace* them.
///
/// ```no_run
/// # use tapir_core::runtime::Runtime;
/// # fn demo(rt: Runtime) {
/// let agent = rt
///     .agent()
///     .model("anthropic", "claude-opus-4-8")
///     .thinking("high")
///     .only_tools(["read", "bash"])
///     .build();
/// # let _ = agent;
/// # }
/// ```
pub struct AgentBuilder {
    runtime: Runtime,
    opts: SessionOptions,
    model: Option<crate::agent::ModelRef>,
    thinking: Option<String>,
    boundary: Option<crate::tools::jail::PathBoundary>,
    exec_ops: Option<Arc<dyn crate::tools::exec::ExecOps>>,
    fs_ops: Option<Arc<dyn crate::tools::fs::FsOps>>,
    hooks: Vec<Arc<dyn crate::hook::Hook>>,
    observers: Vec<Arc<dyn Observer>>,
}

impl AgentBuilder {
    /// Defaults mirror [`SessionOptions::new`] (cwd `.`, all tools, no metadata)
    /// with no model, thinking, boundary, op overrides, or extra hooks/observers.
    fn new(runtime: Runtime) -> Self {
        Self {
            runtime,
            opts: SessionOptions::new(PathBuf::from(".")),
            model: None,
            thinking: None,
            boundary: None,
            exec_ops: None,
            fs_ops: None,
            hooks: Vec::new(),
            observers: Vec::new(),
        }
    }

    /// The working directory the session runs in (default `.`).
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.opts.cwd = cwd.into();
        self
    }

    /// Point the session at a model: a provider id and one of its model ids.
    pub fn model(
        mut self,
        provider: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        self.model = Some(crate::agent::ModelRef {
            provider: provider.into(),
            id: id.into(),
        });
        self
    }

    /// Set the reasoning/thinking level (e.g. `"off"`, `"low"`, `"high"`).
    pub fn thinking(mut self, level: impl Into<String>) -> Self {
        self.thinking = Some(level.into());
        self
    }

    /// Replace the prompt-assembly knobs (context/skills/append/custom).
    pub fn prompt(mut self, spec: PromptSpec) -> Self {
        self.opts.prompt = spec;
        self
    }

    /// Restrict the session to only the named tools (default: all registered).
    pub fn only_tools<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.opts.tools =
            ToolSelection::Subset(names.into_iter().map(Into::into).collect());
        self
    }

    /// Opaque per-session metadata, carried to each tool's execution context.
    pub fn metadata<I, K, V>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.opts.metadata =
            entries.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }

    /// Confine the session's tool file access to a workspace boundary.
    pub fn boundary(
        mut self,
        boundary: crate::tools::jail::PathBoundary,
    ) -> Self {
        self.boundary = Some(boundary);
        self
    }

    /// Override how shell commands run (default: the Runtime's exec ops).
    pub fn exec_ops(
        mut self,
        ops: Arc<dyn crate::tools::exec::ExecOps>,
    ) -> Self {
        self.exec_ops = Some(ops);
        self
    }

    /// Override how the file tools touch disk (default: the Runtime's fs ops).
    pub fn fs_ops(mut self, ops: Arc<dyn crate::tools::fs::FsOps>) -> Self {
        self.fs_ops = Some(ops);
        self
    }

    /// Add a hook for this session, on top of the Runtime's hooks.
    pub fn hook(mut self, hook: Arc<dyn crate::hook::Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Add an observer for this session, on top of the Runtime's observers.
    pub fn observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Build the configured [`Agent`].
    pub fn build(self) -> Agent {
        let mut agent = self.runtime.session_with(self.opts);
        if let Some(model) = self.model {
            agent.set_model(Some(model));
        }
        if let Some(level) = self.thinking {
            agent.set_thinking(level);
        }
        if self.boundary.is_some() {
            agent.set_boundary(self.boundary);
        }
        if let Some(ops) = self.exec_ops {
            agent.set_exec_ops(ops);
        }
        if let Some(ops) = self.fs_ops {
            agent.set_fs_ops(ops);
        }
        // Hooks/observers extend the Runtime's defaults (already on the agent).
        if !self.hooks.is_empty() {
            let mut hooks: Vec<_> =
                agent.hooks_handle().iter().cloned().collect();
            hooks.extend(self.hooks);
            agent.set_hooks(Arc::from(hooks));
        }
        if !self.observers.is_empty() {
            let mut observers: Vec<_> =
                agent.observers_handle().iter().cloned().collect();
            observers.extend(self.observers);
            agent.set_observers(Arc::from(observers));
        }
        agent
    }
}

impl Runtime {
    /// Start building a Runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    /// The process-wide configuration.
    pub fn config(&self) -> &Config {
        &self.0.config
    }

    /// Spawn a fresh conversation for `cwd`, realized as the engine's [`Agent`].
    /// Each session is independent — its own history, steering/follow-up queue,
    /// and cancel handle — so one Runtime can drive many at once. Delivery modes
    /// come from the Runtime's config. (Richer per-conversation options — model,
    /// system prompt, thinking, tool selection — arrive with the
    /// per-session-context slice.)
    pub fn session(&self, cwd: PathBuf) -> Agent {
        self.session_with(SessionOptions::new(cwd))
    }

    /// Start an [`AgentBuilder`] for a fluent, one-shot session setup — the
    /// working directory, model, thinking level, tool selection, metadata, and
    /// per-session hooks/observers in one chain, ending in
    /// [`build`](AgentBuilder::build). Sugar over [`session_with`] plus the
    /// `Agent` setters; `session`/`session_with` remain for adapters that hold a
    /// [`SessionOptions`] directly.
    ///
    /// [`session_with`]: Runtime::session_with
    pub fn agent(&self) -> AgentBuilder {
        AgentBuilder::new(self.clone())
    }

    /// Spawn a session from full [`SessionOptions`] — the working directory and a
    /// tool selection (a subset of the Runtime's registered tools).
    pub fn session_with(&self, opts: SessionOptions) -> Agent {
        let mode = |m: &Option<String>| {
            m.as_deref()
                .map(crate::queue::Mode::from_setting)
                .unwrap_or(crate::queue::Mode::OneAtATime)
        };
        let mut agent = Agent::new(
            Vec::new(),
            opts.cwd,
            crate::queue::new(),
            mode(&self.0.config.steering_mode),
            mode(&self.0.config.follow_up_mode),
        );
        agent.set_observers(self.0.observers.clone());
        agent.set_tools(self.select_tools(&opts.tools));
        agent.set_metadata(Arc::new(opts.metadata));
        agent.set_hooks(self.0.hooks.clone());
        agent.set_exec_ops(self.0.exec_ops.clone());
        agent.set_fs_ops(self.0.fs_ops.clone());
        agent.set_runtime(self.clone());
        agent.set_prompt(opts.prompt);
        agent
    }

    /// The registered [`Provider`](crate::providers::Provider) with this id, if
    /// any — how a turn resolves the wire protocol for the active provider.
    pub fn find_provider(
        &self,
        id: &str,
    ) -> Option<Arc<dyn crate::providers::Provider>> {
        self.0.providers.iter().find(|p| p.id() == id).cloned()
    }

    /// Every registered Provider, in registration order — how a frontend
    /// enumerates the model catalog (validation and help for a runtime model
    /// switch).
    pub fn providers(&self) -> &[Arc<dyn crate::providers::Provider>] {
        &self.0.providers
    }

    /// The injected [`CredentialProvider`](crate::credentials::CredentialProvider)
    /// — handed to [`Provider::creds`](crate::providers::Provider::creds) by
    /// whoever spawns a turn.
    pub fn credentials(
        &self,
    ) -> Arc<dyn crate::credentials::CredentialProvider> {
        self.0.credentials.clone()
    }

    /// The injected [`SessionStore`](crate::store::SessionStore) — where a
    /// frontend persists conversation entries as a turn progresses (the TUI
    /// drains its write queue here; headless print appends inline).
    pub fn store(&self) -> Arc<dyn crate::store::SessionStore> {
        self.0.store.clone()
    }

    /// Resume a stored conversation: load its entries from the injected store
    /// and rebuild a session whose durable history replays them — messages,
    /// compaction checkpoint, shell context, and interruption markers — so any
    /// frontend inherits context management instead of reimplementing it.
    /// (The TUI resumes onto its existing session — same `store::replay`,
    /// preserving the model/thinking slots — rather than minting a new one.)
    pub async fn resume(
        &self,
        opts: SessionOptions,
        id: &str,
    ) -> anyhow::Result<Agent> {
        let entries = self.0.store.load(id).await?;
        let mut agent = self.session_with(opts);
        crate::store::replay(&mut agent, &entries);
        Ok(agent)
    }

    /// Dispatch a registered [`Command`](crate::command::Command) by name
    /// against `session`, lending it `ui` for whatever this adapter mode can
    /// show. The same command works across adapters — rich UI or none.
    // Adapter-facing, like the registry itself.
    pub async fn run_command(
        &self,
        name: &str,
        args: &str,
        session: &mut Agent,
        ui: &mut dyn crate::command::CommandUi,
    ) -> anyhow::Result<()> {
        let command = self
            .0
            .commands
            .iter()
            .find(|c| c.name() == name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown command: {name}"))?;
        let mut ctx = crate::command::CommandCtx { session, runtime: self, ui };
        command.run(args.trim(), &mut ctx).await
    }

    /// The registered tools a selection enables.
    fn select_tools(&self, sel: &ToolSelection) -> crate::tools::Tools {
        match sel {
            ToolSelection::All => self.0.tools.clone(),
            ToolSelection::Subset(names) => self
                .0
                .tools
                .iter()
                .filter(|t| names.iter().any(|n| n == t.name()))
                .cloned()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_builder_applies_config() {
        let rt = Runtime::builder().build();
        let agent = rt
            .agent()
            .cwd("/tmp/agent-builder-test")
            .model("anthropic", "claude-opus-4-8")
            .thinking("high")
            .only_tools(["read", "bash"])
            .metadata([("session", "demo")])
            .build();

        assert_eq!(
            agent.cwd(),
            std::path::Path::new("/tmp/agent-builder-test")
        );
        let model = agent.model().expect("model set");
        assert_eq!(
            (model.provider.as_str(), model.id.as_str()),
            ("anthropic", "claude-opus-4-8"),
        );
        assert_eq!(agent.thinking(), Some("high"));
        let tools: Vec<&str> =
            agent.tool_definitions().iter().map(|d| d.name).collect();
        assert_eq!(
            tools,
            ["read", "bash"],
            "only the selected tools, in registry order",
        );
        assert_eq!(
            agent.metadata_handle().get("session").map(String::as_str),
            Some("demo"),
        );
    }

    #[test]
    fn agent_builder_extends_runtime_hooks() {
        use crate::hook::Hook;
        struct Noop;
        impl Hook for Noop {}

        // One hook on the Runtime; the builder adds a second for this session.
        let rt = Runtime::builder().hook(Arc::new(Noop)).build();
        assert_eq!(
            rt.session(".".into()).hooks_handle().len(),
            1,
            "a plain session inherits the runtime's hook",
        );
        let agent = rt.agent().hook(Arc::new(Noop)).build();
        assert_eq!(
            agent.hooks_handle().len(),
            2,
            "the builder's hook adds to the runtime's, not replaces it",
        );
    }

    #[test]
    fn runtime_is_cloneable_and_shareable() {
        // The Runtime must be shareable across tasks (the SDK front door).
        fn assert_send_sync<T: Send + Sync + Clone>() {}
        assert_send_sync::<Runtime>();
    }

    #[test]
    fn a_spawned_session_starts_empty() {
        let rt = Runtime::builder().build();
        assert!(rt.session(".".into()).history().is_empty());
    }

    #[test]
    fn a_session_restricted_to_a_subset_excludes_other_tools() {
        // Capability scoped by construction: a session built with only `read`
        // cannot reach the other registered tools.
        let rt = Runtime::builder().build();
        let agent =
            rt.session_with(SessionOptions::new(".".into()).only(["read"]));
        assert!(
            agent.find_tool("read").is_some(),
            "the selected tool is present"
        );
        assert!(
            agent.find_tool("bash").is_none(),
            "an unselected tool is excluded"
        );
        assert!(
            agent.find_tool("write").is_none(),
            "an unselected tool is excluded"
        );
    }

    #[test]
    fn a_restricted_session_advertises_only_its_subset() {
        // The model-facing definitions are scoped to the session's tools, so an
        // excluded tool isn't even exposed.
        let rt = Runtime::builder().build();
        let agent = rt
            .session_with(SessionOptions::new(".".into()).only(["read", "ls"]));
        let names: Vec<&str> =
            agent.tool_definitions().iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            vec!["read", "ls"],
            "only the selected tools are advertised"
        );
    }

    // Independence — two sessions from one Runtime drive without cross-talk — is
    // exercised in `agent::tests` (it needs the turn-loop test runner there).

    /// A test Observer that records a label for every event it is notified of.
    #[derive(Default)]
    struct Recorder(std::sync::Mutex<Vec<String>>);

    impl Recorder {
        fn seen(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    impl crate::observer::Observer for Recorder {
        fn on_event(&self, event: &crate::agent::TurnEvent) {
            use crate::agent::TurnEvent::*;
            let label = match event {
                AgentStart => "agent_start".to_string(),
                AgentEnd => "agent_end".to_string(),
                Text { delta } => format!("text:{delta}"),
                Done => "done".to_string(),
                _ => "other".to_string(),
            };
            self.0.lock().unwrap().push(label);
        }
    }

    #[test]
    fn registered_tools_reach_the_spawned_session() {
        use crate::tools::ToolResult;
        use crate::tools::tool::{Tool, ToolCtx};
        use async_trait::async_trait;
        use serde_json::{Value, json};

        struct NoopTool;
        #[async_trait]
        impl Tool for NoopTool {
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
                _: &ToolCtx,
            ) -> anyhow::Result<ToolResult> {
                Ok(ToolResult::text("ok"))
            }
        }

        // The built-in read tool is registered by default; a custom tool added
        // on the builder reaches the spawned session too.
        let rt = Runtime::builder().tool(Arc::new(NoopTool)).build();
        let agent = rt.session(".".into());
        assert!(
            agent.find_tool("read").is_some(),
            "built-in read is registered"
        );
        assert!(
            agent.find_tool("noop").is_some(),
            "the custom tool reached the session"
        );
        assert!(
            agent.find_tool("nope").is_none(),
            "an unregistered name is absent"
        );
    }

    #[tokio::test]
    async fn a_plugin_command_runs_against_a_session_through_the_registry() {
        use crate::command::{Command, CommandCtx, NullUi};

        // A plugin command: registered on the builder, dispatched by name, its
        // async handler operates on the Session through the ctx.
        struct Hello;
        #[async_trait::async_trait]
        impl Command for Hello {
            fn name(&self) -> &str {
                "hello"
            }
            fn description(&self) -> &str {
                "seeds a greeting into the conversation"
            }
            async fn run(
                &self,
                args: &str,
                ctx: &mut CommandCtx<'_>,
            ) -> anyhow::Result<()> {
                ctx.session
                    .submit(crate::agent::Input::text(format!("hello {args}")));
                Ok(())
            }
        }

        let rt = Runtime::builder().command(Arc::new(Hello)).build();
        let mut session = rt.session(".".into());
        rt.run_command("hello", "world", &mut session, &mut NullUi)
            .await
            .expect("the registered command runs");
        assert!(
            matches!(
                session.history().last(),
                Some(crate::providers::Step::User { text, .. }) if text == "hello world"
            ),
            "the command's effect landed on the session",
        );

        // An unregistered name is an error the adapter can surface.
        let err = rt
            .run_command("nope", "", &mut session, &mut NullUi)
            .await
            .expect_err("an unknown command errors");
        assert!(
            err.to_string().contains("nope"),
            "the error names the command: {err}"
        );
    }

    #[tokio::test]
    async fn the_default_reset_and_new_commands_clear_the_conversation() {
        use crate::command::NullUi;

        // The conversation-level commands ship registered — no builder calls.
        let rt = Runtime::builder().build();
        for name in ["reset", "new"] {
            let mut session = rt.session(".".into());
            session.submit(crate::agent::Input::text("question"));
            session.push_assistant("answer");
            assert_eq!(session.history().len(), 2, "seeded");

            rt.run_command(name, "", &mut session, &mut NullUi)
                .await
                .unwrap_or_else(|e| {
                    panic!("the built-in {name} command runs: {e}")
                });
            assert!(
                session.history().is_empty(),
                "{name} cleared the conversation history",
            );
        }
    }

    /// A rich-mode test UI: records what a command shows. (The thin/absent mode
    /// is [`crate::command::NullUi`], used by the other command tests.)
    #[derive(Default)]
    struct RecordingUi(Vec<String>);
    impl crate::command::CommandUi for RecordingUi {
        fn notify(&mut self, message: &str) {
            self.0.push(message.to_string());
        }
    }

    #[tokio::test]
    async fn the_model_command_sets_and_reports_the_session_model() {
        let rt = Runtime::builder().build();
        let mut session = rt.session(".".into());
        assert!(
            session.model().is_none(),
            "a fresh session has no model selected"
        );

        // `model <provider>/<id>` selects the session's model…
        let mut ui = RecordingUi::default();
        rt.run_command("model", "openrouter/some-model", &mut session, &mut ui)
            .await
            .expect("selecting a model succeeds");
        let model = session.model().expect("the session now carries a model");
        assert_eq!(
            (model.provider.as_str(), model.id.as_str()),
            ("openrouter", "some-model")
        );

        // …and a bare `model` reports the current one through the adapter's UI.
        rt.run_command("model", "", &mut session, &mut ui)
            .await
            .expect("querying succeeds");
        assert!(
            ui.0.iter().any(|m| m.contains("openrouter/some-model")),
            "the current model is reported via the UI capability, got: {:?}",
            ui.0,
        );
    }

    #[tokio::test]
    async fn the_compact_command_summarizes_through_the_provider() {
        use crate::command::NullUi;
        use crate::providers::Step;

        // A real provider round, no UI: the command resolves the session's
        // model + provider + credentials through the Runtime, summarizes the
        // history, and applies the checkpoint — bot-grade compaction. The mock
        // SSE server plays openrouter (the deepseek override belongs to another
        // test; separate vars keep parallel tests from racing).
        let port = crate::providers::testing::spawn_mock_sse(vec![
            crate::providers::testing::sse_text("THE-SUMMARY"),
        ]);
        // SAFETY: only this test reads the openrouter base-URL override.
        unsafe {
            std::env::set_var(
                "TAPIR_BASE_URL_OPENROUTER",
                format!("http://127.0.0.1:{port}"),
            )
        };
        crate::providers::set_runtime_api_key("openrouter", "sk-test");

        let rt = Runtime::builder().build();
        let mut session = rt.session(".".into());
        rt.run_command(
            "model",
            "openrouter/some-model",
            &mut session,
            &mut NullUi,
        )
        .await
        .expect("the model command selects the session model");
        session.submit(crate::agent::Input::text("an old question"));
        session.push_assistant("an old answer");

        rt.run_command("compact", "", &mut session, &mut NullUi)
            .await
            .expect("compaction runs through the provider");

        assert_eq!(
            session.history().len(),
            1,
            "the checkpoint replaced the conversation"
        );
        let history = session.history();
        let Some(Step::User { text, .. }) = history.first() else {
            panic!("the checkpoint is a user step");
        };
        assert!(
            text.contains("THE-SUMMARY"),
            "the provider's summary is the context: {text}"
        );
        assert!(!text.contains("an old question"), "the old turns are gone");

        unsafe { std::env::remove_var("TAPIR_BASE_URL_OPENROUTER") };
    }

    #[tokio::test]
    async fn a_stored_conversation_resumes_through_the_runtime() {
        use crate::agent::Role;
        use crate::providers::Step;
        use crate::store::{Entry, InMemoryStore, SessionStore};

        // A conversation persisted under an identifier — the in-memory store is
        // the test double every engine test reuses.
        let store = Arc::new(InMemoryStore::default());
        let rt = Runtime::builder().store(store.clone()).build();
        store
            .append(
                "conv-1",
                &Entry::Message {
                    role: Role::User,
                    text: "first question".into(),
                },
            )
            .await
            .unwrap();
        store
            .append(
                "conv-1",
                &Entry::Message {
                    role: Role::Assistant,
                    text: "first answer".into(),
                },
            )
            .await
            .unwrap();

        // The engine drives resume through the trait: the rebuilt session
        // carries the stored conversation as its durable history.
        let agent = rt
            .resume(SessionOptions::new(".".into()), "conv-1")
            .await
            .expect("the id resumes");
        let history = agent.history();
        let texts: Vec<&str> = history
            .iter()
            .map(|s| match s {
                Step::User { text, .. } => text.as_str(),
                Step::Assistant { text, .. } => text.as_str(),
                Step::ToolResult { output, .. } => output.as_str(),
            })
            .collect();
        assert_eq!(
            texts,
            ["first question", "first answer"],
            "the resumed history replays the stored conversation in order",
        );
    }

    #[tokio::test]
    async fn an_injected_credential_resolver_feeds_provider_creds() {
        use crate::credentials::CredentialProvider;
        use crate::providers::Creds;

        // A test resolver: keys derived from the provider id — no file, no env,
        // no network. Injection means a bot supplies its own secret store.
        struct TestResolver;
        #[async_trait::async_trait]
        impl CredentialProvider for TestResolver {
            async fn resolve(
                &self,
                _client: &reqwest::Client,
                provider: &str,
            ) -> anyhow::Result<Creds> {
                Ok(Creds::ApiKey { key: format!("sk-test-{provider}") })
            }
        }

        // Injected on the builder, the resolver is what a Provider's default
        // credential path consults — the engine performs no acquisition itself.
        let rt = Runtime::builder().credentials(Arc::new(TestResolver)).build();
        let p = rt
            .find_provider("deepseek")
            .expect("built-in deepseek is registered");
        let client = crate::providers::base_client(5);
        let creds = p
            .creds(&client, rt.credentials().as_ref())
            .await
            .expect("the resolver resolved");
        assert!(
            matches!(creds, Creds::ApiKey { ref key } if key == "sk-test-deepseek"),
            "the provider obtained its credentials through the injected resolver",
        );
    }

    #[test]
    fn builtin_providers_are_registered_with_catalog_models() {
        // The six built-in providers are pre-registered (like built-in tools)…
        let rt = Runtime::builder().build();
        for id in [
            "copilot",
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "openrouter",
        ] {
            assert!(
                rt.find_provider(id).is_some(),
                "built-in {id} is registered"
            );
        }
        // …and advertise the embedded catalog's models — the catalog keeps
        // feeding the Footer and model picker; registered providers coexist.
        let anthropic = rt.find_provider("anthropic").unwrap();
        let ids: Vec<String> =
            anthropic.models().iter().map(|m| m.id.clone()).collect();
        assert!(!ids.is_empty(), "the built-in advertises models");
        assert_eq!(
            ids,
            crate::catalog::models_for("anthropic"),
            "models come from the catalog"
        );
    }

    /// A Provider that replays scripted text rounds — the offline mock the
    /// `run(input)` seam tests drive whole turns through.
    struct ScriptedProvider {
        id: &'static str,
        rounds: std::sync::Mutex<std::collections::VecDeque<&'static str>>,
        /// What each round received: (instructions, effort) — for prompt-
        /// assembly assertions.
        seen: std::sync::Mutex<Vec<(String, Option<String>)>>,
    }

    impl ScriptedProvider {
        fn new(id: &'static str, rounds: &[&'static str]) -> Self {
            Self {
                id,
                rounds: std::sync::Mutex::new(rounds.iter().copied().collect()),
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn seen(&self) -> Vec<(String, Option<String>)> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::providers::Provider for ScriptedProvider {
        fn id(&self) -> &str {
            self.id
        }
        fn models(&self) -> Vec<crate::providers::ModelInfo> {
            Vec::new()
        }
        // Offline mock: its own credential authority (no resolver chain).
        async fn creds(
            &self,
            _client: &reqwest::Client,
            _resolver: &dyn crate::credentials::CredentialProvider,
        ) -> anyhow::Result<crate::providers::Creds> {
            Ok(crate::providers::Creds::ApiKey { key: "scripted".into() })
        }
        async fn stream(
            &self,
            ctx: &crate::providers::RoundCtx<'_>,
            _history: &[crate::providers::Step],
        ) -> Result<crate::providers::RoundOutcome, crate::agent::RoundError>
        {
            self.seen.lock().unwrap().push((
                ctx.instructions.to_string(),
                ctx.effort.map(str::to_string),
            ));
            let text = self
                .rounds
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted rounds left");
            let _ = ctx
                .tx
                .send(crate::agent::TurnEvent::Text { delta: text.into() })
                .await;
            Ok(crate::providers::RoundOutcome {
                usage: Default::default(),
                tool_calls: Vec::new(),
                assistant: crate::providers::Step::Assistant {
                    text: text.into(),
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    raw: None,
                },
            })
        }
    }

    /// Drain a turn's stream with a per-event timeout, collecting text, the
    /// terminal marker, and any error messages (for assertions and diagnosis).
    async fn drain_turn(
        rx: &mut tokio::sync::mpsc::Receiver<crate::agent::TurnEvent>,
    ) -> (String, bool, Vec<String>) {
        use crate::agent::TurnEvent;
        let cap = std::time::Duration::from_secs(10);
        let (mut text, mut done, mut errors) =
            (String::new(), false, Vec::new());
        while let Ok(Some(ev)) = tokio::time::timeout(cap, rx.recv()).await {
            match ev {
                TurnEvent::Text { delta } => text.push_str(&delta),
                TurnEvent::Done => done = true,
                TurnEvent::Error { message } => errors.push(message),
                _ => {}
            }
        }
        (text, done, errors)
    }

    #[tokio::test]
    async fn a_turn_is_one_call_on_the_session() {
        use crate::agent::{Input, ModelRef};
        use crate::providers::Step;

        // The deepened seam: run(input) returns the Event stream, and the
        // *held* Session's history carries the conversation afterwards — no
        // ephemeral agent, no event mirroring, no idle-marking by the caller.
        let rt = Runtime::builder()
            .provider(Arc::new(ScriptedProvider::new(
                "scripted",
                &["the reply"],
            )))
            .build();
        let mut session = rt.session(".".into());
        session.set_model(Some(ModelRef {
            provider: "scripted".into(),
            id: "model-1".into(),
        }));

        let mut rx = session.run(Input::text("the question"));
        let (text, done, errors) = drain_turn(&mut rx).await;
        assert!(errors.is_empty(), "the turn errored: {errors:?}");
        assert_eq!(text, "the reply", "the scripted round streamed");
        assert!(done, "the turn completed");

        // The held Session's history grew by itself — the turn task wrote the
        // same conversation the Session reads.
        let history = session.history();
        assert_eq!(
            history.len(),
            2,
            "user input + assistant reply: {history:?}"
        );
        assert!(
            matches!(&history[0], Step::User { text, .. } if text == "the question")
        );
        assert!(
            matches!(&history[1], Step::Assistant { text, .. } if text == "the reply")
        );

        // And the running flag cleared with no set_idle from the caller.
        assert!(
            !session.is_running(),
            "the stream's end cleared the running flag"
        );
    }

    #[tokio::test]
    async fn the_turn_prompt_is_assembled_from_the_session_knobs() {
        use crate::agent::{Input, ModelRef};

        // A project with real context and a real skill on disk — the engine
        // assembles the prompt per turn from the Session's knobs, matching the
        // TUI's assembly (tools list, append, context, skills, date/cwd tail).
        let cwd = std::env::temp_dir()
            .join(format!("tapir-knobs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cwd);
        std::fs::create_dir_all(cwd.join(".tapir/skills/marker-skill"))
            .unwrap();
        std::fs::write(cwd.join("AGENTS.md"), "CTX-MARKER-77").unwrap();
        std::fs::write(
            cwd.join(".tapir/skills/marker-skill/SKILL.md"),
            "---\nname: marker-skill\ndescription: SKILL-MARKER-77\n---\nbody\n",
        )
        .unwrap();

        let provider = Arc::new(ScriptedProvider::new("scripted", &["ok"]));
        let rt = Runtime::builder().provider(provider.clone()).build();
        let mut session = rt.session_with(
            SessionOptions::new(cwd.clone()).prompt(PromptSpec {
                context: true,
                skills: true,
                skill_paths: Vec::new(),
                trust_project: true,
                append: Some(vec!["APPEND-MARKER-77".into()]),
                custom: None,
            }),
        );
        session.set_model(Some(ModelRef {
            provider: "scripted".into(),
            id: "m".into(),
        }));

        let mut rx = session.run(Input::text("hi"));
        let (_, done, errors) = drain_turn(&mut rx).await;
        assert!(errors.is_empty(), "the turn errored: {errors:?}");
        assert!(done);

        let seen = provider.seen();
        let prompt = &seen[0].0;
        assert!(
            prompt.contains("CTX-MARKER-77"),
            "context files loaded per turn"
        );
        assert!(prompt.contains("SKILL-MARKER-77"), "project skills listed");
        assert!(
            prompt.contains("APPEND-MARKER-77"),
            "the append block is merged"
        );
        assert!(
            prompt.contains("Available tools:"),
            "the tools list is present"
        );
        assert!(
            prompt.contains("- read:"),
            "the session's tools are advertised"
        );
        assert!(
            prompt.contains("Current date and time:"),
            "the grounding tail"
        );
        assert!(
            prompt.contains(&cwd.display().to_string()),
            "the session's cwd grounds it"
        );
        // The documented order: tools → append → context → date/cwd tail.
        let tools_at = prompt.find("Available tools:").unwrap();
        let append_at = prompt.find("APPEND-MARKER-77").unwrap();
        let ctx_at = prompt.find("CTX-MARKER-77").unwrap();
        let date_at = prompt.find("Current date and time:").unwrap();
        assert!(tools_at < append_at && append_at < ctx_at && ctx_at < date_at);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[tokio::test]
    async fn unset_knobs_keep_the_minimal_prompt_and_custom_replaces_the_body()
    {
        use crate::agent::{Input, ModelRef};

        // A cwd that HAS context on disk — but a session created without knobs
        // must not read it (defaults preserve the minimal prompt; loading
        // project files is an explicit opt-in).
        let cwd = std::env::temp_dir()
            .join(format!("tapir-minimal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cwd);
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(cwd.join("AGENTS.md"), "CTX-MARKER-88").unwrap();

        let provider =
            Arc::new(ScriptedProvider::new("scripted", &["ok", "ok"]));
        let rt = Runtime::builder().provider(provider.clone()).build();
        let mut session = rt.session(cwd.clone());
        session.set_model(Some(ModelRef {
            provider: "scripted".into(),
            id: "m".into(),
        }));
        let mut rx = session.run(Input::text("hi"));
        let (_, _, errors) = drain_turn(&mut rx).await;
        assert!(errors.is_empty(), "{errors:?}");
        let prompt = provider.seen()[0].0.clone();
        assert!(
            !prompt.contains("CTX-MARKER-88"),
            "unset knobs read no context files"
        );
        assert!(
            prompt.contains("Available tools:"),
            "the default body is present"
        );
        assert!(
            prompt.contains("Current date and time:"),
            "the grounding tail stays"
        );

        // A custom override replaces the body but keeps the tail (the same
        // contract the TUI's --system-prompt has).
        let mut session = rt.session_with(
            SessionOptions::new(cwd.clone()).prompt(PromptSpec {
                custom: Some("CUSTOM-ROOT-88".into()),
                ..Default::default()
            }),
        );
        session.set_model(Some(ModelRef {
            provider: "scripted".into(),
            id: "m".into(),
        }));
        let mut rx = session.run(Input::text("hi"));
        let (_, _, errors) = drain_turn(&mut rx).await;
        assert!(errors.is_empty(), "{errors:?}");
        let prompt = provider.seen()[1].0.clone();
        assert!(
            prompt.starts_with("CUSTOM-ROOT-88"),
            "the custom prompt replaces the body"
        );
        assert!(
            !prompt.contains("Available tools:"),
            "the default body is gone"
        );
        assert!(
            prompt.contains("Current date and time:"),
            "the tail still grounds it"
        );

        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[tokio::test]
    async fn the_session_thinking_level_maps_to_the_provider_effort() {
        use crate::agent::{Input, ModelRef};

        // The model-slot treatment for thinking: a settable Session slot,
        // mapped to the provider effort each turn. Default is medium.
        async fn effort_for(level: Option<&str>) -> Option<String> {
            let provider = Arc::new(ScriptedProvider::new("scripted", &["ok"]));
            let rt = Runtime::builder().provider(provider.clone()).build();
            let mut session = rt.session(".".into());
            session.set_model(Some(ModelRef {
                provider: "scripted".into(),
                id: "m".into(),
            }));
            if let Some(level) = level {
                session.set_thinking(level);
            }
            let mut rx = session.run(Input::text("hi"));
            let (_, _, errors) = drain_turn(&mut rx).await;
            assert!(errors.is_empty(), "the turn errored: {errors:?}");
            provider.seen()[0].1.clone()
        }

        assert_eq!(
            effort_for(None).await.as_deref(),
            Some("medium"),
            "default is medium"
        );
        assert_eq!(effort_for(Some("high")).await.as_deref(), Some("high"));
        assert_eq!(
            effort_for(Some("off")).await,
            None,
            "off sends no reasoning"
        );
    }

    #[tokio::test]
    async fn steering_joins_a_running_turn_on_the_session() {
        use crate::agent::{Input, ModelRef, TurnEvent};

        // A steering message queued ahead of run(input) is delivered at the
        // first idle round boundary, extending the turn by a round — and the
        // held Session's history carries it, with no mirroring by the caller.
        let rt = Runtime::builder()
            .provider(Arc::new(ScriptedProvider::new(
                "scripted",
                &["first", "second"],
            )))
            .build();
        let mut session = rt.session(".".into());
        session.set_model(Some(ModelRef {
            provider: "scripted".into(),
            id: "m".into(),
        }));
        session.steer(Input::text("also do this"));

        let mut rx = session.run(Input::text("go"));
        let cap = std::time::Duration::from_secs(10);
        let (mut queued, mut text) = (None, String::new());
        while let Ok(Some(ev)) = tokio::time::timeout(cap, rx.recv()).await {
            match ev {
                TurnEvent::QueuedDelivered { display, .. } => {
                    queued = Some(display)
                }
                TurnEvent::Text { delta } => text.push_str(&delta),
                _ => {}
            }
        }
        assert_eq!(
            queued.as_deref(),
            Some("also do this"),
            "the steer was delivered"
        );
        assert!(
            text.contains("first") && text.contains("second"),
            "two rounds ran: {text}"
        );
        let history = session.history();
        assert!(
            history
                .iter()
                .any(|s| matches!(s, crate::providers::Step::User { text, .. }
                if text == "also do this")),
            "the steer entered the held history by itself: {history:?}",
        );
    }

    #[tokio::test]
    async fn abort_ends_a_running_turn_on_the_session() {
        use crate::agent::{Input, ModelRef};

        // A provider that never answers — only abort can end the turn.
        struct Hanging;
        #[async_trait::async_trait]
        impl crate::providers::Provider for Hanging {
            fn id(&self) -> &str {
                "hanging"
            }
            fn models(&self) -> Vec<crate::providers::ModelInfo> {
                Vec::new()
            }
            async fn creds(
                &self,
                _client: &reqwest::Client,
                _resolver: &dyn crate::credentials::CredentialProvider,
            ) -> anyhow::Result<crate::providers::Creds> {
                Ok(crate::providers::Creds::ApiKey { key: "hanging".into() })
            }
            async fn stream(
                &self,
                _ctx: &crate::providers::RoundCtx<'_>,
                _history: &[crate::providers::Step],
            ) -> Result<crate::providers::RoundOutcome, crate::agent::RoundError>
            {
                std::future::pending().await
            }
        }

        let rt = Runtime::builder().provider(Arc::new(Hanging)).build();
        let mut session = rt.session(".".into());
        session.set_model(Some(ModelRef {
            provider: "hanging".into(),
            id: "m".into(),
        }));

        let mut rx = session.run(Input::text("hang"));
        // The lifecycle opens (the turn is genuinely in flight)…
        let first =
            tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
                .await
                .expect("an opening event arrives");
        assert!(first.is_some(), "the turn started");
        assert!(session.is_running(), "the turn is in flight");

        // …abort ends the stream and leaves the session idle.
        session.abort();
        let cap = std::time::Duration::from_secs(10);
        while let Ok(Some(_)) = tokio::time::timeout(cap, rx.recv()).await {}
        assert!(!session.is_running(), "abort left the session idle");
    }

    #[tokio::test]
    async fn run_surfaces_setup_failures_in_channel() {
        use crate::agent::{Input, ModelRef};

        // Every failure arrives as an Error event on the one stream the caller
        // already consumes — no Result to match, no second error path.

        // No model selected:
        let rt = Runtime::builder().build();
        let mut session = rt.session(".".into());
        let mut rx = session.run(Input::text("hi"));
        let (_, done, errors) = drain_turn(&mut rx).await;
        assert!(!done, "a failed setup is not a successful turn");
        assert!(
            errors.iter().any(|e| e.contains("no model selected")),
            "the missing model surfaces in-channel: {errors:?}",
        );
        assert!(
            !session.is_running(),
            "a failed setup still leaves the session idle"
        );

        // Unknown provider:
        let mut session = rt.session(".".into());
        session.set_model(Some(ModelRef {
            provider: "nope".into(),
            id: "x".into(),
        }));
        let mut rx = session.run(Input::text("hi"));
        let (_, _, errors) = drain_turn(&mut rx).await;
        assert!(
            errors.iter().any(|e| e.contains("unknown provider: nope")),
            "the unknown provider surfaces in-channel: {errors:?}",
        );
    }

    #[tokio::test]
    async fn a_registered_provider_streams_a_round_through_the_trait() {
        use crate::agent::{RoundError, TurnEvent};
        use crate::providers::{
            Creds, ModelInfo, Provider, RoundCtx, RoundOutcome, Step,
        };

        // A deterministic mock Provider — one scripted text round, no network.
        // The maintainer seam the PRD calls for: engine behavior testable offline.
        struct MockProvider;
        #[async_trait::async_trait]
        impl Provider for MockProvider {
            fn id(&self) -> &str {
                "mock"
            }
            fn models(&self) -> Vec<ModelInfo> {
                vec![ModelInfo {
                    id: "mock-1".into(),
                    name: "Mock One".into(),
                    ..Default::default()
                }]
            }
            async fn stream(
                &self,
                ctx: &RoundCtx<'_>,
                _history: &[Step],
            ) -> Result<RoundOutcome, RoundError> {
                let _ = ctx
                    .tx
                    .send(TurnEvent::Text { delta: "scripted".into() })
                    .await;
                Ok(RoundOutcome {
                    usage: Default::default(),
                    tool_calls: Vec::new(),
                    assistant: Step::Assistant {
                        text: "scripted".into(),
                        thinking: String::new(),
                        tool_calls: Vec::new(),
                        raw: None,
                    },
                })
            }
        }

        // Registered on the builder, the provider is resolvable by id and
        // advertises its models — the registry contract adapters rely on.
        let rt = Runtime::builder().provider(Arc::new(MockProvider)).build();
        let p = rt
            .find_provider("mock")
            .expect("the registered provider is resolvable by id");
        assert_eq!(
            p.models().first().map(|m| m.id.clone()).as_deref(),
            Some("mock-1")
        );
        assert!(
            rt.find_provider("nope").is_none(),
            "an unregistered id is absent"
        );

        // One round streams through the trait object — the seam every turn uses.
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let client = crate::providers::retrying_client(
            crate::providers::base_client(5),
            0,
        );
        let creds = Creds::ApiKey { key: "unused".into() };
        let ctx = RoundCtx {
            client: &client,
            provider: p.id(),
            creds: &creds,
            model: "mock-1",
            instructions: "",
            tools: &[],
            effort: None,
            tx: &tx,
        };
        let history = [Step::User { text: "hi".into(), images: Vec::new() }];
        let outcome =
            p.stream(&ctx, &history).await.expect("the round streams");
        drop(tx);

        assert!(
            matches!(outcome.assistant, Step::Assistant { ref text, .. } if text == "scripted"),
            "the trait returned the scripted round outcome",
        );
        assert!(
            matches!(rx.recv().await, Some(TurnEvent::Text { delta }) if delta == "scripted"),
            "the delta streamed over the event channel",
        );
    }

    #[tokio::test]
    async fn a_runtime_observer_is_notified_of_every_event() {
        // An Observer registered on the Runtime sees every event of a session's
        // turn, in order — the seam a bot/logger/hook hangs off.
        let recorder = Arc::new(Recorder::default());
        let rt = Runtime::builder().observer(recorder.clone()).build();
        let mut agent = rt.session(".".into());

        let mut rx = agent.run_with(|tx| async move {
            use crate::agent::TurnEvent;
            let _ = tx.send(TurnEvent::AgentStart).await;
            let _ = tx.send(TurnEvent::Text { delta: "hi".into() }).await;
            let _ = tx.send(TurnEvent::Done).await;
            let _ = tx.send(TurnEvent::AgentEnd).await;
        });
        // Bounded so a stuck relay fails the test fast instead of hanging.
        let cap = std::time::Duration::from_secs(10);
        while let Ok(Some(_)) = tokio::time::timeout(cap, rx.recv()).await {}

        assert_eq!(
            recorder.seen(),
            vec!["agent_start", "text:hi", "done", "agent_end"],
            "the observer is notified of every event in order",
        );
    }
}
