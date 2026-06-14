//! The frontend-agnostic [`Command`] registry.
//!
//! A [`Command`] is the SDK's pluggable conversation operation: a name and an
//! async handler over the Session/Runtime, registered on the
//! [`RuntimeBuilder`](crate::runtime::RuntimeBuilder) and dispatched by name —
//! the same command works across Adapters. The handler's [`CommandCtx`] carries
//! a [`CommandUi`] whose capabilities vary by Adapter mode: rich in the TUI,
//! thin or absent (a [`NullUi`]) in a bot or RPC adapter.
//!
//! The TUI's slash-*menu* (the command list shown while typing, with fuzzy
//! matching) is presentation and lives in the frontend (`tapir-tui`'s `menu`).

use async_trait::async_trait;

/// A pluggable conversation-level command, dispatched by name through the
/// Runtime's registry. Async (the handler may run a model round — `/compact`
/// does) and object-safe so a heterogeneous set lives in one registry.
//
// Adapter-facing: the TUI still dispatches its slash commands directly (same
// behavior, richer UI), so in this binary the registry surface is exercised by
// the registry seam tests; adapters dispatch through Runtime::run_command.
#[async_trait]
pub trait Command: Send + Sync {
    /// The command's name, without the leading slash (matched on dispatch).
    fn name(&self) -> &str;
    /// One-line description (menus, help).
    fn description(&self) -> &str;
    /// Run the command with `args` (the text after the name, trimmed).
    async fn run(
        &self,
        args: &str,
        ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()>;
}

/// What a running command operates on: the conversation (Session), the shared
/// wiring (Runtime), and a UI whose capabilities vary by Adapter mode.
// Adapter-facing — see the note on `Command`.
pub struct CommandCtx<'a> {
    pub session: &'a mut crate::agent::Agent,
    pub runtime: &'a crate::runtime::Runtime,
    pub ui: &'a mut dyn CommandUi,
}

/// The UI capabilities an Adapter lends a running command. Every method has a
/// no-op default, so a thin adapter implements nothing and a rich one (the
/// TUI) overrides what it can show.
// Adapter-facing — see the note on `Command`.
pub trait CommandUi: Send {
    /// Show a short status message to the user (the TUI's status line; a bot
    /// might post it to the channel). Default: dropped.
    fn notify(&mut self, _message: &str) {}
}

/// The absent-UI mode: every capability is the no-op default — how a command
/// runs in a headless adapter (or a test that asserts only session effects).
// Adapter-facing — see the note on `Command`.
pub struct NullUi;

impl CommandUi for NullUi {}

/// The conversation-level commands every Runtime ships with — the same set
/// across adapters. The TUI's `/new` additionally rotates the session file;
/// that frontend layer stays in the TUI.
// Adapter-facing — see the note on `Command`.
pub fn builtin_commands() -> Vec<std::sync::Arc<dyn Command>> {
    vec![
        std::sync::Arc::new(Reset {
            name: "reset",
            description: "Clear the conversation",
        }),
        std::sync::Arc::new(Reset {
            name: "new",
            description: "Start a new conversation",
        }),
        std::sync::Arc::new(Model),
        std::sync::Arc::new(Compact),
    ]
}

/// `reset` / `new`: clear the conversation history. At the engine level the
/// two are the same operation; frontends layer their extras (a fresh session
/// file, a status line) on top.
struct Reset {
    name: &'static str,
    description: &'static str,
}

#[async_trait]
impl Command for Reset {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        self.description
    }
    async fn run(
        &self,
        _args: &str,
        ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        ctx.session.reset_history();
        ctx.ui.notify("Conversation cleared.");
        Ok(())
    }
}

/// `model [<provider>/<id>]`: point the session at a model, or report the
/// current selection. The argument names a registered provider and one of its
/// model ids — the same reference a turn is spawned with.
struct Model;

#[async_trait]
impl Command for Model {
    fn name(&self) -> &str {
        "model"
    }
    fn description(&self) -> &str {
        "Switch the session's model (provider/id), or show the current one"
    }
    async fn run(
        &self,
        args: &str,
        ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        if args.is_empty() {
            match ctx.session.model() {
                Some(m) => ctx.ui.notify(&format!("Model: {m}")),
                None => {
                    ctx.ui.notify("No model selected — model <provider>/<id>")
                }
            }
            return Ok(());
        }
        let (provider, id) = args
            .split_once('/')
            .map(|(p, m)| (p.trim(), m.trim()))
            .filter(|(p, m)| !p.is_empty() && !m.is_empty())
            .ok_or_else(|| anyhow::anyhow!("usage: model <provider>/<id>"))?;
        ctx.session.set_model(Some(crate::agent::ModelRef {
            provider: provider.to_string(),
            id: id.to_string(),
        }));
        ctx.ui.notify(&format!("Model set to {provider}/{id}."));
        Ok(())
    }
}

/// `compact [instructions]`: summarize the conversation into a checkpoint —
/// resolve the session's model and its Provider through the Runtime, run one
/// summarization round, and replace the history with the summary. The async
/// handler that justifies the trait being async: a bot compacts exactly like
/// the TUI without reimplementing the round.
struct Compact;

#[async_trait]
impl Command for Compact {
    fn name(&self) -> &str {
        "compact"
    }
    fn description(&self) -> &str {
        "Compact the context now (optional instructions)"
    }
    async fn run(
        &self,
        args: &str,
        ctx: &mut CommandCtx<'_>,
    ) -> anyhow::Result<()> {
        let model = ctx.session.model().cloned().ok_or_else(|| {
            anyhow::anyhow!("no model selected — run model <provider>/<id>")
        })?;
        if ctx.session.history().is_empty() {
            return Err(anyhow::anyhow!("nothing to compact yet"));
        }
        let provider =
            ctx.runtime.find_provider(&model.provider).ok_or_else(|| {
                anyhow::anyhow!("unknown provider: {}", model.provider)
            })?;

        let config = ctx.runtime.config();
        let base = crate::providers::base_client(
            config.http_timeout_secs.unwrap_or(120),
        );
        let creds =
            provider.creds(&base, ctx.runtime.credentials().as_ref()).await?;
        let client = crate::providers::retrying_client(
            base,
            config.http_retries.unwrap_or(2),
        );

        // The conversation plus the summary request as a trailing user turn;
        // `compact <instructions>` overrides the default structured prompt.
        let prompt =
            if args.is_empty() { crate::agent::SUMMARY_PROMPT } else { args };
        let mut history = ctx.session.history();
        history.push(crate::providers::Step::User {
            text: prompt.to_string(),
            images: Vec::new(),
        });
        let summary = crate::providers::summarize(
            &client,
            provider.as_ref(),
            &creds,
            &model.id,
            "",
            &history,
            None,
        )
        .await
        .map_err(|e| anyhow::anyhow!(e.message))?;
        if summary.trim().is_empty() {
            return Err(anyhow::anyhow!("the model returned an empty summary"));
        }
        ctx.session.compact(&summary);
        ctx.ui.notify("Context compacted.");
        Ok(())
    }
}
