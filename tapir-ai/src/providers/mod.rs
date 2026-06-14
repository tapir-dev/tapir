//! Multi-provider chat — pluggable [`Provider`]s over one streaming client per
//! API shape, behind a neutral conversation history so the round loop never
//! sees wire formats.
//!
//! A [`Provider`] (id, model list, credentials, one streamed round) registers
//! on the `Runtime` (in `tapir-core`) and is dispatched by id. The
//! built-ins are [`WireProvider`] instances over four shapes:
//! - **Responses** (`responses`) — GitHub Copilot's OpenAI Responses API.
//! - **Chat** (`chat`) — OpenAI Chat Completions: OpenAI, DeepSeek, OpenRouter.
//! - **Anthropic** (`anthropic`) — the Messages API.
//! - **Gemini** (`gemini`) — Google's `streamGenerateContent`.
//!
//! A custom endpoint over an existing shape (an OpenAI-compatible server) is a
//! [`WireProvider`] with an endpoint, credentials, and a model list — no new
//! streaming code; implementing the trait is only for genuinely new protocols.
//!
//! Each shape client serializes the shared [`Step`] history into its own request
//! and parses its own stream back into a [`RoundOutcome`], streaming
//! text/thinking deltas to the UI via the channel. Credentials resolve through
//! [`Provider::creds`], which consults the Runtime's injected
//! [`CredentialProvider`](crate::credentials::CredentialProvider) (default:
//! [`FileCreds`](crate::credentials::FileCreds) — Copilot via OAuth, everyone
//! else via an API key from `--api-key`, env var, then `auth.toml`).

pub mod anthropic;
pub mod chat;
pub mod gemini;
pub mod responses;

use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::RetryTransientMiddleware;
use reqwest_retry::policies::ExponentialBackoff;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::auth;
use crate::message::{Image, RoundError, ToolCall, ToolDef, TurnEvent, Usage};

/// One step of the neutral conversation. Every client rebuilds its wire request
/// from a `&[Step]` each round.
#[derive(Debug, Clone)]
pub enum Step {
    User {
        text: String,
        images: Vec<Image>,
    },
    Assistant {
        text: String,
        thinking: String,
        tool_calls: Vec<ToolCall>,
        /// Verbatim Responses output items, kept only for Copilot so its echo
        /// (incl. encrypted reasoning) is byte-for-byte identical. Other shapes
        /// rebuild from `text`/`tool_calls` and ignore this.
        raw: Option<Vec<Value>>,
    },
    ToolResult {
        call_id: String,
        name: String,
        output: String,
        is_error: bool,
    },
}

/// What one streamed round produced: usage, any tool calls to run, and the
/// assistant step to append to the history (already streamed to the UI).
pub struct RoundOutcome {
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
    pub assistant: Step,
}

/// The HTTP API shape a provider speaks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Api {
    Responses,
    Chat,
    Anthropic,
    Gemini,
}

/// One model a [`Provider`] serves — owned, so a custom provider can build its
/// list at runtime. Built-ins convert from the embedded catalog
/// ([`crate::catalog::models::ModelInfo`], `&'static` data), which keeps feeding
/// the Footer and model picker; the two coexist. Costs are USD per million
/// tokens; zeroes mean unknown.
//
// The metadata fields beyond `id` are adapter-facing (an adapter's picker reads
// them off the trait); the TUI's Footer/picker read the catalog directly, so in
// the engine only the seam tests touch them; adapters read them off the trait.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub context_window: u32,
    pub max_output: u32,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read: f64,
    pub reasoning: bool,
}

impl From<&crate::catalog::models::ModelInfo> for ModelInfo {
    fn from(m: &crate::catalog::models::ModelInfo) -> Self {
        Self {
            id: m.id.to_string(),
            name: m.name.to_string(),
            context_window: m.context_window,
            max_output: m.max_output,
            input_cost: m.input_cost,
            output_cost: m.output_cost,
            cache_read: m.cache_read,
            reasoning: m.reasoning,
        }
    }
}

/// A pluggable model provider: an identifier, the models it serves, and one
/// streamed round of a turn. Implementations are registered on the `Runtime`
/// (in `tapir-core`) and dispatched by id — the SDK's seam
/// for new wire protocols. Object-safe (`async-trait` boxes the future) so a
/// heterogeneous set lives in one registry.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// The provider's identifier (matched against the active provider id).
    fn id(&self) -> &str;
    /// The models this provider serves (offline metadata for pickers/footers).
    // Adapter-facing (an adapter lists a session's choices from the trait); the
    // TUI's picker reads the catalog, so only the seam tests call it here.
    fn models(&self) -> Vec<ModelInfo>;
    /// Resolve this provider's credentials for a turn through the Runtime's
    /// injected [`CredentialProvider`](crate::credentials::CredentialProvider)
    /// — the default just asks it for this id. A provider constructed with an
    /// explicit key overrides this and ignores the resolver.
    async fn creds(
        &self,
        client: &reqwest::Client,
        resolver: &dyn crate::credentials::CredentialProvider,
    ) -> anyhow::Result<Creds> {
        resolver.resolve(client, self.id()).await
    }
    /// Stream one round: serialize `history` into the wire request, stream
    /// text/thinking deltas to `ctx.tx`, and return the round's outcome.
    async fn stream(
        &self,
        ctx: &RoundCtx<'_>,
        history: &[Step],
    ) -> Result<RoundOutcome, RoundError>;
}

/// A shared, immutable list of [`Provider`]s — the Runtime's registry.
pub type Providers = std::sync::Arc<[std::sync::Arc<dyn Provider>]>;

/// A [`Provider`] over one of the built-in wire shapes: an id and the [`Api`]
/// the endpoint speaks. The six built-ins are instances of this; a custom
/// OpenAI-compatible (or other-shape) endpoint registers another with **no new
/// streaming code** — the wire serialization and SSE accumulation are the
/// shape's, shared.
pub struct WireProvider {
    id: String,
    api: Api,
    /// An explicit API base URL; `None` keeps the shape's default for the id.
    endpoint: Option<String>,
    /// An explicit API key; `None` keeps the engine's resolution chain.
    api_key: Option<String>,
    /// An explicit model list; `None` advertises the embedded catalog's.
    models: Option<Vec<ModelInfo>>,
}

impl WireProvider {
    /// A provider named `id` speaking the `api` wire shape. Built-ins advertise
    /// the embedded catalog's models for their id; the builders below override
    /// the endpoint, credentials, and model list for a custom one.
    pub fn new(id: impl Into<String>, api: Api) -> Self {
        Self { id: id.into(), api, endpoint: None, api_key: None, models: None }
    }

    /// Point the provider at this API base URL (a custom endpoint).
    // Adapter/embedder surface (with `api_key`/`models` below): exercised by the
    // seam tests; an embedder wires a custom endpoint through these.
    pub fn endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = Some(url.into());
        self
    }

    /// Authenticate with this explicit API key instead of the engine's chain.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Advertise these model ids instead of the embedded catalog's.
    pub fn models<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.models = Some(
            ids.into_iter()
                .map(|id| {
                    let id = id.into();
                    ModelInfo { name: id.clone(), id, ..Default::default() }
                })
                .collect(),
        );
        self
    }
}

#[async_trait::async_trait]
impl Provider for WireProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn models(&self) -> Vec<ModelInfo> {
        match &self.models {
            Some(models) => models.clone(),
            None => crate::catalog::models::for_provider(&self.id)
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }

    async fn creds(
        &self,
        client: &reqwest::Client,
        resolver: &dyn crate::credentials::CredentialProvider,
    ) -> anyhow::Result<Creds> {
        match &self.api_key {
            Some(key) => Ok(Creds::ApiKey { key: key.clone() }),
            None => resolver.resolve(client, self.id()).await,
        }
    }

    async fn stream(
        &self,
        ctx: &RoundCtx<'_>,
        history: &[Step],
    ) -> Result<RoundOutcome, RoundError> {
        let base = self.endpoint.as_deref();
        match self.api {
            Api::Responses => responses::stream(base, ctx, history).await,
            Api::Chat => chat::stream(base, ctx, history).await,
            Api::Anthropic => anthropic::stream(base, ctx, history).await,
            Api::Gemini => gemini::stream(base, ctx, history).await,
        }
    }
}

/// The built-in providers — the five tapir ships (over four wire shapes),
/// seeded into the Runtime's registry. Same ids as [`crate::catalog::PROVIDERS`].
pub fn builtin_providers() -> Vec<std::sync::Arc<dyn Provider>> {
    [
        ("copilot", Api::Responses),
        ("openai", Api::Responses),
        ("anthropic", Api::Anthropic),
        ("google", Api::Gemini),
        ("deepseek", Api::Chat),
        ("openrouter", Api::Chat),
    ]
    .into_iter()
    .map(|(id, api)| std::sync::Arc::new(WireProvider::new(id, api)) as _)
    .collect()
}

/// Resolved credentials for a turn.
pub enum Creds {
    /// Copilot bearer token (short-lived; minted per turn).
    Copilot { access: String },
    /// A provider API key (from env or `auth.toml`).
    ApiKey { key: String },
}

/// The environment variable holding a provider's API key.
pub fn env_var(provider: &str) -> Option<&'static str> {
    Some(match provider {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "google" => "GEMINI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        _ => return None,
    })
}

/// Process-wide runtime API keys from `--api-key`, taking precedence over the
/// environment and `auth.toml` and never written to disk.
fn runtime_keys()
-> &'static std::sync::Mutex<std::collections::HashMap<String, String>> {
    static KEYS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, String>>,
    > = std::sync::OnceLock::new();
    KEYS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Set a runtime API key for `provider` (from `--api-key`), in memory only.
pub fn set_runtime_api_key(provider: &str, key: &str) {
    runtime_keys()
        .lock()
        .unwrap()
        .insert(provider.to_string(), key.to_string());
}

/// A provider's API key from an already-loaded credential file: a runtime
/// `--api-key` first, then the environment variable (so a freshly exported key
/// wins over a saved one), then the key saved in the file. `None` if none is
/// set. The file part is parameterized so
/// [`FileCreds`](crate::credentials::FileCreds) can point at an explicit
/// directory.
pub fn api_key_with(auth: &auth::Auth, provider: &str) -> Option<String> {
    if let Some(k) = runtime_keys().lock().unwrap().get(provider)
        && !k.trim().is_empty()
    {
        return Some(k.clone());
    }
    if let Some(var) = env_var(provider)
        && let Ok(v) = std::env::var(var)
        && !v.trim().is_empty()
    {
        return Some(v.trim().to_string());
    }
    auth.providers
        .get(provider)
        .and_then(|p| p.api_key.clone())
        .filter(|k| !k.trim().is_empty())
}

/// The default API base URL for an API-key provider.
pub fn base_url(provider: &str) -> String {
    // Integration-test override: point a provider at a local mock server, e.g.
    // `TAPIR_BASE_URL_DEEPSEEK=http://127.0.0.1:1234`.
    if let Ok(url) = std::env::var(format!(
        "TAPIR_BASE_URL_{}",
        provider.to_ascii_uppercase()
    )) && !url.is_empty()
    {
        return url;
    }
    match provider {
        "openai" => "https://api.openai.com/v1",
        "deepseek" => "https://api.deepseek.com",
        "openrouter" => "https://openrouter.ai/api/v1",
        "anthropic" => "https://api.anthropic.com",
        "google" => "https://generativelanguage.googleapis.com/v1beta",
        _ => "",
    }
    .to_string()
}

/// A plain HTTP client with an idle (not total) read timeout, so long
/// generations aren't cut off but a stalled connection still trips. Used for
/// credential resolution (OAuth token exchange) and as the base for the
/// retrying client below.
pub fn base_client(timeout_secs: u64) -> reqwest::Client {
    reqwest::Client::builder()
        .read_timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .unwrap_or_default()
}

/// Wrap `base` with retry middleware: transient failures (connection errors,
/// 429, 5xx) are retried with exponential backoff and jitter — honoring a
/// `Retry-After` header — up to `retries` times. Replaces the hand-rolled retry
/// loops; non-retryable 4xx and mid-stream errors are surfaced immediately.
pub fn retrying_client(
    base: reqwest::Client,
    retries: u32,
) -> ClientWithMiddleware {
    let policy = ExponentialBackoff::builder().build_with_max_retries(retries);
    ClientBuilder::new(base)
        .with(RetryTransientMiddleware::new_with_policy(policy))
        .build()
}

/// The fixed context of an agent turn — everything a streamed round needs except
/// the growing `history`. Built once per turn and borrowed by each round, so the
/// round functions take two args instead of nine. All fields are shared borrows.
#[derive(Clone, Copy)]
pub struct RoundCtx<'a> {
    pub client: &'a ClientWithMiddleware,
    pub provider: &'a str,
    pub creds: &'a Creds,
    pub model: &'a str,
    pub instructions: &'a str,
    /// The model-facing tool schemas for this turn, resolved by the engine
    /// (`tapir-core`) from the session's active tools. Each shape serializes
    /// these into its own wire format; the provider never reaches into the
    /// tool registry itself.
    pub tools: &'a [ToolDef],
    pub effort: Option<&'a str>,
    pub tx: &'a mpsc::Sender<TurnEvent>,
}

/// A [`RoundRunner`](crate::message::RoundRunner) backed by a registered
/// [`Provider`]. Holds a turn's fixed context (client, credentials, model,
/// instructions, tools) and streams each round through the provider trait — the
/// seam the TUI and headless modes use in production, where tests script rounds
/// instead.
pub struct LiveRounds {
    pub client: ClientWithMiddleware,
    pub provider: std::sync::Arc<dyn Provider>,
    pub creds: Creds,
    pub model: String,
    pub instructions: String,
    pub tools: Vec<ToolDef>,
    pub effort: Option<String>,
}

impl crate::message::RoundRunner for LiveRounds {
    async fn run_round(
        &self,
        history: &[Step],
        tx: &mpsc::Sender<TurnEvent>,
    ) -> Result<RoundOutcome, RoundError> {
        let ctx = RoundCtx {
            client: &self.client,
            provider: self.provider.id(),
            creds: &self.creds,
            model: &self.model,
            instructions: &self.instructions,
            tools: &self.tools,
            effort: self.effort.as_deref(),
            tx,
        };
        self.provider.stream(&ctx, history).await
    }
}

/// Summarize the conversation into a checkpoint (auto-compaction). Streams one
/// round through the [`Provider`] with tools disabled and a throwaway channel;
/// the assistant text is the summary.
pub async fn summarize(
    client: &ClientWithMiddleware,
    provider: &dyn Provider,
    creds: &Creds,
    model: &str,
    instructions: &str,
    history: &[Step],
    effort: Option<&str>,
) -> Result<String, RoundError> {
    // A live receiver, kept in scope, so deltas are silently dropped. No tools
    // during summarization.
    let (tx, _rx) = mpsc::channel::<TurnEvent>(64);
    let ctx = RoundCtx {
        client,
        provider: provider.id(),
        creds,
        model,
        instructions,
        tools: &[],
        effort,
        tx: &tx,
    };
    let outcome = provider.stream(&ctx, history).await?;
    Ok(match outcome.assistant {
        Step::Assistant { text, .. } => text,
        _ => String::new(),
    })
}

// ---- helpers shared by the clients ----------------------------------------

/// Case-insensitive glob match supporting `*` (used for `--models` scope
/// patterns). A pattern with no `*` matches as a substring (a lenient fuzzy
/// fallback); with `*`, the literal parts must appear in order, anchored at
/// the start/end unless that side has a `*`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p = pattern.to_lowercase();
    let t = text.to_lowercase();
    if !p.contains('*') {
        return t.contains(&p);
    }
    let anchored_start = !p.starts_with('*');
    let anchored_end = !p.ends_with('*');
    let parts: Vec<&str> = p.split('*').filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return true; // "*" matches everything
    }
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        let Some(rel) = t[pos..].find(part) else {
            return false;
        };
        let abs = pos + rel;
        if i == 0 && anchored_start && abs != 0 {
            return false;
        }
        pos = abs + part.len();
    }
    if anchored_end && !t.ends_with(parts.last().unwrap()) {
        return false;
    }
    true
}

/// Whether `TAPIR_CACHE_RETENTION=long` requests extended prompt caching (the
/// providers that support it apply a longer cache TTL).
pub(crate) fn long_cache_retention() -> bool {
    std::env::var("TAPIR_CACHE_RETENTION").map(|v| v == "long").unwrap_or(false)
}

/// Map an effort string to a thinking token budget for the shapes that take one
/// (Anthropic, Gemini).
pub(crate) fn budget_for(effort: &str) -> u32 {
    match effort {
        "low" => 2_048,
        "medium" => 4_096,
        "high" => 8_192,
        "xhigh" => 16_384,
        _ => 4_096,
    }
}

/// Whether the active model reasons (from the catalog) — gates reasoning params,
/// which several providers reject on non-reasoning models.
pub(crate) fn model_reasons(provider: &str, model: &str) -> bool {
    crate::catalog::models::get(provider, model)
        .map(|m| m.reasoning)
        .unwrap_or(false)
}

/// An incremental parser folding SSE events into a round's state. Each provider
/// client implements it for its accumulator; the shared [`drive_sse`] loop reads
/// the wire stream and calls [`apply`](SseAccumulator::apply) per event.
pub(crate) trait SseAccumulator {
    /// Fold one parsed SSE `data:` payload (as JSON), returning any text/thinking
    /// deltas the caller should stream to the UI.
    fn apply(&mut self, ev: &Value) -> Vec<TurnEvent>;
}

/// Drive an SSE response through `acc`, streaming each emitted delta to `tx`.
/// Shared by every client — the per-API parsing lives in `acc.apply`. The
/// `eventsource-stream` parser handles SSE framing (multi-line `data:`, CRLF,
/// and UTF-8 split across network chunks); we skip the `[DONE]` sentinel and any
/// non-JSON payload.
pub(crate) async fn drive_sse(
    resp: reqwest::Response,
    acc: &mut impl SseAccumulator,
    tx: &mpsc::Sender<TurnEvent>,
) -> Result<(), RoundError> {
    use eventsource_stream::Eventsource;
    use futures::StreamExt;
    let mut stream = resp.bytes_stream().eventsource();
    while let Some(event) = stream.next().await {
        // A mid-stream error is not retryable — text may already be on screen.
        let event = event.map_err(|e| RoundError { message: e.to_string() })?;
        if event.data == "[DONE]" {
            continue;
        }
        if let Ok(ev) = serde_json::from_str::<Value>(&event.data) {
            for delta in acc.apply(&ev) {
                let _ = tx.send(delta).await;
            }
        }
    }
    Ok(())
}

/// Turn a transport error (after the client exhausted its retries) into a
/// [`RoundError`].
pub(crate) fn send_err(e: reqwest_middleware::Error) -> RoundError {
    RoundError { message: e.to_string() }
}

/// Map a non-success HTTP response to a [`RoundError`], reading the body for
/// context (the client already retried 429 / 5xx before this surfaced).
pub(crate) async fn status_err(resp: reqwest::Response) -> RoundError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    RoundError { message: format!("{status}: {body}") }
}

/// Test doubles shared across the workspace's seam tests (enabled for
/// dependents via the `test-util` feature): a canned Chat-Completions SSE
/// server and body builders.
#[cfg(any(test, feature = "test-util"))]
pub mod testing {
    use std::io::{Read, Write};

    /// A canned SSE body: one streamed text reply, a usage tail, `[DONE]`.
    pub fn sse_text(reply: &str) -> String {
        format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{reply}\"}}}}]}}\n\n\
             data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}],\
             \"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}}}\n\n\
             data: [DONE]\n\n"
        )
    }

    /// A canned SSE body where the model requests one tool call. The arguments
    /// are embedded as an escaped JSON string, e.g. `{\"path\":\"f.txt\"}`
    /// arrives as `r#"{\\"path\\":\\"f.txt\\"}"#`.
    pub fn sse_tool_call(name: &str, args_json_escaped: &str) -> String {
        format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"call-1\",\
             \"function\":{{\"name\":\"{name}\",\"arguments\":\"{args_json_escaped}\"}}}}]}}}}]}}\n\n\
             data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}],\
             \"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}}}\n\n\
             data: [DONE]\n\n"
        )
    }

    /// Serve `responses` in order, one per connection (each model round is one
    /// POST); after the list is exhausted, the last entry repeats. An empty
    /// list = a stalling server: accept and hold connections, never answer.
    /// Returns the bound port. (Prior art: `tests/tui_pty.rs` drives the whole
    /// TUI against the same canned shape.)
    pub fn spawn_mock_sse(responses: Vec<String>) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let mut served = 0usize;
            let mut held = Vec::new();
            for mut stream in listener.incoming().flatten() {
                // Drain the request so the client isn't reset mid-send.
                let mut buf = [0u8; 4096];
                let mut req = Vec::new();
                loop {
                    match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            req.extend_from_slice(&buf[..n]);
                            if let Some(p) =
                                req.windows(4).position(|w| w == b"\r\n\r\n")
                            {
                                let head = String::from_utf8_lossy(&req[..p])
                                    .to_lowercase();
                                let clen: usize = head
                                    .lines()
                                    .find_map(|l| {
                                        l.strip_prefix("content-length:")
                                    })
                                    .and_then(|v| v.trim().parse().ok())
                                    .unwrap_or(0);
                                if req.len() >= p + 4 + clen {
                                    break;
                                }
                            }
                        }
                    }
                }
                if responses.is_empty() {
                    held.push(stream);
                    continue;
                }
                let sse = &responses[served.min(responses.len() - 1)];
                served += 1;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                    sse.len(),
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        port
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The two production-shaped seam tests that drive a `Runtime` session
    // through `LiveRounds` live in `tapir-core` (tests/live_rounds.rs) now —
    // the `Runtime` and `Session` they need are the agent engine's, which sits
    // a layer above this crate.

    #[test]
    fn builtin_providers_cover_each_wire_shape() {
        // The five tapir providers collapse onto four wire shapes — the mapping
        // the old name dispatch encoded, now construction.
        let by_id = |id: &str| {
            builtin_providers()
                .into_iter()
                .find(|p| p.id() == id)
                .unwrap_or_else(|| panic!("{id} is a built-in"))
        };
        for id in [
            "copilot",
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "openrouter",
        ] {
            assert_eq!(by_id(id).id(), id);
        }
    }

    #[test]
    fn api_key_prefers_runtime_then_env_then_file() {
        // An empty credential file isolates the runtime-key / env precedence.
        let auth = crate::auth::Auth::default();
        // SAFETY: single-threaded test; we set and clear our own scratch var.
        unsafe { std::env::set_var("OPENAI_API_KEY", "sk-env") };
        assert_eq!(api_key_with(&auth, "openai").as_deref(), Some("sk-env"));
        // A runtime key (from --api-key) wins over the environment.
        set_runtime_api_key("openai", "sk-runtime");
        assert_eq!(
            api_key_with(&auth, "openai").as_deref(),
            Some("sk-runtime")
        );
        runtime_keys().lock().unwrap().remove("openai");
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
    }

    #[test]
    fn glob_matches_anchors_and_wildcards() {
        assert!(glob_match("claude-opus*", "claude-opus-4-8"));
        assert!(glob_match("*sonnet*", "claude-sonnet-4"));
        assert!(glob_match("gpt-5*", "gpt-5.4"));
        assert!(glob_match("anthropic/claude*", "anthropic/claude-opus-4"));
        assert!(!glob_match("gpt-5*", "claude-opus")); // anchored start fails
        assert!(!glob_match("*haiku", "claude-haiku-4.5")); // anchored end fails
        assert!(glob_match("opus", "claude-opus-4-8")); // no `*` → substring
    }
}
