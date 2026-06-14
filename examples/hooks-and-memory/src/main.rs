//! hooks-and-memory — the [`AgentBuilder`] with a per-session hook, plus session
//! persistence (memory across sessions). Fully offline and deterministic: a
//! scripted provider drives one tool call then a reply, so the hook fires and
//! there's a conversation to persist and resume.
//!
//! What it shows:
//! - `rt.agent().hook(..).build()` — fluent session setup with a hook.
//! - a [`Hook`] firing around the tool call (before/after).
//! - a [`SessionStore`] — append the turn's entries, then `Runtime::resume` a
//!   fresh agent that replays the stored conversation ("memory").
//!
//! ```text
//! cargo run --bin hooks-and-memory
//! ```
//!
//! [`AgentBuilder`]: tapir::runtime::AgentBuilder

use std::sync::Arc;

use serde_json::{Value, json};
use tapir::hook::{HookCtx, ToolDecision};
use tapir::prelude::*;
use tapir::providers::{Creds, ModelInfo, RoundCtx, RoundOutcome, Step};
use tapir::runtime::SessionOptions;
use tapir::store::{Entry, InMemoryStore};
use tapir::tools::{ToolCtx, ToolResult};

/// A two-round scripted provider (no network): round 1 calls the `note` tool,
/// round 2 — once the tool result is in the history — gives a final reply.
struct Scripted;

#[async_trait::async_trait]
impl Provider for Scripted {
    fn id(&self) -> &str {
        "scripted"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "scripted-1".into(),
            name: "Scripted".into(),
            ..Default::default()
        }]
    }

    async fn creds(
        &self,
        _client: &reqwest::Client,
        _resolver: &dyn CredentialProvider,
    ) -> anyhow::Result<Creds> {
        Ok(Creds::ApiKey { key: String::new() })
    }

    async fn stream(
        &self,
        ctx: &RoundCtx<'_>,
        history: &[Step],
    ) -> Result<RoundOutcome, RoundError> {
        let tool_ran =
            history.iter().any(|s| matches!(s, Step::ToolResult { .. }));
        if !tool_ran {
            // Round 1: request the note tool (no assistant text yet).
            let call = ToolCall {
                call_id: "call-1".into(),
                name: "note".into(),
                args: json!({ "text": "the build is green" }),
            };
            Ok(RoundOutcome {
                usage: Usage::default(),
                tool_calls: vec![call.clone()],
                assistant: Step::Assistant {
                    text: String::new(),
                    thinking: String::new(),
                    tool_calls: vec![call],
                    raw: None,
                },
            })
        } else {
            // Round 2: the tool result is in context — reply and finish.
            let reply = "Done — your note is saved.";
            for chunk in reply.split_inclusive(' ') {
                let _ = ctx
                    .tx
                    .send(TurnEvent::Text { delta: chunk.to_string() })
                    .await;
            }
            Ok(RoundOutcome {
                usage: Usage::default(),
                tool_calls: Vec::new(),
                assistant: Step::Assistant {
                    text: reply.to_string(),
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    raw: None,
                },
            })
        }
    }
}

/// A trivial tool the scripted model calls.
struct Note;

#[async_trait::async_trait]
impl Tool for Note {
    fn name(&self) -> &'static str {
        "note"
    }
    fn description(&self) -> &'static str {
        "Save a short note."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        })
    }
    async fn run(
        &self,
        args: &Value,
        _ctx: &ToolCtx,
    ) -> anyhow::Result<ToolResult> {
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");
        Ok(ToolResult::text(format!("saved: {text}")))
    }
}

/// A hook that logs (and could deny/modify) every tool call — the policy seam.
struct LogHook;

#[async_trait::async_trait]
impl Hook for LogHook {
    async fn before_tool(
        &self,
        call: &ToolCall,
        _ctx: &HookCtx<'_>,
    ) -> ToolDecision {
        println!("  [hook] before {}({})", call.name, call.args);
        ToolDecision::Allow
    }

    async fn after_tool(
        &self,
        call: &ToolCall,
        result: &ToolResult,
        _ctx: &HookCtx<'_>,
    ) -> Option<ToolResult> {
        println!("  [hook] after  {} -> {}", call.name, result.model_text);
        None // leave the result unchanged
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A runtime with the scripted provider, the note tool, and in-memory
    // persistence (swap in FileStore for durable, cross-process memory).
    let rt = Runtime::builder()
        .provider(Arc::new(Scripted))
        .tool(Arc::new(Note))
        .store(Arc::new(InMemoryStore::default()))
        .build();

    // A session built fluently, with a per-session logging hook on top of the
    // runtime's (none here).
    let mut session = rt
        .agent()
        .model("scripted", "scripted-1")
        .hook(Arc::new(LogHook))
        .build();

    let id = "demo-session";
    let store = rt.store();
    let prompt = "Take a note for me.";

    println!("== turn 1 (the hook fires around the tool call) ==");
    println!("> {prompt}\n");
    store
        .append(id, &Entry::Message { role: Role::User, text: prompt.into() })
        .await?;

    // Drive the turn and persist its entries as they stream — the same mapping
    // a real frontend does from TurnEvents to store Entries.
    let mut rx = session.run(Input::text(prompt));
    let mut reply = String::new();
    let mut pending_tool: Option<(String, String)> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            TurnEvent::Text { delta } => {
                print!("{delta}");
                reply.push_str(&delta);
            }
            TurnEvent::ToolStart { name, title, .. } => {
                pending_tool = Some((name, title))
            }
            TurnEvent::ToolEnd { output, is_error, took_ms, .. } => {
                if let Some((name, title)) = pending_tool.take() {
                    let entry =
                        Entry::Tool { name, title, output, is_error, took_ms };
                    store.append(id, &entry).await?;
                }
            }
            TurnEvent::Done => break,
            TurnEvent::Error { message } => {
                eprintln!("\nerror: {message}");
                break;
            }
            _ => {}
        }
    }
    store
        .append(id, &Entry::Message { role: Role::Assistant, text: reply })
        .await?;
    println!("\n");

    // Memory: a brand-new agent, resumed by id, replays the stored conversation.
    println!("== resume: a fresh agent replays the stored conversation ==");
    let resumed = rt.resume(SessionOptions::new(".".into()), id).await?;
    for step in resumed.history() {
        match step {
            Step::User { text, .. } => println!("  user:      {text}"),
            Step::Assistant { text, .. } if !text.is_empty() => {
                println!("  assistant: {text}")
            }
            _ => {}
        }
    }
    println!(
        "\n{} entries persisted under {id:?}.",
        store.load(id).await?.len()
    );
    Ok(())
}
