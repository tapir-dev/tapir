//! custom-provider — implement the [`Provider`] trait to plug in your own LLM
//! backend. This one is a self-contained *echo* provider: it needs no network
//! and no API key, so the example runs deterministically anywhere.
//!
//! A real provider serializes the neutral [`Step`] history into its wire
//! request, streams the response back as `TurnEvent::Text`/`Thinking` deltas on
//! `ctx.tx`, and returns the round's [`RoundOutcome`]. The engine owns the rest
//! of the turn (the loop, tools, lifecycle events).
//!
//! ```text
//! cargo run --bin custom-provider "anything you like"
//! ```

use std::io::Write;
use std::sync::Arc;

use tapir::prelude::*;
use tapir::providers::{
    Creds, ModelInfo, Provider, RoundCtx, RoundOutcome, Step,
};

/// A provider that replies with the user's own words, prefixed — no network.
struct EchoProvider;

#[async_trait::async_trait]
impl Provider for EchoProvider {
    fn id(&self) -> &str {
        "echo"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: "echo-1".into(),
            name: "Echo".into(),
            ..Default::default()
        }]
    }

    // Offline: be its own credential authority instead of consulting the
    // Runtime's resolver (which has no key for "echo").
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
        // Echo the most recent user message back, streamed word by word.
        let last_user = history
            .iter()
            .rev()
            .find_map(|s| match s {
                Step::User { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let reply = format!("echo: {last_user}");
        for chunk in reply.split_inclusive(' ') {
            let _ =
                ctx.tx.send(TurnEvent::Text { delta: chunk.to_string() }).await;
        }
        Ok(RoundOutcome {
            usage: Usage::default(),
            tool_calls: Vec::new(),
            assistant: Step::Assistant {
                text: reply,
                thinking: String::new(),
                tool_calls: Vec::new(),
                raw: None,
            },
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Register the provider alongside the built-ins and point a session at it.
    let rt = Runtime::builder().provider(Arc::new(EchoProvider)).build();
    let mut session = rt.session(std::env::current_dir()?);
    session.set_model(Some(ModelRef {
        provider: "echo".into(),
        id: "echo-1".into(),
    }));

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello, custom provider!".into());
    println!("> {prompt}\n");

    let mut rx = session.run(Input::text(prompt));
    let mut out = std::io::stdout();
    while let Some(ev) = rx.recv().await {
        match ev {
            TurnEvent::Text { delta } => {
                print!("{delta}");
                out.flush().ok();
            }
            TurnEvent::Done => break,
            TurnEvent::Error { message } => {
                eprintln!("\nerror: {message}");
                break;
            }
            _ => {}
        }
    }
    println!();
    Ok(())
}
