//! hello-agent — the smallest useful tapir program.
//!
//! Builds a [`Runtime`], inspects the built-in providers and a session's tools
//! (offline), then — if a provider API key is present — runs one streamed turn
//! and prints the reply.
//!
//! Run it:
//!
//! ```text
//! cargo run --bin hello-agent                       # offline: prints setup
//! ANTHROPIC_API_KEY=sk-… cargo run --bin hello-agent "Say hi"
//! TAPIR_PROVIDER=openai OPENAI_API_KEY=sk-… cargo run --bin hello-agent
//! TAPIR_MODEL=claude-sonnet-4-6 ANTHROPIC_API_KEY=sk-… cargo run --bin hello-agent
//! ```
//!
//! The model defaults to the catalog's curated flagship for the provider
//! (`tapir::catalog::default_model`); set `$TAPIR_MODEL` to override it.

use std::io::Write;

// The common front door: Runtime, Input, ModelRef, TurnEvent, …
use tapir::prelude::*;
// `env_var` is a provider helper, not part of the prelude — reach it by path.
use tapir::providers::env_var;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A runtime seeds the built-in providers, tools, and credential resolver.
    let rt = Runtime::builder().build();

    println!("Built-in providers:");
    for p in rt.providers() {
        println!("  - {} ({} models)", p.id(), p.models().len());
    }

    // A session is one conversation, rooted at a working directory.
    let mut session = rt.session(std::env::current_dir()?);
    let tools: Vec<&str> =
        session.tool_definitions().iter().map(|d| d.name).collect();
    println!("\nSession tools: {}", tools.join(", "));

    // Pick a provider (default anthropic; override with $TAPIR_PROVIDER) and a
    // model: $TAPIR_MODEL if set, else the catalog's curated default for the
    // provider (a current flagship — unlike the raw first-listed model, which
    // can be one your account no longer serves).
    let provider_id =
        std::env::var("TAPIR_PROVIDER").unwrap_or_else(|_| "anthropic".into());
    if rt.find_provider(&provider_id).is_none() {
        anyhow::bail!("unknown provider {provider_id:?}");
    }
    let model = match std::env::var("TAPIR_MODEL") {
        Ok(m) => m,
        Err(_) => tapir::catalog::default_model(&provider_id)
            .map(|m| m.id.to_string())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no catalog default model for provider {provider_id:?}"
                )
            })?,
    };
    println!("\nModel: {provider_id}/{model}");

    // No key? Show how to run a real turn and stop — the setup above already
    // exercised the public API with no network.
    let has_key = env_var(&provider_id)
        .map(|v| std::env::var(v).is_ok())
        .unwrap_or(false);
    if !has_key {
        let var = env_var(&provider_id).unwrap_or("<provider key>");
        println!("\nSet {var} to run a turn, e.g.:");
        println!("  {var}=… cargo run --bin hello-agent \"Say hi\"");
        return Ok(());
    }

    // Run one turn against the selected model and stream the reply.
    session.set_model(Some(ModelRef { provider: provider_id, id: model }));
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in one short sentence.".into());
    println!("\n> {prompt}\n");
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
