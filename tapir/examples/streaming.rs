// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Render a reply token-by-token off the flat [`AgentEvent`] stream instead of
//! awaiting the settled message. A `Run` is both a future (await the reply) and
//! a `Stream` of events; here we drive it as a stream and print each provider
//! text delta as it arrives.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example streaming --features anthropic
//! ```

#[cfg(feature = "anthropic")]
use std::io::Write;

#[cfg(feature = "anthropic")]
use futures_util::StreamExt;
#[cfg(feature = "anthropic")]
use tapir::prelude::*;
#[cfg(feature = "anthropic")]
use tapir_provider::StreamEvent;

#[cfg(feature = "anthropic")]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5")
        .system("You are helpful. Keep answers to a couple of sentences.")
        .build()?;

    // `converse`/`prompt` return a `Run`. Consumed as a `Stream`, it yields the
    // whole event vocabulary; a `MessageUpdate` carries the provider delta
    // verbatim, so a token-by-token UI renders those and ignores the rest.
    let mut run = agent.prompt("Explain what a tapir is, briefly.");
    while let Some(event) = run.next().await {
        match event {
            AgentEvent::MessageUpdate {
                delta: StreamEvent::TextDelta { text, .. },
                ..
            } => {
                print!("{text}");
                let _ = std::io::stdout().flush();
            }
            AgentEvent::AgentEnd { .. } => {
                println!();
                break;
            }
            AgentEvent::Error { error, .. } => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!("build with `--features anthropic` to run the streaming example");
}
