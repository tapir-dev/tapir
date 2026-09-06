// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Persist a conversation and reopen it. A [`SessionStore`] makes the agent
//! write history through as a run progresses; [`Agent::resume`] rebuilds an
//! agent from a builder and seeds its history from the store, so a later prompt
//! continues where the last session left off. This uses the reference
//! `FileSessionStore` (JSONL on disk), behind the `store-file` feature.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example persistence \
//!     --features "anthropic store-file"
//! ```

#[cfg(all(feature = "anthropic", feature = "store-file"))]
use std::sync::Arc;

#[cfg(all(feature = "anthropic", feature = "store-file"))]
use tapir::message::NoCustom;
#[cfg(all(feature = "anthropic", feature = "store-file"))]
use tapir::prelude::*;
#[cfg(all(feature = "anthropic", feature = "store-file"))]
use tapir::store::file::FileSessionStore;

#[cfg(all(feature = "anthropic", feature = "store-file"))]
fn builder() -> tapir::agent::AgentBuilder {
    Agent::builder()
        .model("claude-haiku-4-5")
        .system("You are a concise assistant with a good memory.")
}

#[cfg(all(feature = "anthropic", feature = "store-file"))]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let path = std::env::temp_dir().join("tapir-persistence-example.jsonl");
    let _ = std::fs::remove_file(&path); // start clean for the demo

    // First session: attach the store, then prompt. History is written through.
    {
        let store: Arc<dyn SessionStore<NoCustom>> = Arc::new(
            FileSessionStore::open(&path)
                .await
                .map_err(tapir::Error::from)?,
        );
        let agent = builder().store(store).build()?;
        let reply = agent
            .prompt("Remember that my favorite animal is the tapir.")
            .await?;
        println!("session 1: {}", reply.text_content());
    }

    // Second session: reopen the same file and continue. `resume` seeds history
    // from the store, so the model still knows the earlier turn.
    {
        let store: Arc<dyn SessionStore<NoCustom>> = Arc::new(
            FileSessionStore::open(&path)
                .await
                .map_err(tapir::Error::from)?,
        );
        let agent = Agent::resume(builder(), store).await?;
        let reply = agent.prompt("What is my favorite animal?").await?;
        println!("session 2: {}", reply.text_content());
    }

    Ok(())
}

#[cfg(not(all(feature = "anthropic", feature = "store-file")))]
fn main() {
    eprintln!(
        "build with `--features \"anthropic store-file\"` to run the persistence example"
    );
}
