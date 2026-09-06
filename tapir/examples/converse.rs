// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! An interactive follow-up run. `converse` parks at a tool-free reply instead
//! of ending: it emits [`AgentEvent::Idle`] and waits. A `RunHandle::steer`
//! wakes it with another user turn (resetting the tool-iteration cap);
//! `RunHandle::finish` ends it, resolving the run with the final reply.
//!
//! This example scripts the follow-ups so it runs unattended; a real chat UI
//! would read the next line from the user each time the run goes idle.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example converse --features anthropic
//! ```

#[cfg(feature = "anthropic")]
use futures_util::StreamExt;
#[cfg(feature = "anthropic")]
use tapir::prelude::*;

#[cfg(feature = "anthropic")]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5")
        .system(
            "You are a terse travel assistant. One or two sentences per reply.",
        )
        .build()?;

    // The queued follow-ups, injected one per idle park.
    let mut follow_ups = [
        "Now suggest one dish to try there.",
        "And the best month to visit?",
    ]
    .into_iter();

    let run = agent.converse("Suggest a single city to visit in Japan.");
    // The handle outlives the `Run` and steers/finishes it from here.
    let handle = run.handle();

    let mut run = run;
    while let Some(event) = run.next().await {
        match event {
            AgentEvent::MessageEnd { message, .. } => {
                println!("assistant: {}", message.text_content());
            }
            AgentEvent::Idle { .. } => match follow_ups.next() {
                Some(text) => {
                    println!("\nyou: {text}");
                    handle.steer(text);
                }
                None => handle.finish(),
            },
            AgentEvent::AgentEnd { .. } => break,
            AgentEvent::Error { error, .. } => return Err(error),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!("build with `--features anthropic` to run the converse example");
}
