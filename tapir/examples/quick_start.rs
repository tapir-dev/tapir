// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The <15-line north-star: name a model by id and get a working tool-using
//! agent, no manual provider construction. Build and run with the `anthropic`
//! feature and an `ANTHROPIC_API_KEY` in the environment:
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example quick_start --features anthropic
//! ```

#[cfg(feature = "anthropic")]
use tapir::prelude::*;

#[cfg(feature = "anthropic")]
#[tool]
/// Get the weather for a city.
async fn weather(city: String) -> String {
    format!("sunny in {city}")
}

// The run future yields `Result<_, Arc<tapir::Error>>` and there is no
// `From<Arc<Error>>` shim (the type stays honest), so `main` returns the std
// `Result` over the `Arc` error — spelled out because the prelude glob brings
// tapir's own single-parameter `Result` alias into scope.
#[cfg(feature = "anthropic")]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5") // ANTHROPIC_API_KEY + reqwest default
        .system("You are helpful.")
        .tool(weather)
        .build()?;

    let reply = agent.prompt("Weather in Paris?").await?;
    println!("{}", reply.text_content());
    Ok(())
}

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!("build with `--features anthropic` to run the quick-start");
}
