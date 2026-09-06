// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Register several typed tools and let the model pick. A tool is an `async fn`
//! tagged with `#[tool]`; its first parameter is the typed, JSON-Schema'd
//! argument (a bare scalar or a `#[derive(Deserialize, JsonSchema)]` struct),
//! and an optional `ctx: &ToolCtx` makes it contextual. `#[tool(read_only)]`
//! marks a tool parallel-safe (`Safe`); the default is `Exclusive`, serialized
//! against other tools in the batch.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example tools --features anthropic
//! ```

#[cfg(feature = "anthropic")]
use schemars::JsonSchema;
#[cfg(feature = "anthropic")]
use serde::Deserialize;
#[cfg(feature = "anthropic")]
use tapir::prelude::*;

#[cfg(feature = "anthropic")]
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct TemperatureArgs {
    /// The city to report the temperature for.
    city: String,
}

#[cfg(feature = "anthropic")]
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ConvertArgs {
    /// The amount to convert.
    amount: f64,
    /// The source currency code, e.g. "USD".
    from: String,
    /// The target currency code, e.g. "EUR".
    to: String,
}

// Read-only: no side effects, so safe to run alongside other reads. The
// argument is a struct so the tool's JSON schema is an `object`, which the
// providers require for a tool's input schema.
#[cfg(feature = "anthropic")]
#[tool(read_only)]
/// Get the current temperature for a city, in Celsius.
async fn temperature(args: TemperatureArgs) -> String {
    format!("It is 21C and clear in {}.", args.city)
}

// Read-only, with a struct argument and a `ctx` handle for the call id.
#[cfg(feature = "anthropic")]
#[tool(read_only)]
/// Convert an amount between two currencies at a fixed demo rate.
async fn convert_currency(args: ConvertArgs, ctx: &ToolCtx) -> String {
    let rate = 0.92; // a stand-in fixed rate for the example
    format!(
        "[{}] {:.2} {} = {:.2} {}",
        ctx.call_id(),
        args.amount,
        args.from,
        args.amount * rate,
        args.to,
    )
}

#[cfg(feature = "anthropic")]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5")
        .system("You are a concise assistant. Use tools when they help.")
        .tool(temperature)
        .tool(convert_currency)
        .build()?;

    let reply = agent
        .prompt("What's the weather in Lisbon, and what is 100 USD in EUR?")
        .await?;
    println!("{}", reply.text_content());
    Ok(())
}

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!("build with `--features anthropic` to run the tools example");
}
