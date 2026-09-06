// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Gate tool calls before they run. `before_tool_call` is awaited once per call
//! in model order, before the batch executes, and returns a [`ToolDecision`]:
//! `Proceed` runs it, `Modify` rewrites the arguments (re-validated), and `Deny`
//! skips it, feeding the model a synthetic error. `after_tool_call` observes
//! each executed result afterwards — it cannot change anything.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run -p tapir --example approval_gate --features anthropic
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
struct ShellArgs {
    /// The shell command to run.
    command: String,
}

// A deliberately dangerous capability the gate will police. The argument is a
// struct so the tool's JSON schema is an `object`, as providers require.
#[cfg(feature = "anthropic")]
#[tool]
/// Run a shell command and return its (pretend) output.
async fn run_shell(args: ShellArgs) -> String {
    format!("ran: {}", args.command)
}

#[cfg(feature = "anthropic")]
#[tokio::main]
async fn main() -> std::result::Result<(), std::sync::Arc<tapir::Error>> {
    let agent = Agent::builder()
        .model("claude-haiku-4-5")
        .system("You can run shell commands with the run_shell tool.")
        .tool(run_shell)
        // Deny anything that looks destructive; let the rest through.
        .before_tool_call(|call| {
            Box::pin(async move {
                let cmd = call
                    .arguments()
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if cmd.contains("rm -rf") || cmd.contains("mkfs") {
                    ToolDecision::Deny {
                        message: "destructive commands are not allowed".into(),
                    }
                } else {
                    ToolDecision::Proceed
                }
            })
        })
        // Observe-only audit log of what actually executed.
        .after_tool_call(|call, result| {
            Box::pin(async move {
                eprintln!(
                    "[audit] {} finished (is_error={})",
                    call.name(),
                    result.is_error,
                );
            })
        })
        .build()?;

    let reply = agent
        .prompt("Delete everything under /tmp with rm -rf, then list the current directory.")
        .await?;
    println!("{}", reply.text_content());
    Ok(())
}

#[cfg(not(feature = "anthropic"))]
fn main() {
    eprintln!(
        "build with `--features anthropic` to run the approval_gate example"
    );
}
