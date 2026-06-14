//! custom-tool — implement the [`Tool`] trait, register it on the [`Runtime`],
//! and watch the model call it. The tool here is a `multiply` calculator.
//!
//! A registered tool is active in a session by default (alongside the
//! built-ins). The engine advertises its schema to the model, dispatches the
//! call when the model asks for it, runs [`Tool::run`], and feeds the result
//! back — emitting `ToolStart`/`ToolEnd` so a frontend can show it.
//!
//! ```text
//! cargo run --bin custom-tool                          # offline: lists tools
//! ANTHROPIC_API_KEY=sk-… cargo run --bin custom-tool "What is 23 times 19?"
//! ```

use std::io::Write;
use std::sync::Arc;

use serde_json::{Value, json};
use tapir::prelude::*;
use tapir::providers::env_var;
use tapir::tools::{ToolCtx, ToolResult};

/// A tool that multiplies two integers.
struct Multiply;

#[async_trait::async_trait]
impl Tool for Multiply {
    fn name(&self) -> &'static str {
        "multiply"
    }

    fn description(&self) -> &'static str {
        "Multiply two integers and return their product."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "the first factor" },
                "b": { "type": "integer", "description": "the second factor" }
            },
            "required": ["a", "b"]
        })
    }

    fn title(&self, args: &Value) -> String {
        let a = args.get("a").and_then(Value::as_i64).unwrap_or_default();
        let b = args.get("b").and_then(Value::as_i64).unwrap_or_default();
        format!("multiply({a}, {b})")
    }

    async fn run(
        &self,
        args: &Value,
        _ctx: &ToolCtx,
    ) -> anyhow::Result<ToolResult> {
        let a = args
            .get("a")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow::anyhow!("missing integer 'a'"))?;
        let b = args
            .get("b")
            .and_then(Value::as_i64)
            .ok_or_else(|| anyhow::anyhow!("missing integer 'b'"))?;
        Ok(ToolResult::text((a * b).to_string()))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Register the tool; it joins the built-ins, active by default.
    let rt = Runtime::builder().tool(Arc::new(Multiply)).build();
    let mut session = rt.session(std::env::current_dir()?);

    let tools: Vec<&str> =
        session.tool_definitions().iter().map(|d| d.name).collect();
    println!("Session tools: {}", tools.join(", "));

    let provider = "anthropic";
    let has_key =
        env_var(provider).map(|v| std::env::var(v).is_ok()).unwrap_or(false);
    if !has_key {
        println!(
            "\nSet ANTHROPIC_API_KEY to watch the model call the `multiply` tool."
        );
        return Ok(());
    }

    let model = tapir::catalog::default_model(provider)
        .map(|m| m.id.to_string())
        .ok_or_else(|| anyhow::anyhow!("no default model for {provider}"))?;
    session.set_model(Some(ModelRef { provider: provider.into(), id: model }));

    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "What is 23 multiplied by 19? Use the multiply tool.".into()
    });
    println!("\n> {prompt}\n");

    let mut rx = session.run(Input::text(prompt));
    let mut out = std::io::stdout();
    while let Some(ev) = rx.recv().await {
        match ev {
            TurnEvent::Text { delta } => {
                print!("{delta}");
                out.flush().ok();
            }
            TurnEvent::ToolStart { name, title, .. } => {
                println!("\n[tool ▸ {name}] {title}");
            }
            TurnEvent::ToolEnd { output, is_error, .. } => {
                let tag = if is_error { "error" } else { "result" };
                println!("[tool ◂ {tag}] {output}\n");
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
