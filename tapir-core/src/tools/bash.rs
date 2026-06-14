//! `bash` — run a shell command, capturing stdout+stderr (tail-truncated).
//!
//! All tool logic lives here — schema, destructive-command heuristic,
//! truncation, timeout semantics — while the raw process execution goes
//! through the execution context's [`ExecOps`](super::exec::ExecOps) (the
//! local host by default; a sandbox backend when injected). Output is streamed
//! through the context's update sink as the command runs (the turn loop
//! forwards each chunk as a `ToolUpdate`), while the full output comes back
//! whole for the authoritative result at the end.

use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_opt_u64, arg_str, truncate};

/// The shell tool behind the [`Tool`] trait — the streaming one: it pumps its
/// output through [`ToolCtx::update`](super::tool::ToolCtx::update).
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        NAME
    }
    fn description(&self) -> &'static str {
        DESCRIPTION
    }
    fn parameters(&self) -> Value {
        schema()
    }
    fn title(&self, args: &Value) -> String {
        title(args)
    }
    async fn run(&self, args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
        run(args, ctx).await
    }
}

pub const NAME: &str = "bash";
pub const DESCRIPTION: &str = "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to the last 2000 lines or 50KB (whichever is hit first). Optionally provide a timeout in seconds.";
pub const SNIPPET: &str = "Execute bash commands (ls, grep, find, etc.)";
pub const GUIDELINES: &[&str] = &[];

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Bash command to execute" },
            "timeout": { "type": "number", "description": "Timeout in seconds (optional, no default timeout)" }
        },
        "required": ["command"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
    match arg_opt_u64(args, "timeout") {
        Some(secs) => format!("$ {command} (timeout {secs}s)"),
        None => format!("$ {command}"),
    }
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let command = arg_str(args, "command")?;
    let timeout = arg_opt_u64(args, "timeout");

    // Refuse obviously destructive commands the model might run by mistake. This
    // is a best-effort heuristic, not a sandbox; the error goes back to the model
    // so it can rethink. (User-typed `!`/`!!` commands are not gated.)
    if let Some(reason) = destructive_reason(&command) {
        return Err(anyhow!(
            "refused: this command looks destructive ({reason}); if it's really intended, ask the user to run it themselves"
        ));
    }

    // All process execution goes through the context's exec operations (the
    // local host by default; a sandbox backend when injected). The ops stream
    // output chunks into the channel as the command runs; forward each one
    // through `ctx.update` so the transcript shows output live.
    // The cwd crosses the seam boundary-validated: the guest cwd translated
    // to its host directory when a boundary is configured, the plain cwd
    // otherwise. (A backend executing inside a container translates back.)
    let cwd = ctx.host_cwd()?;
    let (chunks_tx, mut chunks_rx) = tokio::sync::mpsc::channel::<String>(64);
    let exec = ctx.exec_ops().exec(
        &command,
        &cwd,
        timeout.map(Duration::from_secs),
        chunks_tx,
    );
    let forward = async {
        while let Some(chunk) = chunks_rx.recv().await {
            ctx.update(chunk).await;
        }
    };
    let (outcome, ()) = tokio::join!(exec, forward);
    let outcome = outcome?;
    if outcome.timed_out {
        let secs = timeout.unwrap_or_default();
        return Err(anyhow!("command timed out after {secs}s"));
    }

    let trunc = truncate::tail(&outcome.output);
    let mut out = trunc.content;
    if trunc.truncated {
        out = format!(
            "[Truncated: showing last {} of {} lines]\n{out}",
            trunc.output_lines, trunc.total_lines
        );
    }
    if let Some(code) = outcome.exit_code
        && code != 0
    {
        out.push_str(&format!("\n[exit {code}]"));
    }
    Ok(ToolResult::text(out))
}

/// A best-effort check for clearly destructive commands. Returns the reason to
/// refuse, or `None`. Deliberately narrow (high signal, low false-positive): it
/// is a guardrail against accidents, not a security boundary.
pub fn destructive_reason(command: &str) -> Option<String> {
    let cmd = command.trim();
    let lower = cmd.to_lowercase();

    // Fork bomb: `:(){ :|:& };:` (ignoring whitespace).
    let compact: String = cmd.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.contains(":(){:|:&};:") {
        return Some("looks like a fork bomb".into());
    }
    // Format a filesystem.
    if has_word(&lower, "mkfs") {
        return Some("formats a filesystem (mkfs)".into());
    }
    // Write straight to a device.
    if lower.contains("dd ") && lower.contains("of=/dev/") {
        return Some("writes directly to a device (dd of=/dev/…)".into());
    }
    for dev in [
        ">/dev/sd",
        "> /dev/sd",
        ">/dev/nvme",
        "> /dev/nvme",
        ">/dev/disk",
        "> /dev/disk",
    ] {
        if lower.contains(dev) {
            return Some("redirects output onto a block device".into());
        }
    }
    // Recursive force-delete (or recursive chmod/chown) of a root/home/cwd-wide
    // target, across each `;`/`&&`/`||`/`|`/newline segment.
    for seg in cmd.split(['\n', ';', '&', '|']) {
        let tokens: Vec<&str> = seg.split_whitespace().collect();
        let mut i = 0;
        while matches!(tokens.get(i), Some(&("sudo" | "doas" | "env"))) {
            i += 1;
        }
        let Some(&program) = tokens.get(i) else { continue };
        let rest = &tokens[i + 1..];
        let is_delete = program == "rm";
        let is_perm = program == "chmod" || program == "chown";
        if !is_delete && !is_perm {
            continue;
        }
        if !is_recursive(rest) {
            continue;
        }
        // `rm` also needs force to be unattended-dangerous; perms always are.
        if is_delete && !is_force(rest) {
            continue;
        }
        for target in rest.iter().filter(|t| !t.starts_with('-')) {
            if is_root_wide_target(target) {
                let verb = if is_delete {
                    "force-delete"
                } else {
                    "permission change"
                };
                let clean = target.trim_matches(['"', '\'']);
                return Some(format!("recursive {verb} of {clean}"));
            }
        }
    }
    None
}

/// A bare word (e.g. `mkfs`) present as a token, not as a substring of another.
fn has_word(haystack: &str, word: &str) -> bool {
    haystack.split(|c: char| !c.is_alphanumeric()).any(|w| w == word)
}

fn is_recursive(flags: &[&str]) -> bool {
    flags.iter().any(|t| {
        *t == "--recursive"
            || single_flag_has(t, 'r')
            || single_flag_has(t, 'R')
    })
}

fn is_force(flags: &[&str]) -> bool {
    flags.iter().any(|t| *t == "--force" || single_flag_has(t, 'f'))
}

/// Whether `t` is a single-dash flag bundle (e.g. `-rf`) containing `c`.
fn single_flag_has(t: &str, c: char) -> bool {
    t.starts_with('-') && !t.starts_with("--") && t.contains(c)
}

fn is_root_wide_target(target: &str) -> bool {
    let t = target.trim_matches(['"', '\'']);
    matches!(
        t,
        "/" | "/*"
            | "~"
            | "~/"
            | "~/*"
            | "$HOME"
            | "$HOME/"
            | "$HOME/*"
            | "${HOME}"
            | "."
            | "./"
            | "./*"
            | "*"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn bash_tool_executes_through_the_exec_ops_on_its_context() {
        use crate::tools::exec::testing::FakeExecOps;
        use std::path::PathBuf;

        // The seam: the tool performs no process execution itself — the exec
        // operations on its context receive the command, cwd and timeout, and
        // their output is the tool's result.
        let fake = std::sync::Arc::new(FakeExecOps::ok("fake output\n", 0));
        let ctx =
            ToolCtx::new("/somewhere".into(), CancelToken::default(), None)
                .with_exec_ops(fake.clone());
        let out = BashTool
            .run(&json!({ "command": "echo hi", "timeout": 7 }), &ctx)
            .await
            .unwrap();

        let calls = fake.calls();
        assert_eq!(calls.len(), 1, "the tool ran exactly one exec");
        assert_eq!(calls[0].command, "echo hi");
        assert_eq!(calls[0].cwd, PathBuf::from("/somewhere"));
        assert_eq!(calls[0].timeout, Some(Duration::from_secs(7)));
        assert!(
            out.model_text.contains("fake output"),
            "the ops' output is the tool result"
        );
    }

    #[tokio::test]
    async fn bash_tool_forwards_exec_ops_chunks_as_progress_updates() {
        use crate::tools::exec::testing::FakeExecOps;

        // Streaming crosses the seam: chunks the ops push while the command
        // runs surface through ctx.update, in order — a sandbox backend's
        // output shows live in the transcript exactly like local output.
        let fake =
            std::sync::Arc::new(FakeExecOps::streaming(&["one\n", "two\n"], 0));
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let ctx = ToolCtx::new(".".into(), CancelToken::default(), Some(tx))
            .with_exec_ops(fake);
        BashTool.run(&json!({ "command": "whatever" }), &ctx).await.unwrap();
        drop(ctx); // release the sink so the receiver closes

        let mut chunks = Vec::new();
        while let Some(c) = rx.recv().await {
            chunks.push(c);
        }
        assert_eq!(
            chunks,
            ["one\n", "two\n"],
            "the ops' chunks arrived as updates, in order"
        );
    }

    #[tokio::test]
    async fn bash_tool_streams_output_through_ctx_update() {
        // The context's update sink is bash's streaming channel: each line is
        // delivered through it while the command runs.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
        let ctx = ToolCtx::new(".".into(), CancelToken::default(), Some(tx));
        let out = BashTool
            .run(&json!({ "command": "printf 'one\\ntwo\\n'" }), &ctx)
            .await
            .unwrap();
        drop(ctx); // release the sink so the receiver closes

        let mut chunks = Vec::new();
        while let Some(c) = rx.recv().await {
            chunks.push(c);
        }
        let streamed: String = chunks.concat();
        assert!(
            streamed.contains("one") && streamed.contains("two"),
            "the deltas stream through ctx.update, got: {streamed:?}",
        );
        assert!(
            out.model_text.contains("one"),
            "the final result still has the output"
        );
    }

    #[test]
    fn flags_clearly_destructive_commands() {
        assert!(destructive_reason("rm -rf /").is_some());
        assert!(destructive_reason("rm -rf /*").is_some());
        assert!(destructive_reason("rm -fr ~").is_some());
        assert!(destructive_reason("sudo rm -rf $HOME").is_some());
        assert!(destructive_reason("rm -rf .").is_some());
        assert!(destructive_reason("rm --recursive --force /").is_some());
        assert!(destructive_reason("mkfs.ext4 /dev/sda1").is_some());
        assert!(destructive_reason("dd if=/dev/zero of=/dev/sda").is_some());
        assert!(destructive_reason("echo hi > /dev/sda").is_some());
        assert!(destructive_reason(":(){ :|:& };:").is_some());
        assert!(
            destructive_reason("true && rm -rf /").is_some(),
            "checks each segment"
        );
    }

    #[test]
    fn allows_ordinary_commands() {
        assert!(destructive_reason("ls -la").is_none());
        assert!(destructive_reason("rm -rf target/debug").is_none());
        assert!(destructive_reason("rm file.txt").is_none());
        assert!(destructive_reason("cargo build && ./run").is_none());
        assert!(
            destructive_reason("grep -rf pattern src").is_none(),
            "-rf to grep is fine"
        );
        assert!(destructive_reason("dd if=a of=b").is_none());
    }
}
