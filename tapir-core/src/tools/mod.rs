//! The agent's tools: read, write, edit, bash, ls, grep, find.
//!
//! `find`/`grep` walk with the shared [`walk`] (`.gitignore`-aware, every
//! filesystem touch through [`fs::FsOps`]) and match with `globset` / the
//! ripgrep engine instead of shelling out to `fd`/`rg`. `bash` executes
//! through [`exec::ExecOps`]; the file tools' raw I/O goes through
//! [`fs::FsOps`].
//!
//! Each tool exposes `NAME`, `DESCRIPTION`, `schema()` (its JSON-Schema
//! parameters), `title()` (the TUI header), and `run()`. //! list sent to the model; `dispatch` dispatches a call.

pub mod exec;
pub mod fs;
pub mod jail;
pub mod tool;
pub mod truncate;

// The pair you implement a custom tool with, re-exported at the `tools` root
// (also available at their canonical `tools::tool::` path).
pub use tool::{Tool, ToolCtx};

mod bash;
mod edit;
mod find;
mod grep;
mod ls;
mod read;
mod walk;
mod write;

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde_json::Value;

/// A sink a streaming tool pushes incremental output chunks to as it runs; the
/// turn loop forwards each chunk as a [`ToolUpdate`](crate::agent::TurnEvent::ToolUpdate).
/// `None` when the caller doesn't want live progress (e.g. headless print).
pub type Progress = Option<tokio::sync::mpsc::Sender<String>>;

/// A shared registry of [`Tool`](tool::Tool)s the turn loop dispatches through.
pub type Tools = std::sync::Arc<[std::sync::Arc<dyn tool::Tool>]>;

/// The built-in tools that have been converted to the [`Tool`](tool::Tool)
/// trait (the rest still run through `dispatch`). Seeds the Runtime's registry;
/// grows as the remaining built-ins are converted.
pub fn builtin_tools() -> Vec<std::sync::Arc<dyn tool::Tool>> {
    vec![
        std::sync::Arc::new(read::ReadTool),
        std::sync::Arc::new(write::WriteTool),
        std::sync::Arc::new(edit::EditTool),
        std::sync::Arc::new(bash::BashTool),
        std::sync::Arc::new(ls::LsTool),
        std::sync::Arc::new(grep::GrepTool),
        std::sync::Arc::new(find::FindTool),
    ]
}

/// A finished tool call. `model_text` is sent back to the model; `display` is
/// shown in the transcript (usually the same, but e.g. `edit` shows a diff).
#[derive(Debug)]
pub struct ToolResult {
    pub model_text: String,
    pub display: String,
}

impl ToolResult {
    pub fn text(s: impl Into<String>) -> Self {
        let s = s.into();
        Self { display: s.clone(), model_text: s }
    }

    pub fn shown(
        model_text: impl Into<String>,
        display: impl Into<String>,
    ) -> Self {
        Self { model_text: model_text.into(), display: display.into() }
    }
}

/// A tool's model-facing definition (sent in the request's `tools` array). The
/// schema type lives in the LLM layer (`tapir-ai`); the registry and execution
/// that produce these are here.
pub use tapir_ai::message::ToolDef;

/// All tools, in a stable order — each describes itself through the
/// [`Tool`](tool::Tool) trait (one source of truth). Built-ins only — a
/// session's full set (incl. custom tools) comes from `Agent::tool_definitions`.
pub(crate) fn definitions() -> Vec<ToolDef> {
    builtin_tools()
        .iter()
        .map(|t| ToolDef {
            name: t.name(),
            description: t.description(),
            parameters: t.parameters(),
        })
        .collect()
}

/// The names of all built-in tools, in [`definitions`] order.
pub fn names() -> Vec<&'static str> {
    definitions().into_iter().map(|d| d.name).collect()
}

/// The active tool set after applying the CLI tool flags (precedence):
/// `allow` (`--tools`) is an exact allowlist; else `--no-tools`/`--no-builtin`
/// start from nothing; else all built-ins. Then `exclude` (`--exclude-tools`)
/// is removed. Only known tool names survive.
pub fn resolve_active(
    allow: &[String],
    exclude: &[String],
    no_tools: bool,
    no_builtin: bool,
) -> Vec<String> {
    let all = names();
    let base: Vec<String> = if !allow.is_empty() {
        allow.to_vec()
    } else if no_tools || no_builtin {
        Vec::new()
    } else {
        all.iter().map(|s| s.to_string()).collect()
    };
    let ex: std::collections::HashSet<&str> =
        exclude.iter().map(String::as_str).collect();
    base.into_iter()
        .filter(|n| !ex.contains(n.as_str()))
        .filter(|n| all.iter().any(|a| a == n))
        .collect()
}

/// One-line `(name, snippet)` pairs for the system prompt's "Available tools"
/// list (same order as [`definitions`]).
pub fn snippets() -> Vec<(&'static str, &'static str)> {
    vec![
        (read::NAME, read::SNIPPET),
        (write::NAME, write::SNIPPET),
        (edit::NAME, edit::SNIPPET),
        (bash::NAME, bash::SNIPPET),
        (ls::NAME, ls::SNIPPET),
        (grep::NAME, grep::SNIPPET),
        (find::NAME, find::SNIPPET),
    ]
}

/// [`snippets`] restricted to the `active` tool names.
pub fn snippets_for(active: &[String]) -> Vec<(&'static str, &'static str)> {
    snippets()
        .into_iter()
        .filter(|(n, _)| active.iter().any(|a| a == n))
        .collect()
}

/// The per-tool guideline groups, keyed by tool name (same order as
/// [`definitions`]).
fn guideline_groups() -> [(&'static str, &'static [&'static str]); 7] {
    [
        (read::NAME, read::GUIDELINES),
        (write::NAME, write::GUIDELINES),
        (edit::NAME, edit::GUIDELINES),
        (bash::NAME, bash::GUIDELINES),
        (ls::NAME, ls::GUIDELINES),
        (grep::NAME, grep::GUIDELINES),
        (find::NAME, find::GUIDELINES),
    ]
}

/// Each active tool's behavioral guidelines, in tool order and de-duplicated,
/// for the system prompt's "Guidelines" section.
pub fn guidelines_for(active: &[String]) -> Vec<&'static str> {
    let mut out = Vec::new();
    for (name, group) in guideline_groups() {
        if active.iter().any(|a| a == name) {
            for g in group {
                if !out.contains(g) {
                    out.push(*g);
                }
            }
        }
    }
    out
}

/// Dispatch a tool call through the registered [`Tool`](tool::Tool) trait. Every
/// built-in is registered, so an unknown name (a model hallucination) is the only
/// miss. The turn loop and the headless print path both go through here.
pub async fn dispatch(
    registry: &Tools,
    name: &str,
    args: &Value,
    ctx: &tool::ToolCtx,
) -> Result<ToolResult> {
    // Don't start new work on an aborted turn (an abort can land between the
    // model's tool request and here).
    if ctx.is_aborted() {
        return Err(anyhow!("tool call aborted"));
    }
    match registry.iter().find(|t| t.name() == name) {
        Some(t) => t.run(args, ctx).await,
        None => Err(anyhow!("unknown tool: {name}")),
    }
}

// --- shared argument / path helpers ------------------------------------

/// Resolve a path argument against `cwd` (absolute and `~` honored),
/// unrestricted. Tools resolve through [`ToolCtx::resolve`](tool::ToolCtx::resolve)
/// instead, which adds the optional workspace boundary ([`jail`]); this stays
/// the bare helper for boundary-free callers (the TUI's local commands).
pub fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else if let Some(rest) = path.strip_prefix("~/") {
        match std::env::var("HOME") {
            Ok(home) => Path::new(&home).join(rest),
            Err(_) => cwd.join(path),
        }
    } else {
        cwd.join(path)
    }
}

pub(crate) fn arg_str(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow!("missing string argument `{key}`"))
}

pub(crate) fn arg_opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(String::from)
}

pub(crate) fn arg_opt_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

pub(crate) fn arg_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn default_is_all_builtins() {
        assert_eq!(resolve_active(&[], &[], false, false), names());
    }

    #[test]
    fn allowlist_keeps_only_named_known_tools() {
        let active =
            resolve_active(&v(&["read", "bash", "bogus"]), &[], false, false);
        assert_eq!(active, ["read", "bash"], "unknown names are dropped");
    }

    #[test]
    fn exclude_removes_from_the_base() {
        let active = resolve_active(&[], &v(&["bash", "write"]), false, false);
        assert!(!active.iter().any(|n| n == "bash" || n == "write"));
        assert!(active.iter().any(|n| n == "read"));
    }

    #[test]
    fn no_tools_and_no_builtin_disable_everything() {
        assert!(resolve_active(&[], &[], true, false).is_empty());
        // tapir only has built-ins, so --no-builtin-tools is also empty.
        assert!(resolve_active(&[], &[], false, true).is_empty());
    }

    #[test]
    fn allowlist_then_exclude_compose() {
        // --tools read,edit,bash --exclude-tools bash → read, edit.
        let active = resolve_active(
            &v(&["read", "edit", "bash"]),
            &v(&["bash"]),
            false,
            false,
        );
        assert_eq!(active, ["read", "edit"]);
    }
}
