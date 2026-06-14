//! `edit` — exact-text replacement(s) in one file, with a +/- diff for display.
//!
//! Each `oldText` must match a unique, non-overlapping region. Edits are matched
//! against the original content and applied in reverse order so offsets stay
//! valid. (Matching is exact-only — no fuzzy/normalized matching.)
//!
//! All tool logic — matching, overlap checks, diff generation — lives here;
//! the raw file read and write go through the execution context's
//! [`FsOps`](super::fs::FsOps).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_str};

/// The file-edit tool behind the [`Tool`] trait. Its result keeps the `+/-`
/// diff in `display`, so the TUI renders it as a diff.
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
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

pub const NAME: &str = "edit";
pub const DESCRIPTION: &str = "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits.";
pub const SNIPPET: &str = "Make precise file edits with exact text replacement, including multiple disjoint edits in one call";
pub const GUIDELINES: &[&str] = &[
    "Use edit for precise changes (edits[].oldText must match exactly)",
    "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
    "Each edits[].oldText is matched against the original file, not after earlier edits; do not emit overlapping or nested edits",
    "Keep edits[].oldText as small as possible while still being unique in the file",
];

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
            "edits": {
                "type": "array",
                "description": "One or more targeted replacements. Each is matched against the original file (not incrementally). Do not include overlapping or nested edits.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": { "type": "string", "description": "Exact text for one targeted replacement. Must be unique in the file." },
                        "newText": { "type": "string", "description": "Replacement text for this edit." }
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["path", "edits"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    format!("edit {}", args.get("path").and_then(|v| v.as_str()).unwrap_or("?"))
}

struct Replacement {
    start: usize,
    end: usize,
    old: String,
    new: String,
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let path_arg = arg_str(args, "path")?;
    let path = ctx.resolve(&path_arg)?;

    let edits = args
        .get("edits")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("missing `edits` array"))?;
    if edits.is_empty() {
        return Err(anyhow!("`edits` must contain at least one edit"));
    }

    let bytes = ctx
        .fs_ops()
        .read(&path)
        .await
        .map_err(|e| anyhow!("cannot read {}: {e}", path.display()))?;
    let content = String::from_utf8(bytes).map_err(|_| {
        anyhow!(
            "cannot read {}: stream did not contain valid UTF-8",
            path.display()
        )
    })?;

    let mut reps: Vec<Replacement> = Vec::new();
    for edit in edits {
        let old = arg_str(edit, "oldText")?;
        let new = arg_str(edit, "newText")?;
        if old.is_empty() {
            return Err(anyhow!("oldText must not be empty"));
        }
        let count = content.matches(&old).count();
        if count == 0 {
            return Err(anyhow!("oldText not found in {path_arg}: {old:?}"));
        }
        if count > 1 {
            return Err(anyhow!(
                "oldText is not unique in {path_arg} ({count} occurrences): {old:?}"
            ));
        }
        let start = content.find(&old).unwrap();
        reps.push(Replacement { start, end: start + old.len(), old, new });
    }

    // Overlap check (by original-text byte ranges).
    let mut by_pos: Vec<&Replacement> = reps.iter().collect();
    by_pos.sort_by_key(|r| r.start);
    for w in by_pos.windows(2) {
        if w[1].start < w[0].end {
            return Err(anyhow!("edits overlap; merge them into one edit"));
        }
    }

    // Apply in reverse order so earlier offsets remain valid.
    let mut result = content.clone();
    let mut in_order: Vec<&Replacement> = reps.iter().collect();
    in_order.sort_by_key(|r| std::cmp::Reverse(r.start));
    for r in &in_order {
        result.replace_range(r.start..r.end, &r.new);
    }
    if result == content {
        return Err(anyhow!("edit produced no change"));
    }

    ctx.fs_ops()
        .write(&path, result.as_bytes())
        .await
        .map_err(|e| anyhow!("cannot write {}: {e}", path.display()))?;

    // Display diff: a real line-level diff per edit (deletions `-`, insertions
    // `+`, unchanged context ` `), via the `similar` crate.
    let mut diff = String::new();
    for r in &reps {
        for change in TextDiff::from_lines(&r.old, &r.new).iter_all_changes() {
            let sign = match change.tag() {
                ChangeTag::Delete => '-',
                ChangeTag::Insert => '+',
                ChangeTag::Equal => ' ',
            };
            let line = change.value();
            let line = line.strip_suffix('\n').unwrap_or(line);
            diff.push_str(&format!("{sign}{line}\n"));
        }
    }

    Ok(ToolResult::shown(
        format!("Successfully replaced {} block(s) in {path_arg}.", reps.len()),
        diff.trim_end().to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn edit_tool_renders_a_diff_through_the_trait() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-edittool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "hello world\nbye").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = EditTool
            .run(
                &json!({ "path": "f.txt", "edits": [{ "oldText": "world", "newText": "there" }] }),
                &ctx,
            )
            .await
            .unwrap();
        // The result still carries the +/- diff for the TUI to render.
        assert!(
            out.display.contains("-world") && out.display.contains("+there")
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("f.txt")).unwrap(),
            "hello there\nbye"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tmp(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("tapir-edit-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn replaces_unique_text() {
        let dir = tmp("replace");
        std::fs::write(dir.join("f.txt"), "hello world\nbye").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(
            &json!({"path": "f.txt", "edits": [{"oldText": "world", "newText": "there"}]}),
            &ctx,
        )
        .await
        .unwrap();
        assert!(out.model_text.contains("1 block"));
        assert!(out.display.contains("-world"));
        assert!(out.display.contains("+there"));
        assert_eq!(
            std::fs::read_to_string(dir.join("f.txt")).unwrap(),
            "hello there\nbye"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn rejects_non_unique() {
        let dir = tmp("nonunique");
        std::fs::write(dir.join("f.txt"), "a\na\n").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let err = run(
            &json!({"path": "f.txt", "edits": [{"oldText": "a", "newText": "b"}]}),
            &ctx,
        )
        .await;
        assert!(err.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
