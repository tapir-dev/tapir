//! `ls` — list a directory (sorted, `/` suffix for dirs, dotfiles included).
//!
//! Sorting, suffixing and the entry limit live here; the raw directory
//! listing goes through the execution context's [`FsOps`](super::fs::FsOps).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_opt_str, arg_opt_u64};

/// The directory-list tool behind the [`Tool`] trait.
pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
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

pub const NAME: &str = "ls";
pub const DESCRIPTION: &str = "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries.";
pub const SNIPPET: &str = "List directory contents";
pub const GUIDELINES: &[&str] = &[];
const DEFAULT_LIMIT: usize = 500;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Directory to list (default: current directory)" },
            "limit": { "type": "number", "description": "Maximum number of entries to return (default: 500)" }
        },
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    format!("ls {}", arg_opt_str(args, "path").as_deref().unwrap_or("."))
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let dir = match arg_opt_str(args, "path") {
        Some(p) => ctx.resolve(&p)?,
        None => ctx.host_cwd()?,
    };
    let limit =
        arg_opt_u64(args, "limit").map(|n| n as usize).unwrap_or(DEFAULT_LIMIT);

    let mut entries: Vec<String> = ctx
        .fs_ops()
        .list_dir(&dir)
        .await
        .map_err(|e| anyhow!("cannot list {}: {e}", dir.display()))?
        .into_iter()
        .map(|e| if e.is_dir { format!("{}/", e.name) } else { e.name })
        .collect();
    entries.sort_by_key(|e| e.to_lowercase());

    let limited = entries.len() > limit;
    entries.truncate(limit);
    let mut out = entries.join("\n");
    if limited {
        out.push_str(&format!("\n\n[Showing first {limit} entries.]"));
    }
    Ok(ToolResult::text(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn ls_tool_lists_through_the_fs_ops_on_its_context() {
        use crate::tools::fs::testing::FakeFsOps;

        // The seam: the listing comes from the injected ops' tree — names
        // sorted, directories suffixed — with nothing on disk.
        let fake = std::sync::Arc::new(
            FakeFsOps::new()
                .file("/fake/a.txt", "")
                .file("/fake/sub/inner.txt", ""),
        );
        let ctx = ToolCtx::new("/fake".into(), CancelToken::default(), None)
            .with_fs_ops(fake);
        let out = LsTool.run(&json!({}), &ctx).await.unwrap();
        assert_eq!(out.model_text, "a.txt\nsub/");
    }

    #[tokio::test]
    async fn ls_tool_lists_through_the_trait() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-lstool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), "").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = LsTool.run(&json!({}), &ctx).await.unwrap();
        assert!(
            out.model_text.contains("a.txt") && out.model_text.contains("sub/")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn lists_sorted_with_dir_suffix() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-ls-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), "").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(&json!({}), &ctx).await.unwrap();
        assert!(out.model_text.contains("a.txt"));
        assert!(out.model_text.contains("sub/"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
