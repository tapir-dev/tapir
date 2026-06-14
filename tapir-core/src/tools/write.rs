//! `write` — create/overwrite a file, making parent directories.
//!
//! The raw mkdir + file write go through the execution context's
//! [`FsOps`](super::fs::FsOps) (the local host by default; a sandbox backend
//! when injected).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_str};

/// The file-write tool behind the [`Tool`] trait.
pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
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

pub const NAME: &str = "write";
pub const DESCRIPTION: &str = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.";
pub const SNIPPET: &str = "Create or overwrite files";
pub const GUIDELINES: &[&str] =
    &["Use write only for new files or complete rewrites."];

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
            "content": { "type": "string", "description": "Content to write to the file" }
        },
        "required": ["path", "content"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    format!(
        "write {}",
        args.get("path").and_then(|v| v.as_str()).unwrap_or("?")
    )
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let path_arg = arg_str(args, "path")?;
    let content = arg_str(args, "content")?;
    let path = ctx.resolve(&path_arg)?;
    if let Some(parent) = path.parent() {
        ctx.fs_ops()
            .create_dir_all(parent)
            .await
            .map_err(|e| anyhow!("cannot create {}: {e}", parent.display()))?;
    }
    ctx.fs_ops()
        .write(&path, content.as_bytes())
        .await
        .map_err(|e| anyhow!("cannot write {}: {e}", path.display()))?;
    Ok(ToolResult::shown(
        format!("Successfully wrote {} bytes to {}", content.len(), path_arg),
        content,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn write_tool_writes_through_the_trait() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-writetool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = WriteTool
            .run(&json!({ "path": "sub/f.txt", "content": "hi" }), &ctx)
            .await
            .unwrap();
        assert!(out.model_text.contains("2 bytes"));
        assert_eq!(
            std::fs::read_to_string(dir.join("sub/f.txt")).unwrap(),
            "hi"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn writes_and_creates_parents() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(&json!({"path": "sub/f.txt", "content": "hi"}), &ctx)
            .await
            .unwrap();
        assert!(out.model_text.contains("2 bytes"));
        assert_eq!(
            std::fs::read_to_string(dir.join("sub/f.txt")).unwrap(),
            "hi"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
