//! `read` — read a text file, with optional 1-indexed offset/limit.
//!
//! All tool logic — offset/limit, truncation, continuation hints — lives
//! here; the raw file read goes through the execution context's
//! [`FsOps`](super::fs::FsOps) (the local host by default; a sandbox backend
//! when injected).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_opt_u64, arg_str, truncate};

/// The file-read tool, behind the [`Tool`] trait. The model-facing metadata
/// mirrors the module constants (one source); `run` reuses [`run`](run()).
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
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

pub const NAME: &str = "read";
pub const DESCRIPTION: &str = "Read the contents of a text file. Output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files; continue with offset until complete.";
/// One-line summary for the system prompt's "Available tools" list.
pub const SNIPPET: &str = "Read file contents";
/// Behavioral guidelines contributed to the system prompt.
pub const GUIDELINES: &[&str] =
    &["Use read to examine files instead of cat or sed."];

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
            "offset": { "type": "number", "description": "Line number to start reading from (1-indexed)" },
            "limit": { "type": "number", "description": "Maximum number of lines to read" }
        },
        "required": ["path"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
    match arg_opt_u64(args, "offset") {
        Some(o) => format!("read {path}:{o}"),
        None => format!("read {path}"),
    }
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let path_arg = arg_str(args, "path")?;
    let path = ctx.resolve(&path_arg)?;
    let offset = arg_opt_u64(args, "offset").unwrap_or(0) as usize;
    let limit = arg_opt_u64(args, "limit").map(|n| n as usize);

    let bytes = ctx
        .fs_ops()
        .read(&path)
        .await
        .map_err(|e| anyhow!("cannot read {}: {e}", path.display()))?;
    let text = String::from_utf8(bytes)
        .map_err(|_| anyhow!("{} is not a UTF-8 text file", path.display()))?;

    let all: Vec<&str> = text.split('\n').collect();
    let total = all.len();
    let start = offset.saturating_sub(1);
    if offset > 0 && start >= all.len() {
        return Err(anyhow!(
            "Offset {offset} is beyond end of file ({total} lines total)"
        ));
    }

    let selected = match limit {
        Some(lim) => all[start..(start + lim).min(all.len())].join("\n"),
        None => all[start..].join("\n"),
    };
    let trunc = truncate::head(&selected);
    let mut out = trunc.content;
    if trunc.truncated {
        let start_display = start + 1;
        let end_display = start_display + trunc.output_lines.saturating_sub(1);
        let next = end_display + 1;
        out.push_str(&format!(
            "\n\n[Showing lines {start_display}-{end_display} of {total}. Use offset={next} to continue.]"
        ));
    } else if let Some(lim) = limit
        && start + lim < all.len()
    {
        let remaining = all.len() - (start + lim);
        let next = start + lim + 1;
        out.push_str(&format!(
            "\n\n[{remaining} more lines in file. Use offset={next} to continue.]"
        ));
    }
    Ok(ToolResult::text(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};
    use serde_json::json;

    #[tokio::test]
    async fn read_tool_reads_through_the_fs_ops_on_its_context() {
        use crate::tools::fs::testing::FakeFsOps;

        // The seam: the tool performs no filesystem I/O itself — the fs
        // operations on its context serve the file, and their content is the
        // tool's result. Nothing exists on disk at this path.
        let fake = std::sync::Arc::new(
            FakeFsOps::new().file("/fake/hello.txt", "alpha\nbeta"),
        );
        let ctx = ToolCtx::new("/fake".into(), CancelToken::default(), None)
            .with_fs_ops(fake.clone());
        let out =
            ReadTool.run(&json!({ "path": "hello.txt" }), &ctx).await.unwrap();
        assert!(
            out.model_text.contains("alpha") && out.model_text.contains("beta")
        );

        let reads = fake.reads();
        assert_eq!(
            reads,
            [std::path::PathBuf::from("/fake/hello.txt")],
            "the tool read exactly this file through the injected ops"
        );
    }

    #[tokio::test]
    async fn read_tool_reads_a_file_through_the_trait() {
        // The read tool is reached through the Tool trait + execution context,
        // not the old name-dispatch.
        let dir = std::env::temp_dir()
            .join(format!("tapir-readtool-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("hello.txt"), "alpha\nbeta\ngamma").unwrap();

        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let result =
            ReadTool.run(&json!({ "path": "hello.txt" }), &ctx).await.unwrap();
        assert!(
            result.model_text.contains("alpha")
                && result.model_text.contains("gamma")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn reads_with_offset_and_limit() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-read-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("f.txt"), "a\nb\nc\nd\ne").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(&json!({"path": "f.txt", "offset": 2, "limit": 2}), &ctx)
            .await
            .unwrap();
        assert!(out.model_text.starts_with("b\nc"));
        assert!(out.model_text.contains("more lines in file"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
