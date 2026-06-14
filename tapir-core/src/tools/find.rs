//! `find` — locate files by glob, respecting `.gitignore` (`globset` for
//! matching, the shared [`walk`](super::walk) for traversal).
//!
//! Glob compilation, matching and the result limit live here; the walk
//! performs every filesystem touch through the execution context's
//! [`FsOps`](super::fs::FsOps).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use globset::Glob;
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{ToolResult, arg_opt_str, arg_opt_u64, arg_str, walk};

/// The file-find tool behind the [`Tool`] trait.
pub struct FindTool;

#[async_trait]
impl Tool for FindTool {
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

pub const NAME: &str = "find";
pub const DESCRIPTION: &str = "Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results.";
pub const SNIPPET: &str = "Find files by glob pattern (respects .gitignore)";
pub const GUIDELINES: &[&str] = &[];
const DEFAULT_LIMIT: usize = 1000;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'" },
            "path": { "type": "string", "description": "Directory to search in (default: current directory)" },
            "limit": { "type": "number", "description": "Maximum number of results (default: 1000)" }
        },
        "required": ["pattern"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    format!(
        "find {}",
        args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?")
    )
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let pattern = arg_str(args, "pattern")?;
    let dir = match arg_opt_str(args, "path") {
        Some(p) => ctx.resolve(&p)?,
        None => ctx.host_cwd()?,
    };
    let limit =
        arg_opt_u64(args, "limit").map(|n| n as usize).unwrap_or(DEFAULT_LIMIT);

    // A pattern without a slash matches anywhere in the tree.
    let glob_pat = if pattern.contains('/') {
        pattern.clone()
    } else {
        format!("**/{pattern}")
    };
    let matcher = Glob::new(&glob_pat)
        .map_err(|e| anyhow!("invalid glob `{pattern}`: {e}"))?
        .compile_matcher();

    let mut results: Vec<String> = Vec::new();
    let mut limited = false;
    for rel in walk::files(ctx.fs_ops(), &dir).await {
        if matcher.is_match(&rel) {
            if results.len() >= limit {
                limited = true;
                break;
            }
            results.push(rel.to_string_lossy().into_owned());
        }
    }
    results.sort();
    let mut out = results.join("\n");
    if limited {
        out.push_str(&format!("\n\n[Truncated to {limit} results.]"));
    }
    Ok(ToolResult::text(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn find_tool_walks_through_the_fs_ops_on_its_context() {
        use crate::tools::fs::testing::FakeFsOps;

        // The seam: the whole walk — directory listings, the .gitignore read,
        // the glob matching against what comes back — runs on the injected
        // ops' tree. Nothing exists on disk.
        let fake = std::sync::Arc::new(
            FakeFsOps::new()
                .file("/fake/src/main.rs", "")
                .file("/fake/src/lib.rs", "")
                .file("/fake/notes.txt", "")
                .file("/fake/.gitignore", "lib.rs\n"),
        );
        let ctx = ToolCtx::new("/fake".into(), CancelToken::default(), None)
            .with_fs_ops(fake);
        let out =
            FindTool.run(&json!({ "pattern": "*.rs" }), &ctx).await.unwrap();
        assert!(out.model_text.contains("src/main.rs"));
        assert!(
            !out.model_text.contains("lib.rs"),
            "the fake's .gitignore hides lib.rs"
        );
        assert!(!out.model_text.contains("notes.txt"));
    }

    #[tokio::test]
    async fn find_tool_finds_through_the_trait() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-findtool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "").unwrap();
        std::fs::write(dir.join("notes.txt"), "").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out =
            FindTool.run(&json!({ "pattern": "*.rs" }), &ctx).await.unwrap();
        assert!(
            out.model_text.contains("src/main.rs")
                && !out.model_text.contains("notes.txt")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn finds_by_glob_and_respects_gitignore() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-find-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), "").unwrap();
        std::fs::write(dir.join("src/lib.rs"), "").unwrap();
        std::fs::write(dir.join("notes.txt"), "").unwrap();
        std::fs::write(dir.join(".gitignore"), "lib.rs\n").unwrap();

        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(&json!({"pattern": "*.rs"}), &ctx).await.unwrap();
        assert!(out.model_text.contains("src/main.rs"));
        assert!(
            !out.model_text.contains("lib.rs"),
            "gitignore should hide lib.rs"
        );
        assert!(!out.model_text.contains("notes.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
