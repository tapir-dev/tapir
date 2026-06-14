//! `grep` — search file contents (the ripgrep engine via the `grep` crate),
//! walking with the shared [`walk`](super::walk) (respects `.gitignore`) and
//! optional glob filtering.
//!
//! Pattern compilation, match formatting, context lines and limits live
//! here; the walk and every file read go through the execution context's
//! [`FsOps`](super::fs::FsOps).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use globset::Glob;
use grep::regex::RegexMatcherBuilder;
use grep::searcher::{Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};
use serde_json::{Value, json};

use super::tool::{Tool, ToolCtx};
use super::{
    ToolResult, arg_bool, arg_opt_str, arg_opt_u64, arg_str, truncate, walk,
};

/// The content-search tool behind the [`Tool`] trait.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
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

pub const NAME: &str = "grep";
pub const DESCRIPTION: &str = "Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB. Long lines are truncated to 500 chars.";
pub const SNIPPET: &str =
    "Search file contents for patterns (respects .gitignore)";
pub const GUIDELINES: &[&str] = &[];
const DEFAULT_LIMIT: usize = 100;

pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Search pattern (regex or literal string)" },
            "path": { "type": "string", "description": "Directory or file to search (default: current directory)" },
            "glob": { "type": "string", "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'" },
            "ignoreCase": { "type": "boolean", "description": "Case-insensitive search (default: false)" },
            "literal": { "type": "boolean", "description": "Treat pattern as a literal string instead of regex (default: false)" },
            "context": { "type": "number", "description": "Lines of context before and after each match (default: 0)" },
            "limit": { "type": "number", "description": "Maximum number of matches to return (default: 100)" }
        },
        "required": ["pattern"],
        "additionalProperties": false
    })
}

pub fn title(args: &Value) -> String {
    let pat = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
    let path = arg_opt_str(args, "path");
    match path {
        Some(p) => format!("grep /{pat}/ in {p}"),
        None => format!("grep /{pat}/"),
    }
}

/// Collects matches and context lines for one file.
struct Collector<'a> {
    path: String,
    out: &'a mut Vec<String>,
    remaining: usize,
    matched: usize,
}

fn clip(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    let s = s.trim_end_matches(['\n', '\r']);
    if s.chars().count() > truncate::GREP_MAX_LINE {
        s.chars().take(truncate::GREP_MAX_LINE).collect::<String>() + "…"
    } else {
        s.to_string()
    }
}

impl Sink for Collector<'_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _: &Searcher,
        m: &SinkMatch<'_>,
    ) -> Result<bool, std::io::Error> {
        if self.matched >= self.remaining {
            return Ok(false);
        }
        let lnum = m.line_number().unwrap_or(0);
        self.out.push(format!("{}:{lnum}: {}", self.path, clip(m.bytes())));
        self.matched += 1;
        Ok(self.matched < self.remaining)
    }

    fn context(
        &mut self,
        _: &Searcher,
        c: &SinkContext<'_>,
    ) -> Result<bool, std::io::Error> {
        let lnum = c.line_number().unwrap_or(0);
        self.out.push(format!("{}-{lnum}- {}", self.path, clip(c.bytes())));
        Ok(true)
    }
}

pub async fn run(args: &Value, ctx: &ToolCtx) -> Result<ToolResult> {
    let pattern = arg_str(args, "pattern")?;
    let root = match arg_opt_str(args, "path") {
        Some(p) => ctx.resolve(&p)?,
        None => ctx.host_cwd()?,
    };
    let context = arg_opt_u64(args, "context").unwrap_or(0) as usize;
    let limit =
        arg_opt_u64(args, "limit").map(|n| n as usize).unwrap_or(DEFAULT_LIMIT);

    let matcher = RegexMatcherBuilder::new()
        .case_insensitive(arg_bool(args, "ignoreCase"))
        .fixed_strings(arg_bool(args, "literal"))
        .build(&pattern)
        .map_err(|e| anyhow!("invalid pattern: {e}"))?;
    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .before_context(context)
        .after_context(context)
        .build();

    let glob = arg_opt_str(args, "glob")
        .map(|g| Glob::new(&g).map(|g| g.compile_matcher()))
        .transpose()
        .map_err(|e| anyhow!("invalid glob: {e}"))?;

    let mut out: Vec<String> = Vec::new();
    let mut remaining = limit;
    let mut limited = false;
    for rel in walk::files(ctx.fs_ops(), &root).await {
        if let Some(glob) = &glob
            && !glob.is_match(&rel)
        {
            continue;
        }
        // An empty rel is the root itself being a file.
        let abs = if rel.as_os_str().is_empty() {
            root.clone()
        } else {
            root.join(&rel)
        };
        let Ok(bytes) = ctx.fs_ops().read(&abs).await else { continue };
        let mut collector = Collector {
            path: rel.to_string_lossy().into_owned(),
            out: &mut out,
            remaining,
            matched: 0,
        };
        // `&mut Collector: Sink` (blanket impl), so the collector isn't moved
        // and we can read its match count afterward.
        let _ = searcher.search_slice(&matcher, &bytes, &mut collector);
        let got = collector.matched;
        drop(collector);
        remaining = remaining.saturating_sub(got);
        if remaining == 0 {
            limited = true;
            break;
        }
    }

    let mut text = out.join("\n");
    if limited {
        text.push_str(&format!("\n\n[Truncated to {limit} matches.]"));
    } else if text.is_empty() {
        text = "No matches found.".to_string();
    }
    Ok(ToolResult::text(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CancelToken;
    use crate::tools::tool::{Tool, ToolCtx};

    #[tokio::test]
    async fn grep_tool_searches_through_the_fs_ops_on_its_context() {
        use crate::tools::fs::testing::FakeFsOps;

        // The seam: the walk and every file read run on the injected ops'
        // tree — matches come back with the same path:line formatting, and
        // nothing exists on disk.
        let fake = std::sync::Arc::new(
            FakeFsOps::new()
                .file(
                    "/fake/a.rs",
                    "fn main() {}\nlet x = 1;\nfn helper() {}\n",
                )
                .file("/fake/b.txt", "no functions here\n"),
        );
        let ctx = ToolCtx::new("/fake".into(), CancelToken::default(), None)
            .with_fs_ops(fake);
        let out = GrepTool
            .run(&json!({ "pattern": "fn ", "literal": true }), &ctx)
            .await
            .unwrap();
        assert!(
            out.model_text.contains("a.rs:1:")
                && out.model_text.contains("a.rs:3:")
        );
        assert!(
            !out.model_text.contains("let x")
                && !out.model_text.contains("b.txt")
        );
    }

    #[tokio::test]
    async fn grep_tool_searches_through_the_trait() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-greptool-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "fn main() {}\nlet x = 1;\n").unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = GrepTool
            .run(&json!({ "pattern": "fn ", "literal": true }), &ctx)
            .await
            .unwrap();
        assert!(
            out.model_text.contains("a.rs:1:")
                && !out.model_text.contains("let x")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn finds_matches_with_line_numbers() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-grep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.rs"),
            "fn main() {}\nlet x = 1;\nfn helper() {}\n",
        )
        .unwrap();
        let ctx = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let out = run(&json!({"pattern": "fn ", "literal": true}), &ctx)
            .await
            .unwrap();
        assert!(out.model_text.contains("a.rs:1:"));
        assert!(out.model_text.contains("a.rs:3:"));
        assert!(!out.model_text.contains("let x"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
