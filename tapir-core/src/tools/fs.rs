//! The filesystem operations seam: where the file tools' raw I/O happens.
//!
//! The file tools (read, write, edit, ls, grep, find) keep all tool logic —
//! schemas, truncation, diff generation, glob matching, match formatting —
//! and perform every filesystem touch through [`FsOps`], carried by their
//! execution context. The default, [`LocalFsOps`], is the host `std::fs`
//! path; a sandbox backend implements the same trait without touching the
//! tools.

use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;

/// What a path is, as `stat` reports it (following symlinks).
#[derive(Clone, Copy, Debug)]
pub struct FsStat {
    pub is_dir: bool,
    pub is_file: bool,
}

/// One entry of a directory listing: its name and (non-following) kind —
/// a symlink is neither a directory nor a file, matching `std::fs::DirEntry`'s
/// `file_type` semantics.
#[derive(Clone, Debug)]
pub struct FsDirEntry {
    pub name: String,
    pub is_dir: bool,
    pub is_file: bool,
}

/// Raw filesystem operations — the narrow seam between the file tools and
/// the machinery that actually touches a filesystem (the local host today; a
/// container or microVM later). Per-session: carried by the tool context and
/// configured through the runtime builder.
///
/// `Err` carries the underlying I/O failure; the tools wrap it in their own
/// user-facing messages.
#[async_trait]
pub trait FsOps: Send + Sync {
    /// Read the entire file at `path`.
    async fn read(&self, path: &Path) -> Result<Vec<u8>>;

    /// Create or overwrite the file at `path` with `content`. The parent
    /// directory must already exist (the tools mkdir first).
    async fn write(&self, path: &Path, content: &[u8]) -> Result<()>;

    /// Create the directory at `path`, including missing ancestors (no-op
    /// when it already exists).
    async fn create_dir_all(&self, path: &Path) -> Result<()>;

    /// List the entries of the directory at `path` (one level, no order
    /// guarantee — the tools sort).
    async fn list_dir(&self, path: &Path) -> Result<Vec<FsDirEntry>>;

    /// What `path` is — directory, file, or neither (following symlinks).
    async fn stat(&self, path: &Path) -> Result<FsStat>;
}

/// The default [`FsOps`]: `std::fs` on the host. This is the file tools'
/// original I/O path, unchanged in behavior.
pub struct LocalFsOps;

#[async_trait]
impl FsOps for LocalFsOps {
    async fn read(&self, path: &Path) -> Result<Vec<u8>> {
        Ok(std::fs::read(path)?)
    }

    async fn write(&self, path: &Path, content: &[u8]) -> Result<()> {
        Ok(std::fs::write(path, content)?)
    }

    async fn create_dir_all(&self, path: &Path) -> Result<()> {
        Ok(std::fs::create_dir_all(path)?)
    }

    async fn list_dir(&self, path: &Path) -> Result<Vec<FsDirEntry>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(path)? {
            // An entry that vanished mid-listing is skipped, not fatal.
            let Ok(entry) = entry else { continue };
            let kind = entry.file_type().ok();
            entries.push(FsDirEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: kind.map(|t| t.is_dir()).unwrap_or(false),
                is_file: kind.map(|t| t.is_file()).unwrap_or(false),
            });
        }
        Ok(entries)
    }

    async fn stat(&self, path: &Path) -> Result<FsStat> {
        let meta = std::fs::metadata(path)?;
        Ok(FsStat { is_dir: meta.is_dir(), is_file: meta.is_file() })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use crate::agent::CancelToken;
    use crate::tools::edit::EditTool;
    use crate::tools::find::FindTool;
    use crate::tools::grep::GrepTool;
    use crate::tools::ls::LsTool;
    use crate::tools::read::ReadTool;
    use crate::tools::tool::{Tool, ToolCtx};
    use crate::tools::write::WriteTool;

    use super::testing::FakeFsOps;

    fn ctx_with(fake: &Arc<FakeFsOps>) -> ToolCtx {
        ToolCtx::new("/fake".into(), CancelToken::default(), None)
            .with_fs_ops(fake.clone())
    }

    #[tokio::test]
    async fn write_read_edit_round_trip_entirely_through_the_fs_ops() {
        // The whole file lifecycle crosses the seam: write creates the file
        // (and its parents) in the injected ops, read serves it back, edit
        // rewrites it — nothing at these paths exists on the host.
        let fake = Arc::new(FakeFsOps::new());
        let ctx = ctx_with(&fake);

        let out = WriteTool
            .run(
                &json!({ "path": "sub/f.txt", "content": "hello world\nbye" }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            out.model_text.contains("15 bytes"),
            "write reported through the seam"
        );

        let out =
            ReadTool.run(&json!({ "path": "sub/f.txt" }), &ctx).await.unwrap();
        assert_eq!(
            out.model_text, "hello world\nbye",
            "read sees the written content"
        );

        let out = EditTool
            .run(
                &json!({ "path": "sub/f.txt", "edits": [{ "oldText": "world", "newText": "there" }] }),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            out.display.contains("-world") && out.display.contains("+there")
        );

        let out =
            ReadTool.run(&json!({ "path": "sub/f.txt" }), &ctx).await.unwrap();
        assert_eq!(
            out.model_text, "hello there\nbye",
            "the edit landed in the fake tree"
        );

        assert!(
            !std::path::Path::new("/fake").exists(),
            "nothing touched the host filesystem"
        );
    }

    // --- symmetry: each tool behaves identically over LocalFsOps and a fake
    // seeded with the same tree ---------------------------------------------

    /// Materialize `tree` twice — as a real temp dir (exercised through the
    /// default [`LocalFsOps`]) and as a [`FakeFsOps`] rooted at `/sym` — and
    /// return a context for each. Tool output never mentions the roots (paths
    /// are argument-relative), so results must match byte for byte.
    fn seeded(
        label: &str,
        tree: &[(&str, &str)],
    ) -> (std::path::PathBuf, ToolCtx, ToolCtx) {
        let dir = std::env::temp_dir()
            .join(format!("tapir-sym-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut fake = FakeFsOps::new().file("/sym/.keep", "");
        std::fs::write(dir.join(".keep"), "").unwrap();
        for (path, content) in tree {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(dir.join(parent)).unwrap();
            }
            std::fs::write(dir.join(path), content).unwrap();
            fake = fake.file(&format!("/sym/{path}"), content);
        }
        let local = ToolCtx::new(dir.clone(), CancelToken::default(), None);
        let fake = ToolCtx::new("/sym".into(), CancelToken::default(), None)
            .with_fs_ops(Arc::new(fake));
        (dir, local, fake)
    }

    /// Run `tool` with `args` against both contexts and assert the results
    /// are identical (model text and display).
    async fn assert_symmetric(
        tool: &dyn Tool,
        args: &serde_json::Value,
        local: &ToolCtx,
        fake: &ToolCtx,
    ) {
        let l = tool.run(args, local).await.unwrap();
        let f = tool.run(args, fake).await.unwrap();
        assert_eq!(
            l.model_text, f.model_text,
            "model text diverged for {args}"
        );
        assert_eq!(l.display, f.display, "display diverged for {args}");
    }

    #[tokio::test]
    async fn read_is_symmetric_across_local_and_fake_ops() {
        // Includes the truncation boundary: more lines than the head cap, so
        // both sides must agree on the continuation footer line-for-line.
        let big: String = (1..=crate::tools::truncate::MAX_LINES + 50)
            .map(|i| format!("line {i}\n"))
            .collect();
        let (dir, local, fake) = seeded(
            "read",
            &[("big.txt", big.as_str()), ("small.txt", "a\nb\nc")],
        );

        assert_symmetric(
            &ReadTool,
            &json!({ "path": "big.txt" }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &ReadTool,
            &json!({ "path": "small.txt" }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &ReadTool,
            &json!({ "path": "small.txt", "offset": 2, "limit": 1 }),
            &local,
            &fake,
        )
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_is_symmetric_across_local_and_fake_ops() {
        let (dir, local, fake) = seeded("write", &[]);

        let args = json!({ "path": "sub/new.txt", "content": "alpha\nbeta\n" });
        assert_symmetric(&WriteTool, &args, &local, &fake).await;
        // The written state matches too, read back through each side's ops.
        assert_symmetric(
            &ReadTool,
            &json!({ "path": "sub/new.txt" }),
            &local,
            &fake,
        )
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn edit_is_symmetric_across_local_and_fake_ops() {
        let content = "fn main() {\n    println!(\"hi\");\n}\n// trailer\n";
        let (dir, local, fake) = seeded("edit", &[("f.rs", content)]);

        // Two disjoint edits in one call: the +/- diff display and the
        // model text must come out identical on both sides.
        let args = json!({ "path": "f.rs", "edits": [
            { "oldText": "println!(\"hi\");", "newText": "println!(\"bye\");" },
            { "oldText": "// trailer", "newText": "// the end" },
        ]});
        assert_symmetric(&EditTool, &args, &local, &fake).await;
        assert_symmetric(&ReadTool, &json!({ "path": "f.rs" }), &local, &fake)
            .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ls_is_symmetric_across_local_and_fake_ops() {
        let (dir, local, fake) = seeded(
            "ls",
            &[
                ("a.txt", ""),
                ("B.txt", ""),
                ("sub/inner.txt", ""),
                (".hidden", ""),
            ],
        );

        assert_symmetric(&LsTool, &json!({}), &local, &fake).await;
        // The entry-limit boundary: both sides truncate to the same names and
        // emit the same footer.
        assert_symmetric(&LsTool, &json!({ "limit": 2 }), &local, &fake).await;
        assert_symmetric(&LsTool, &json!({ "path": "sub" }), &local, &fake)
            .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_is_symmetric_across_local_and_fake_ops() {
        let (dir, local, fake) = seeded(
            "grep",
            &[
                ("a.rs", "fn one() {}\nlet x = 1;\nfn two() {}\n"),
                ("sub/b.rs", "fn three() {}\nfn four() {}\n"),
                ("sub/c.txt", "fn not_rust() {}\n"),
                (".gitignore", "c.txt\n"),
            ],
        );

        assert_symmetric(
            &GrepTool,
            &json!({ "pattern": "fn ", "literal": true }),
            &local,
            &fake,
        )
        .await;
        // Glob filtering and the match limit: both sides must pick the same
        // matches (deterministic walk order) and the same truncation note.
        assert_symmetric(
            &GrepTool,
            &json!({ "pattern": "fn ", "literal": true, "glob": "**/*.rs", "limit": 3 }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &GrepTool,
            &json!({ "pattern": "let", "context": 1, "path": "a.rs" }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &GrepTool,
            &json!({ "pattern": "nowhere" }),
            &local,
            &fake,
        )
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn find_is_symmetric_across_local_and_fake_ops() {
        let (dir, local, fake) = seeded(
            "find",
            &[
                ("src/main.rs", ""),
                ("src/lib.rs", ""),
                ("src/deep/mod.rs", ""),
                ("notes.txt", ""),
                (".gitignore", "lib.rs\n"),
            ],
        );

        assert_symmetric(
            &FindTool,
            &json!({ "pattern": "*.rs" }),
            &local,
            &fake,
        )
        .await;
        // The result limit and a rooted glob: same kept set, same footer.
        assert_symmetric(
            &FindTool,
            &json!({ "pattern": "*.rs", "limit": 1 }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &FindTool,
            &json!({ "pattern": "src/**/*.rs" }),
            &local,
            &fake,
        )
        .await;
        assert_symmetric(
            &FindTool,
            &json!({ "pattern": "*.rs", "path": "src" }),
            &local,
            &fake,
        )
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Mutex;

    use anyhow::anyhow;

    use super::*;

    /// One node of the in-memory tree.
    enum Node {
        File(Vec<u8>),
        Dir,
    }

    /// An in-memory [`FsOps`]: a seeded tree of files and directories, plus a
    /// record of every read — proof that a tool touched the filesystem only
    /// through the seam.
    pub struct FakeFsOps {
        nodes: Mutex<HashMap<PathBuf, Node>>,
        reads: Mutex<Vec<PathBuf>>,
    }

    impl FakeFsOps {
        /// An empty tree.
        pub fn new() -> Self {
            Self {
                nodes: Mutex::new(HashMap::new()),
                reads: Mutex::new(Vec::new()),
            }
        }

        /// Seed a file (creating its ancestor directories).
        pub fn file(self, path: &str, content: &str) -> Self {
            let path = PathBuf::from(path);
            {
                let mut nodes = self.nodes.lock().unwrap();
                for dir in path.ancestors().skip(1) {
                    nodes.insert(dir.to_path_buf(), Node::Dir);
                }
                nodes.insert(path, Node::File(content.as_bytes().to_vec()));
            }
            self
        }

        /// Every path read so far, in order.
        pub fn reads(&self) -> Vec<PathBuf> {
            self.reads.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl FsOps for FakeFsOps {
        async fn read(&self, path: &Path) -> Result<Vec<u8>> {
            self.reads.lock().unwrap().push(path.to_path_buf());
            match self.nodes.lock().unwrap().get(path) {
                Some(Node::File(bytes)) => Ok(bytes.clone()),
                Some(Node::Dir) => Err(anyhow!("Is a directory (os error 21)")),
                None => Err(anyhow!("No such file or directory (os error 2)")),
            }
        }

        async fn write(&self, path: &Path, content: &[u8]) -> Result<()> {
            let mut nodes = self.nodes.lock().unwrap();
            match path.parent().map(|p| nodes.get(p)) {
                Some(Some(Node::Dir)) => {}
                _ => {
                    return Err(anyhow!(
                        "No such file or directory (os error 2)"
                    ));
                }
            }
            if let Some(Node::Dir) = nodes.get(path) {
                return Err(anyhow!("Is a directory (os error 21)"));
            }
            nodes.insert(path.to_path_buf(), Node::File(content.to_vec()));
            Ok(())
        }

        async fn create_dir_all(&self, path: &Path) -> Result<()> {
            let mut nodes = self.nodes.lock().unwrap();
            for dir in std::iter::once(path).chain(path.ancestors().skip(1)) {
                match nodes.get(dir) {
                    Some(Node::File(_)) => {
                        return Err(anyhow!("Not a directory (os error 20)"));
                    }
                    _ => {
                        nodes.insert(dir.to_path_buf(), Node::Dir);
                    }
                }
            }
            Ok(())
        }

        async fn list_dir(&self, path: &Path) -> Result<Vec<FsDirEntry>> {
            let nodes = self.nodes.lock().unwrap();
            match nodes.get(path) {
                Some(Node::Dir) => {}
                Some(Node::File(_)) => {
                    return Err(anyhow!("Not a directory (os error 20)"));
                }
                None => {
                    return Err(anyhow!(
                        "No such file or directory (os error 2)"
                    ));
                }
            }
            let mut entries: Vec<FsDirEntry> = nodes
                .iter()
                .filter(|(p, _)| p.parent() == Some(path))
                .map(|(p, node)| FsDirEntry {
                    name: p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    is_dir: matches!(node, Node::Dir),
                    is_file: matches!(node, Node::File(_)),
                })
                .collect();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(entries)
        }

        async fn stat(&self, path: &Path) -> Result<FsStat> {
            match self.nodes.lock().unwrap().get(path) {
                Some(Node::Dir) => Ok(FsStat { is_dir: true, is_file: false }),
                Some(Node::File(_)) => {
                    Ok(FsStat { is_dir: false, is_file: true })
                }
                None => Err(anyhow!("No such file or directory (os error 2)")),
            }
        }
    }
}
