//! Core-side tree walking shared by `grep` and `find`: depth-first with
//! entries sorted by name (deterministic across backends), honoring
//! `.gitignore` files within the tree (nested files override shallower ones,
//! like git) and skipping `.git`. All walking logic lives here; every
//! filesystem touch — stat, list, read — goes through the context's
//! [`FsOps`](super::fs::FsOps), so the walk behaves identically over the
//! local filesystem and a sandbox backend.

use std::path::{Path, PathBuf};

use futures::future::BoxFuture;
use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use super::fs::FsOps;

/// All regular files under `root`, as paths relative to it, in a stable
/// depth-first order. A `root` that is itself a file yields one empty
/// relative path (what stripping the root prefix from itself produces). An
/// unreadable root or subdirectory contributes nothing — like the previous
/// walker, which skipped errored entries silently.
pub(crate) async fn files(fs: &dyn FsOps, root: &Path) -> Vec<PathBuf> {
    let Ok(stat) = fs.stat(root).await else { return Vec::new() };
    if stat.is_file {
        return vec![PathBuf::new()];
    }
    if !stat.is_dir {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut matchers = Vec::new();
    walk_dir(fs, root.to_path_buf(), PathBuf::new(), &mut matchers, &mut out)
        .await;
    out
}

/// One directory level: layer its `.gitignore` (if any) over the ancestors',
/// then visit the surviving entries in name order, descending into
/// subdirectories and collecting files. Boxed because async recursion needs
/// a sized future.
fn walk_dir<'a>(
    fs: &'a dyn FsOps,
    abs: PathBuf,
    rel: PathBuf,
    matchers: &'a mut Vec<Gitignore>,
    out: &'a mut Vec<PathBuf>,
) -> BoxFuture<'a, ()> {
    Box::pin(async move {
        let Ok(mut entries) = fs.list_dir(&abs).await else { return };
        let mut layered = false;
        if entries.iter().any(|e| e.is_file && e.name == ".gitignore")
            && let Ok(bytes) = fs.read(&abs.join(".gitignore")).await
        {
            let mut builder = GitignoreBuilder::new(&abs);
            for line in String::from_utf8_lossy(&bytes).lines() {
                let _ = builder.add_line(None, line);
            }
            if let Ok(gitignore) = builder.build() {
                matchers.push(gitignore);
                layered = true;
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        for entry in entries {
            if entry.name == ".git" && entry.is_dir {
                continue;
            }
            let entry_abs = abs.join(&entry.name);
            if ignored(matchers, &entry_abs, entry.is_dir) {
                continue;
            }
            let entry_rel = rel.join(&entry.name);
            if entry.is_dir {
                walk_dir(fs, entry_abs, entry_rel, matchers, out).await;
            } else if entry.is_file {
                out.push(entry_rel);
            }
            // Neither (a symlink, a socket): skipped, as before.
        }
        if layered {
            matchers.pop();
        }
    })
}

/// Whether the deepest `.gitignore` with an opinion on `path` says ignore
/// (a deeper file overrides a shallower one, like git).
fn ignored(matchers: &[Gitignore], path: &Path, is_dir: bool) -> bool {
    for gitignore in matchers.iter().rev() {
        match gitignore.matched(path, is_dir) {
            Match::None => continue,
            Match::Ignore(_) => return true,
            Match::Whitelist(_) => return false,
        }
    }
    false
}
