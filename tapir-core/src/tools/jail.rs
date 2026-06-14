//! The path jail: an optional workspace boundary carried by the tool context.
//!
//! A [`PathBoundary`] maps a guest root (what the model sees, e.g.
//! `/workspace`) to a host root (where the files actually live, e.g. a
//! bind-mounted channel workspace). When a context carries one, every path a
//! tool resolves is translated guest→host and validated before any
//! filesystem operation; a path that escapes fails with an error naming the
//! offending path and the boundary, so the model can correct itself. Without
//! a boundary, resolution is unrestricted — exactly today's local behavior.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

/// An optional workspace boundary: a guest root the model addresses mapped
/// to a host root where the files live. Carried by the tool context; built
/// by a sandbox backend (e.g. `/workspace` → the channel's host workspace).
#[derive(Clone, Debug)]
pub struct PathBoundary {
    guest_root: PathBuf,
    host_root: PathBuf,
}

impl PathBoundary {
    /// A boundary mapping `guest_root` (the absolute path the model sees) to
    /// `host_root` (the absolute host directory backing it).
    pub fn new(
        guest_root: impl Into<PathBuf>,
        host_root: impl Into<PathBuf>,
    ) -> Self {
        Self { guest_root: guest_root.into(), host_root: host_root.into() }
    }

    /// The root the model addresses (e.g. `/workspace`).
    pub fn guest_root(&self) -> &Path {
        &self.guest_root
    }

    /// The host directory backing the guest root.
    pub fn host_root(&self) -> &Path {
        &self.host_root
    }

    /// Translate an absolute guest path to its host path, validating that it
    /// cannot escape the boundary: the real path of its closest existing
    /// ancestor (symlinks followed, like `realpath`) must land inside the
    /// host root, and the not-yet-existing tail must be plain names — so an
    /// absolute path outside the boundary, a `..` traversal, and a symlink
    /// pointing out are all rejected. Validation runs against the host
    /// filesystem: by design the boundary lives where the host root does
    /// (e.g. a bind-mounted workspace).
    pub fn resolve(&self, guest: &Path) -> Result<PathBuf> {
        let rest = guest
            .strip_prefix(&self.guest_root)
            .map_err(|_| self.deny(guest))?;
        let host = self.host_root.join(rest);
        let root = std::fs::canonicalize(&self.host_root).map_err(|e| {
            anyhow!(
                "workspace boundary root {} is unusable: {e}",
                self.host_root.display()
            )
        })?;

        // Walk up to the closest existing ancestor, collecting the tail that
        // doesn't exist yet (a file about to be written, a dir about to be
        // made). A tail component that isn't a plain name (`..`, `.`) can't
        // be checked against reality — reject it.
        let mut dir = host.clone();
        let mut tail = Vec::new();
        let real = loop {
            match std::fs::canonicalize(&dir) {
                Ok(real) => break real,
                Err(_) => {
                    // The path exists but doesn't canonicalize: a dangling
                    // symlink. Writing through it would land on its target,
                    // which realpath can't vouch for — reject.
                    if std::fs::symlink_metadata(&dir).is_ok() {
                        return Err(self.deny(guest));
                    }
                    let Some(name) = dir.file_name() else {
                        return Err(self.deny(guest));
                    };
                    tail.push(name.to_os_string());
                    let Some(parent) = dir.parent() else {
                        return Err(self.deny(guest));
                    };
                    dir = parent.to_path_buf();
                }
            }
        };
        if !real.starts_with(&root) {
            return Err(self.deny(guest));
        }
        Ok(tail.into_iter().rev().fold(real, |p, name| p.join(name)))
    }

    /// The rejection: names the offending path and the boundary, so the
    /// model can correct itself.
    fn deny(&self, guest: &Path) -> anyhow::Error {
        anyhow!(
            "{} is outside the workspace boundary {}: only paths under {} are accessible",
            guest.display(),
            self.guest_root.display(),
            self.guest_root.display(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use serde_json::json;

    use crate::agent::CancelToken;
    use crate::tools::read::ReadTool;
    use crate::tools::tool::{Tool, ToolCtx};
    use crate::tools::write::WriteTool;

    use super::PathBoundary;

    /// A fresh host workspace under the temp dir, plus a context whose cwd is
    /// the guest root `/jail` and whose boundary maps it to that workspace.
    fn jailed(label: &str) -> (PathBuf, ToolCtx) {
        let host = std::env::temp_dir()
            .join(format!("tapir-jail-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&host);
        std::fs::create_dir_all(&host).unwrap();
        let ctx = ToolCtx::new("/jail".into(), CancelToken::default(), None)
            .with_boundary(PathBoundary::new("/jail", &host));
        (host, ctx)
    }

    #[tokio::test]
    async fn read_inside_the_boundary_is_translated_and_an_absolute_escape_rejected()
     {
        let (host, ctx) = jailed("read");
        std::fs::write(host.join("inside.txt"), "guarded content").unwrap();

        // A guest path inside the boundary reaches the host file backing it.
        let out = ReadTool
            .run(&json!({ "path": "/jail/inside.txt" }), &ctx)
            .await
            .unwrap();
        assert_eq!(
            out.model_text, "guarded content",
            "guest path served from the host root"
        );

        // An absolute path outside the boundary is rejected, not read.
        let err = ReadTool
            .run(&json!({ "path": "/etc/passwd" }), &ctx)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("/etc/passwd"),
            "the error names the offending path: {err}"
        );

        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn a_dot_dot_traversal_escaping_the_boundary_is_rejected() {
        let (host, ctx) = jailed("dotdot");
        // A real file one level above the host root: reachable lexically
        // through `..`, but outside the boundary.
        let outside =
            host.parent().unwrap().join("tapir-jail-dotdot-outside.txt");
        std::fs::write(&outside, "secret").unwrap();

        for path in [
            "/jail/../tapir-jail-dotdot-outside.txt",
            "../tapir-jail-dotdot-outside.txt",
        ] {
            let err =
                ReadTool.run(&json!({ "path": path }), &ctx).await.unwrap_err();
            assert!(
                !err.to_string().contains("secret")
                    && err.to_string().contains("boundary"),
                "traversal via {path} must be rejected, got: {err}"
            );
        }

        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn a_symlink_pointing_outside_the_boundary_is_rejected() {
        let (host, ctx) = jailed("symlink-out");
        // A real escape on disk: a directory outside the workspace holding a
        // secret, and a symlink inside the workspace pointing at it.
        let outside = std::env::temp_dir()
            .join(format!("tapir-jail-symlink-outside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("s.txt"), "leaked bytes").unwrap();
        std::os::unix::fs::symlink(&outside, host.join("link")).unwrap();
        std::os::unix::fs::symlink(
            outside.join("s.txt"),
            host.join("file-link"),
        )
        .unwrap();

        // Through a symlinked directory, and through a symlinked file: both
        // look like guest paths inside the workspace, both really point out.
        for path in ["/jail/link/s.txt", "/jail/file-link"] {
            let err =
                ReadTool.run(&json!({ "path": path }), &ctx).await.unwrap_err();
            assert!(
                !err.to_string().contains("leaked bytes")
                    && err.to_string().contains("boundary"),
                "symlink escape via {path} must be rejected, got: {err}"
            );
        }

        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn write_creates_new_paths_inside_the_boundary_and_rejects_outside_ones()
     {
        let (host, ctx) = jailed("write");

        // A brand-new nested path: nothing past the host root exists yet, so
        // validation is realpath-of-ancestor plus a plain-name tail. The
        // mkdir and the write both land under the host root.
        let out = WriteTool
            .run(&json!({ "path": "/jail/sub/deep/new.txt", "content": "made it" }), &ctx)
            .await
            .unwrap();
        assert!(out.model_text.contains("7 bytes"));
        assert_eq!(
            std::fs::read_to_string(host.join("sub/deep/new.txt")).unwrap(),
            "made it",
            "the guest write landed inside the host root"
        );

        // A new path outside the boundary is rejected before any I/O.
        let outside = std::env::temp_dir().join("tapir-jail-write-escape.txt");
        let _ = std::fs::remove_file(&outside);
        let err = WriteTool
            .run(&json!({ "path": outside.to_str().unwrap(), "content": "nope" }), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("boundary"), "rejected: {err}");
        assert!(!outside.exists(), "nothing was written outside the boundary");

        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn writing_through_a_dangling_symlink_cannot_escape_the_boundary() {
        let (host, ctx) = jailed("dangling");
        // The sneaky shape: the link's target doesn't exist, so the path
        // doesn't canonicalize — but writing "the link" would create the
        // target, outside the boundary.
        let target =
            std::env::temp_dir().join("tapir-jail-dangling-target.txt");
        let _ = std::fs::remove_file(&target);
        std::os::unix::fs::symlink(&target, host.join("trap")).unwrap();

        let err = WriteTool
            .run(&json!({ "path": "/jail/trap", "content": "nope" }), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("boundary"), "rejected: {err}");
        assert!(
            !target.exists(),
            "the dangling link's outside target was never created"
        );

        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn the_rejection_names_the_offending_path_and_the_boundary() {
        // The error is the model's feedback loop: it must see which path was
        // refused and what the boundary is, so it can correct itself.
        let (host, ctx) = jailed("message");
        let err = ReadTool
            .run(&json!({ "path": "/etc/shadow" }), &ctx)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("/etc/shadow"), "names the offending path: {msg}");
        assert!(
            msg.contains("/jail"),
            "states the boundary the model must stay under: {msg}"
        );
        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn bash_cwd_resolution_honors_the_boundary() {
        use crate::tools::bash::BashTool;
        use crate::tools::exec::testing::FakeExecOps;

        // The guest cwd is translated and validated before reaching the exec
        // ops: with the boundary set, the cwd handed across the seam is the
        // host directory backing the guest one.
        let (host, _) = jailed("bash");
        std::fs::create_dir_all(host.join("sub")).unwrap();
        let fake = Arc::new(FakeExecOps::ok("ok\n", 0));
        let ctx =
            ToolCtx::new("/jail/sub".into(), CancelToken::default(), None)
                .with_boundary(PathBoundary::new("/jail", &host))
                .with_exec_ops(fake.clone());
        BashTool.run(&json!({ "command": "pwd" }), &ctx).await.unwrap();
        assert_eq!(
            fake.calls()[0].cwd,
            std::fs::canonicalize(host.join("sub")).unwrap(),
            "the exec ops received the validated host cwd"
        );

        // A guest cwd that escapes never reaches the ops.
        let ctx =
            ToolCtx::new("/jail/../etc".into(), CancelToken::default(), None)
                .with_boundary(PathBoundary::new("/jail", &host))
                .with_exec_ops(fake.clone());
        let err =
            BashTool.run(&json!({ "command": "pwd" }), &ctx).await.unwrap_err();
        assert!(err.to_string().contains("boundary"), "rejected: {err}");
        assert_eq!(fake.calls().len(), 1, "no exec ran with an escaping cwd");

        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn every_file_builtin_honors_the_boundary() {
        use crate::tools::edit::EditTool;
        use crate::tools::find::FindTool;
        use crate::tools::grep::GrepTool;
        use crate::tools::ls::LsTool;

        let (host, ctx) = jailed("builtins");
        std::fs::write(host.join("inside.txt"), "needle here").unwrap();

        // Inside, each works through the translated root — including the
        // ls/grep/find default (no path argument): the guest cwd.
        let out = LsTool.run(&json!({}), &ctx).await.unwrap();
        assert_eq!(
            out.model_text, "inside.txt",
            "ls defaults to the translated cwd"
        );
        let out =
            GrepTool.run(&json!({ "pattern": "needle" }), &ctx).await.unwrap();
        assert!(
            out.model_text.contains("inside.txt:1"),
            "grep walks the translated cwd"
        );
        let out =
            FindTool.run(&json!({ "pattern": "*.txt" }), &ctx).await.unwrap();
        assert_eq!(
            out.model_text, "inside.txt",
            "find walks the translated cwd"
        );
        EditTool
            .run(
                &json!({ "path": "/jail/inside.txt",
                         "edits": [{ "oldText": "needle", "newText": "thread" }] }),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(host.join("inside.txt")).unwrap(),
            "thread here"
        );

        // Outside, each is rejected with the boundary error.
        let edit_args = json!({ "path": "/etc/hostname",
                                "edits": [{ "oldText": "a", "newText": "b" }] });
        let cases: [(&dyn Tool, &serde_json::Value); 4] = [
            (&EditTool, &edit_args),
            (&LsTool, &json!({ "path": "/etc" })),
            (&GrepTool, &json!({ "pattern": "root", "path": "/etc" })),
            (&FindTool, &json!({ "pattern": "*", "path": "/etc" })),
        ];
        for (tool, args) in cases {
            let err = tool.run(args, &ctx).await.unwrap_err();
            assert!(
                err.to_string().contains("boundary"),
                "{} must reject an outside path, got: {err}",
                tool.name()
            );
        }

        let _ = std::fs::remove_dir_all(&host);
    }

    #[tokio::test]
    async fn a_symlink_staying_inside_the_boundary_works_normally() {
        let (host, ctx) = jailed("symlink-in");
        std::fs::create_dir_all(host.join("real")).unwrap();
        std::fs::write(host.join("real/data.txt"), "still inside").unwrap();
        std::os::unix::fs::symlink(host.join("real"), host.join("alias"))
            .unwrap();

        let out = ReadTool
            .run(&json!({ "path": "/jail/alias/data.txt" }), &ctx)
            .await
            .unwrap();
        assert_eq!(
            out.model_text, "still inside",
            "an inside symlink resolves and serves"
        );

        let _ = std::fs::remove_dir_all(&host);
    }
}
