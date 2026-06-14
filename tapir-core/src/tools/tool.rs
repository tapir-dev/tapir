//! The [`Tool`] trait and its per-call execution [`ToolCtx`].
//!
//! A `Tool` is the SDK's pluggable unit of capability: it carries its
//! model-facing definition (name, description, JSON-Schema parameters) and an
//! async `run`. Tools are registered on the Runtime and stored as
//! `Arc<dyn Tool>`, so the turn loop dispatches a call through the trait instead
//! of a name-matched static function.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::agent::CancelToken;

use super::exec::{ExecOps, LocalExecOps};
use super::fs::{FsOps, LocalFsOps};
use super::jail::PathBoundary;
use super::{Progress, ToolResult};

/// A session's opaque metadata — carried to every tool's execution context but
/// never interpreted by the engine (an adapter stashes its own keys here).
pub type Metadata = HashMap<String, String>;

/// A pluggable tool. Object-safe (`async-trait` boxes the future) so a
/// heterogeneous set can live in one registry.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's identifier, matched against a model's tool call.
    fn name(&self) -> &'static str;
    /// The model-facing description (sent in the request's `tools` array).
    fn description(&self) -> &'static str;
    /// The JSON-Schema for the tool's arguments.
    fn parameters(&self) -> Value;
    /// The transcript header for a call; defaults to the tool name.
    fn title(&self, _args: &Value) -> String {
        self.name().to_string()
    }
    /// Run the call to completion, streaming partials via [`ctx`](ToolCtx).
    async fn run(&self, args: &Value, ctx: &ToolCtx) -> Result<ToolResult>;
}

/// Per-call execution context: where the call runs, whether it has been
/// aborted, and a sink for streaming partial output.
pub struct ToolCtx {
    cwd: PathBuf,
    cancel: CancelToken,
    progress: Progress,
    metadata: Arc<Metadata>,
    exec: Arc<dyn ExecOps>,
    fs: Arc<dyn FsOps>,
    boundary: Option<PathBoundary>,
}

impl ToolCtx {
    pub fn new(cwd: PathBuf, cancel: CancelToken, progress: Progress) -> Self {
        Self {
            cwd,
            cancel,
            progress,
            metadata: Arc::new(Metadata::new()),
            exec: Arc::new(LocalExecOps),
            fs: Arc::new(LocalFsOps),
            boundary: None,
        }
    }

    /// Attach a workspace [`PathBoundary`]: from here on, every path this
    /// context resolves is translated guest→host and validated against it.
    /// Without one (the default), resolution is unrestricted — today's local
    /// behavior.
    pub fn with_boundary(mut self, boundary: PathBoundary) -> Self {
        self.boundary = Some(boundary);
        self
    }

    /// The workspace boundary, when one is configured.
    pub fn boundary(&self) -> Option<&PathBoundary> {
        self.boundary.as_ref()
    }

    /// Resolve a model-supplied path argument. Without a boundary this is
    /// the unrestricted resolution (absolute, `~/`, relative against the
    /// cwd). With one, the path is a guest path: absolute as-is, otherwise
    /// joined to the (guest) cwd — no `~` expansion, the guest has no host
    /// home — then translated guest→host and validated; an escape is an
    /// `Err` naming the path and the boundary.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        match &self.boundary {
            None => Ok(super::resolve(&self.cwd, path)),
            Some(boundary) => {
                let p = Path::new(path);
                let guest = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    self.cwd.join(p)
                };
                boundary.resolve(&guest)
            }
        }
    }

    /// Attach the session's metadata (the turn loop does this per call).
    pub fn with_metadata(mut self, metadata: Arc<Metadata>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach the session's exec operations (the turn loop does this per
    /// call). Defaults to [`LocalExecOps`] — the host spawn path.
    pub fn with_exec_ops(mut self, exec: Arc<dyn ExecOps>) -> Self {
        self.exec = exec;
        self
    }

    /// The exec operations `bash` performs its process execution through.
    pub fn exec_ops(&self) -> &dyn ExecOps {
        self.exec.as_ref()
    }

    /// Attach the session's filesystem operations (the turn loop does this
    /// per call). Defaults to [`LocalFsOps`] — the host `std::fs` path.
    pub fn with_fs_ops(mut self, fs: Arc<dyn FsOps>) -> Self {
        self.fs = fs;
        self
    }

    /// The filesystem operations the file tools perform their raw I/O through.
    pub fn fs_ops(&self) -> &dyn FsOps {
        self.fs.as_ref()
    }

    /// The working directory the call runs in. With a boundary configured
    /// this is the guest view (what the model addresses); the host directory
    /// behind it comes from [`host_cwd`](Self::host_cwd).
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// The working directory on the side that performs I/O: the cwd itself
    /// without a boundary; with one, the (guest) cwd translated to its host
    /// directory and validated against the boundary. `bash` execs here, and
    /// the directory-rooted tools (ls, grep, find) default to it.
    pub fn host_cwd(&self) -> Result<PathBuf> {
        match &self.boundary {
            None => Ok(self.cwd.clone()),
            Some(boundary) => boundary.resolve(&self.cwd),
        }
    }

    /// The session's opaque metadata map. (Read by an adapter's custom tools; no
    /// built-in tool needs it.)
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Whether the turn has been aborted; a long-running tool should bail out.
    pub fn is_aborted(&self) -> bool {
        self.cancel.is_aborted()
    }

    /// Stream a chunk of partial output (the turn loop forwards it as a
    /// [`ToolUpdate`](crate::agent::TurnEvent::ToolUpdate)). A no-op when the
    /// caller supplied no sink. `bash` streams its output through here.
    pub async fn update(&self, delta: impl Into<String>) {
        if let Some(tx) = &self.progress {
            let _ = tx.send(delta.into()).await;
        }
    }
}
