// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The reference `store-file` backend: durable, crash-safe append-only JSONL
//! persistence.
//!
//! One [`FileSessionStore`] is one session log: an append-only file of
//! externally-tagged [`AgentMessage`]s, one per LF-terminated line, no header.
//! The store only ever appends; branch and rewind stay above the seam.
//!
//! # Durability
//!
//! Writes go through an `O_APPEND` handle so every append lands atomically at
//! the end regardless of concurrent writers, and each append is followed by an
//! `fsync` before it resolves `Ok` — the [`SessionStore`] durable-on-await
//! contract. At [`open`](FileSessionStore::open) the parent directory is
//! `fsync`ed once so the newly created log's directory entry is itself durable.
//!
//! # Exclusivity
//!
//! Opening takes an exclusive advisory lock (`flock`) on the log and holds it
//! for the store's lifetime. A second opener on a contended log fails closed
//! rather than interleaving writes into the same session.
//!
//! # Permissions
//!
//! The log is created `0o600` and any parent directories `0o700`: a session
//! transcript is private to its owner.

use std::io::{self, ErrorKind, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fs4::FileExt;
use fs4::TryLockError;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::message::AgentMessage;
use crate::store::{SessionError, SessionStore};

/// A durable, crash-safe, append-only JSONL [`SessionStore`] backend.
///
/// Open one with [`open`](Self::open); it holds an exclusive `flock` on the log
/// for as long as the store lives. Generic over the custom-message type `M`,
/// defaulting to [`NoCustom`](crate::message::NoCustom).
pub struct FileSessionStore<M = crate::message::NoCustom> {
    path: PathBuf,
    /// The `O_APPEND` write handle, holding the exclusive advisory lock. Guarded
    /// so appends serialize; the lock releases when the handle drops. `Arc` so an
    /// append can move a clone onto a blocking thread.
    handle: Arc<Mutex<std::fs::File>>,
    _marker: PhantomData<fn() -> M>,
}

/// Wrap any displayable error as a [`SessionError`]. `SessionError` carries only
/// an operator message, so every failure the backend surfaces — I/O, a join
/// panic, serde — funnels through here.
fn session_err(error: impl ToString) -> SessionError {
    SessionError::new(error.to_string())
}

impl<M> FileSessionStore<M> {
    /// Open (creating if absent) the session log at `path`.
    ///
    /// Creates any missing parent directories (`0o700`), creates or reopens the
    /// log (`0o600`) for appending, takes an exclusive advisory lock, and
    /// `fsync`s the parent directory so the log entry is durable. Fails closed
    /// with a [`SessionError`] if the log is already locked by another writer.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, SessionError> {
        let path = path.into();
        let handle = {
            let path = path.clone();
            tokio::task::spawn_blocking(move || open_locked(&path))
                .await
                .map_err(session_err)?
                .map_err(session_err)?
        };
        Ok(Self {
            path,
            handle: Arc::new(Mutex::new(handle)),
            _marker: PhantomData,
        })
    }
}

#[async_trait]
impl<M> SessionStore<M> for FileSessionStore<M>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn append(
        &self,
        message: &AgentMessage<M>,
    ) -> Result<(), SessionError> {
        // Serialize on the async side; `to_string` is compact single-line JSON,
        // so the record occupies exactly one LF-terminated line.
        let mut line = serde_json::to_string(message).map_err(session_err)?;
        line.push('\n');

        let handle = self.handle.clone();
        // Write and fsync under the lock on a blocking thread; never await while
        // holding the guard.
        tokio::task::spawn_blocking(move || -> Result<(), SessionError> {
            let mut file = handle.lock().unwrap();
            file.write_all(line.as_bytes())
                .and_then(|()| file.sync_all())
                .map_err(session_err)
        })
        .await
        .map_err(session_err)?
    }

    async fn load(&self) -> Result<Vec<AgentMessage<M>>, SessionError> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || load_path::<M>(&path))
            .await
            .map_err(session_err)?
    }
}

/// Create dirs, open the log for appending, take the exclusive lock, and
/// `fsync` the parent directory. All the blocking, `unix`-flavored open work.
fn open_locked(path: &Path) -> io::Result<std::fs::File> {
    use std::fs::{DirBuilder, OpenOptions};

    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());

    if let Some(parent) = parent {
        let mut builder = DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(parent)?;
    }

    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(path)?;

    // Exclusive advisory lock, non-blocking: a contended log fails closed rather
    // than blocking or interleaving a second writer.
    match FileExt::try_lock(&file) {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            return Err(io::Error::new(
                ErrorKind::WouldBlock,
                "session log is locked by another writer",
            ));
        }
        Err(e) => return Err(e.into()),
    }

    // fsync the parent directory so the log's directory entry survives a crash.
    let dir = parent.unwrap_or_else(|| Path::new("."));
    std::fs::File::open(dir)?.sync_all()?;

    Ok(file)
}

/// Read the whole log and parse it back into history.
///
/// Missing or empty → `Ok(vec![])`. An unterminated trailing line is an
/// uncommitted torn write and is dropped. Any other unparseable or unknown-tag
/// line is committed corruption and fails closed with the log left intact.
fn load_path<M>(path: &Path) -> Result<Vec<AgentMessage<M>>, SessionError>
where
    M: DeserializeOwned,
{
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(session_err(e)),
    };
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    // Split on LF. `split` always yields a trailing element after the final
    // separator: on a well-formed log that is the empty tail after the last
    // `\n`; on a torn log it is the uncommitted partial line. Dropping the last
    // element handles both — a committed line always ends in `\n`, so anything
    // after the final `\n` is never durable.
    let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    lines.pop();

    lines
        .into_iter()
        .map(|line| {
            serde_json::from_slice(line).map_err(|e| {
                SessionError::new(format!("corrupt session log line: {e}"))
            })
        })
        .collect()
}
