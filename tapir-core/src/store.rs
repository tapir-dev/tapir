//! Session persistence behind the [`SessionStore`] trait.
//!
//! A store is injected on the [`RuntimeBuilder`](crate::runtime::RuntimeBuilder)
//! and keyed by conversation identifier: append an [`Entry`], load a
//! conversation, list identifiers. The engine drives resume (and the
//! compaction checkpoint) through the trait, so every frontend inherits
//! context management instead of reimplementing it; an adapter may register a
//! database-backed store the same way.
//!
//! Two stores ship: [`FileStore`] preserves the existing on-disk JSONL
//! conversation format (it delegates to the same code in [`crate::session`]),
//! and [`InMemoryStore`] serves embedders and tests (the default).

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;
use async_trait::async_trait;

use crate::agent::Role;

/// One persisted conversation record — the engine's vocabulary for the entry
/// kinds the existing session format stores. Every frontend (TUI, headless
/// print, and adapters) persists conversations as these.
//
// File LIFECYCLE stays outside this vocabulary on purpose: creation with a
// parent link, fork/clone/import, and listing with metadata are operations on
// the on-disk layout (`crate::session`). Generalizing the trait to cover them
// waits for a store that isn't file-backed (the bot adapter).
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    /// A conversation message.
    Message { role: Role, text: String },
    /// An auto-compaction checkpoint (earlier context replaced by the summary).
    Compaction { summary: String },
    /// A `!`/`!!` shell command and its output.
    Bash {
        command: String,
        output: String,
        exit_code: Option<i32>,
        cancelled: bool,
        /// `!!` ran outside the model's context; its output is not replayed.
        exclude_from_context: bool,
    },
    /// An agent tool call and its result (display history; never model context).
    Tool {
        name: String,
        title: String,
        output: String,
        is_error: bool,
        took_ms: u64,
    },
    /// The conversation's user-set display name.
    Name(String),
}

/// Pluggable session persistence, keyed by conversation identifier. Async and
/// object-safe (`async-trait`) so a database-backed store slots in.
// Adapter-facing — see the note on `Entry`.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Append one entry to the conversation `id` (created on first append).
    async fn append(&self, id: &str, entry: &Entry) -> Result<()>;
    /// The conversation's entries, in append order.
    async fn load(&self, id: &str) -> Result<Vec<Entry>>;
    /// The stored conversation identifiers.
    async fn list(&self) -> Result<Vec<String>>;
}

/// An in-memory store: the engine tests' double and the default on a Runtime
/// (no surprise disk writes for embedders that never resume).
#[derive(Default)]
pub struct InMemoryStore {
    conversations: Mutex<HashMap<String, Vec<Entry>>>,
}

#[async_trait]
impl SessionStore for InMemoryStore {
    async fn append(&self, id: &str, entry: &Entry) -> Result<()> {
        self.conversations
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_default()
            .push(entry.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Vec<Entry>> {
        self.conversations
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no stored conversation: {id}"))
    }

    async fn list(&self) -> Result<Vec<String>> {
        Ok(self.conversations.lock().unwrap().keys().cloned().collect())
    }
}

/// A file-backed store over the existing on-disk conversation layout: one
/// JSONL file per conversation under a sessions directory, written and parsed
/// by the same code as [`crate::session`] — the format is unchanged, so files
/// written by either path read back through the other.
// Adapter-facing — see the note on `Entry`.
pub struct FileStore {
    /// The sessions directory conversations live in (one cwd's folder in the
    /// existing layout); where this store creates new conversations.
    dir: std::path::PathBuf,
    /// Recorded in the header of conversations this store creates.
    cwd: std::path::PathBuf,
    /// When set, lookups search this whole tree (the sessions root) instead
    /// of only `dir`, so ids from other working directories resolve too.
    root: Option<std::path::PathBuf>,
}

impl FileStore {
    /// A store over `dir`, stamping `cwd` into the header of new conversations
    /// (the existing layout keys the directory by working directory).
    pub fn new(
        dir: impl Into<std::path::PathBuf>,
        cwd: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self { dir: dir.into(), cwd: cwd.into(), root: None }
    }

    /// The production store over the existing sessions layout under a config
    /// dir: new conversations are created in `cwd`'s folder, and lookups span
    /// the whole root (the TUI's session picker switches into conversations
    /// started from other working directories). The TUI and headless print
    /// both inject exactly this.
    pub fn in_layout(
        config_dir: &std::path::Path,
        cwd: &std::path::Path,
    ) -> Self {
        Self::new(crate::session::dir_for(config_dir, cwd), cwd)
            .with_root(crate::session::sessions_root(config_dir))
    }

    /// Widen lookups to `root` (the sessions root holding every cwd's
    /// folder): `load`/`append` then resolve a conversation id wherever it
    /// lives — the picker's "all" scope switches into sessions started from
    /// another working directory — while new conversations are still created
    /// in this store's own dir.
    pub fn with_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.root = Some(root.into());
        self
    }

    /// The conversation file for `id`, if it exists (`<stamp>_<id>.jsonl`),
    /// searching the lookup root's tree when one is set.
    fn path_for(&self, id: &str) -> Option<std::path::PathBuf> {
        let suffix = format!("_{id}.jsonl");
        find_with_suffix(self.root.as_deref().unwrap_or(&self.dir), &suffix)
    }

    /// A write handle for `id` — the existing typed writer over the located
    /// (or freshly created) file, so the bytes match the existing format.
    fn writer(&self, id: &str) -> Result<crate::session::Session> {
        match self.path_for(id) {
            Some(path) => Ok(crate::session::Session {
                id: id.to_string(),
                name: None,
                parent: None,
                path,
            }),
            None => crate::session::Session::create_with_id(
                &self.dir,
                &self.cwd,
                None,
                id.to_string(),
            ),
        }
    }
}

/// Depth-first search for the first file under `dir` whose name ends in
/// `suffix`. The sessions tree is small (one folder per cwd); conversation
/// ids are unique across it, so the first match is the conversation.
fn find_with_suffix(
    dir: &std::path::Path,
    suffix: &str,
) -> Option<std::path::PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_with_suffix(&path, suffix) {
                return Some(found);
            }
        } else if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(suffix))
        {
            return Some(path);
        }
    }
    None
}

#[async_trait]
impl SessionStore for FileStore {
    async fn append(&self, id: &str, entry: &Entry) -> Result<()> {
        let mut writer = self.writer(id)?;
        match entry {
            Entry::Message { role, text } => writer.append_message(*role, text),
            Entry::Compaction { summary } => writer.append_compaction(summary),
            Entry::Bash {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            } => writer.append_bash(
                command,
                output,
                *exit_code,
                *cancelled,
                *exclude_from_context,
            ),
            Entry::Tool { name, title, output, is_error, took_ms } => {
                writer.append_tool(name, title, output, *is_error, *took_ms)
            }
            Entry::Name(name) => writer.set_name(name),
        }
    }

    async fn load(&self, id: &str) -> Result<Vec<Entry>> {
        let path = self
            .path_for(id)
            .ok_or_else(|| anyhow::anyhow!("no stored conversation: {id}"))?;
        Ok(entries(crate::session::load(&path)?))
    }

    async fn list(&self) -> Result<Vec<String>> {
        Ok(crate::session::list_dir(&self.dir)
            .into_iter()
            .map(|m| m.id)
            .collect())
    }
}

/// A loaded conversation's records in the store's vocabulary — the single
/// translation from the on-disk loader. The display name is collapsed by the
/// loader (last one wins) and surfaces as a trailing [`Entry::Name`], so a
/// consumer sees the effective name. [`FileStore::load`] and a frontend
/// resuming from a file it already located both read through this.
pub fn entries(loaded: crate::session::Loaded) -> Vec<Entry> {
    let mut entries: Vec<Entry> = loaded
        .items
        .into_iter()
        .map(|item| match item {
            crate::session::LoadedItem::Message { role, text } => {
                Entry::Message { role, text }
            }
            crate::session::LoadedItem::Compaction { summary } => {
                Entry::Compaction { summary }
            }
            crate::session::LoadedItem::Bash {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            } => Entry::Bash {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            },
            crate::session::LoadedItem::Tool {
                name,
                title,
                output,
                is_error,
                took_ms,
            } => Entry::Tool { name, title, output, is_error, took_ms },
        })
        .collect();
    if let Some(name) = loaded.name {
        entries.push(Entry::Name(name));
    }
    entries
}

/// Replay stored entries into a session's durable history — the engine half of
/// resume, mirroring the TUI's live replay: messages re-enter the conversation,
/// a compaction checkpoint replaces what came before it, a context-included
/// shell record re-enters as user context, and a cancelled tool call leaves the
/// interruption marker. Tool calls and `!!` records are display history only.
pub fn replay(agent: &mut crate::agent::Agent, entries: &[Entry]) {
    for entry in entries {
        match entry {
            Entry::Message { role: Role::User, text } => {
                agent.submit(crate::agent::Input {
                    display: String::new(),
                    model_text: text.clone(),
                    images: Vec::new(),
                });
            }
            Entry::Message { role: Role::Assistant, text } => {
                agent.push_assistant(text)
            }
            Entry::Compaction { summary } => agent.compact(summary),
            Entry::Bash { exclude_from_context: false, .. } => {
                agent.submit(crate::agent::Input {
                    display: String::new(),
                    model_text: bash_context_text(entry),
                    images: Vec::new(),
                });
            }
            Entry::Bash { .. } => {}
            Entry::Tool { is_error: true, output, .. }
                if output == "cancelled" =>
            {
                agent.interrupted();
            }
            Entry::Tool { .. } | Entry::Name(_) => {}
        }
    }
}

/// How a shell record reads when fed to the model as context — used by
/// [`replay`] on resume, and exposed so a frontend's live `!` path submits
/// the same text instead of mirroring the format.
pub fn bash_context_text(entry: &Entry) -> String {
    let Entry::Bash { command, output, exit_code, cancelled, .. } = entry
    else {
        return String::new();
    };
    let mut text = format!("Ran `{command}`\n");
    if output.trim().is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str(&format!("```\n{}\n```", output.trim_end()));
    }
    if *cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = exit_code
        && *code != 0
    {
        text.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(label: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir()
            .join(format!("tapir-store-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[tokio::test]
    async fn file_store_round_trips_the_existing_format() {
        let dir = tmp("fmt");
        let cwd = std::path::Path::new("/home/u/proj");

        // Existing writer → trait reader: a conversation file written by
        // today's session code is loadable through the trait by its id.
        let s = crate::session::Session::create(&dir, cwd, None).unwrap();
        s.append_message(Role::User, "hello").unwrap();
        s.append_message(Role::Assistant, "hi there").unwrap();

        let store = FileStore::new(&dir, cwd);
        let entries =
            store.load(&s.id).await.expect("the existing file loads by id");
        assert_eq!(
            entries,
            vec![
                Entry::Message { role: Role::User, text: "hello".into() },
                Entry::Message {
                    role: Role::Assistant,
                    text: "hi there".into()
                },
            ],
            "the existing format reads back through the trait",
        );

        // Trait writer → existing reader: entries appended through the trait
        // parse with the existing loader — the on-disk format is unchanged.
        store
            .append(
                "conv-x",
                &Entry::Message { role: Role::User, text: "via trait".into() },
            )
            .await
            .expect("first append creates the conversation");
        let path = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with("_conv-x.jsonl"))
            })
            .expect("the trait append created a session file named like the existing layout");
        let loaded =
            crate::session::load(&path).expect("the existing loader parses it");
        assert_eq!(loaded.id, "conv-x");
        assert!(
            matches!(&loaded.items[0], crate::session::LoadedItem::Message { text, .. } if text == "via trait"),
            "the existing loader reads what the trait wrote",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn resume_applies_the_compaction_checkpoint() {
        use crate::providers::Step;
        // A stored conversation with a compaction checkpoint: everything before
        // it is replaced by the summary; what follows accumulates after it —
        // exactly the live session's semantics, inherited through the trait.
        let store = std::sync::Arc::new(InMemoryStore::default());
        for entry in [
            Entry::Message { role: Role::User, text: "old question".into() },
            Entry::Message { role: Role::Assistant, text: "old answer".into() },
            Entry::Compaction { summary: "SUMMARY".into() },
            Entry::Message { role: Role::User, text: "new question".into() },
        ] {
            store.append("conv-c", &entry).await.unwrap();
        }

        let rt = crate::runtime::Runtime::builder().store(store).build();
        let agent = rt
            .resume(crate::runtime::SessionOptions::new(".".into()), "conv-c")
            .await
            .expect("resumes");

        // The checkpoint replaced the earlier turns; the follow-up user message
        // coalesces after the summary (user steps merge, as live).
        assert_eq!(
            agent.history().len(),
            1,
            "compaction left a single user step"
        );
        let history = agent.history();
        let Some(Step::User { text, .. }) = history.first() else {
            panic!("the checkpoint is a user step");
        };
        assert!(text.contains("SUMMARY"), "the summary is the context");
        assert!(
            text.contains("new question"),
            "later turns follow the checkpoint"
        );
        assert!(
            !text.contains("old question"),
            "pre-checkpoint turns are gone"
        );
    }

    #[tokio::test]
    async fn resume_replays_shell_context_and_interruption_markers() {
        use crate::providers::Step;
        // A `!` record re-enters the model context; a `!!` (excluded) record and
        // a finished tool call do not; a cancelled tool call leaves the
        // interruption marker — mirroring the TUI's live replay.
        let store = std::sync::Arc::new(InMemoryStore::default());
        for entry in [
            Entry::Message { role: Role::User, text: "q".into() },
            Entry::Message { role: Role::Assistant, text: "a".into() },
            Entry::Bash {
                command: "ls".into(),
                output: "a.txt".into(),
                exit_code: Some(0),
                cancelled: false,
                exclude_from_context: false,
            },
            Entry::Bash {
                command: "secret".into(),
                output: "hidden".into(),
                exit_code: Some(0),
                cancelled: false,
                exclude_from_context: true,
            },
            Entry::Tool {
                name: "read".into(),
                title: "read f".into(),
                output: "cancelled".into(),
                is_error: true,
                took_ms: 1,
            },
        ] {
            store.append("conv-b", &entry).await.unwrap();
        }

        let rt = crate::runtime::Runtime::builder().store(store).build();
        let agent = rt
            .resume(crate::runtime::SessionOptions::new(".".into()), "conv-b")
            .await
            .expect("resumes");

        let all: String = agent
            .history()
            .iter()
            .map(|s| match s {
                Step::User { text, .. } | Step::Assistant { text, .. } => {
                    text.clone()
                }
                Step::ToolResult { output, .. } => output.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n---\n");
        assert!(
            all.contains("Ran `ls`"),
            "the `!` record re-entered the context"
        );
        assert!(!all.contains("hidden"), "the excluded `!!` record stayed out");
        assert!(
            all.contains(crate::agent::INTERRUPTED),
            "the cancelled tool left the interruption marker",
        );
    }

    #[tokio::test]
    async fn both_stores_list_their_conversation_identifiers() {
        // In-memory…
        let mem = InMemoryStore::default();
        mem.append("a", &Entry::Message { role: Role::User, text: "x".into() })
            .await
            .unwrap();
        mem.append("b", &Entry::Message { role: Role::User, text: "y".into() })
            .await
            .unwrap();
        let mut ids = mem.list().await.unwrap();
        ids.sort();
        assert_eq!(ids, ["a", "b"]);

        // …and file-backed, over the existing layout.
        let dir = tmp("list");
        let cwd = std::path::Path::new("/home/u/proj");
        let file = FileStore::new(&dir, cwd);
        file.append(
            "c1",
            &Entry::Message { role: Role::User, text: "x".into() },
        )
        .await
        .unwrap();
        file.append(
            "c2",
            &Entry::Message { role: Role::User, text: "y".into() },
        )
        .await
        .unwrap();
        let mut ids = file.list().await.unwrap();
        ids.sort();
        assert_eq!(ids, ["c1", "c2"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_rooted_store_finds_a_conversation_in_a_sibling_cwd_dir() {
        // The sessions layout keeps one folder per cwd under a common root.
        // A store told about that root resolves ids across cwd folders — the
        // session picker's "all" scope switches into conversations started
        // elsewhere — instead of only its own dir.
        let root = tmp("root-load");
        let dir_a = root.join("proj-a");
        let dir_b = root.join("proj-b");

        let s = crate::session::Session::create(
            &dir_a,
            std::path::Path::new("/home/u/proj-a"),
            None,
        )
        .unwrap();
        s.append_message(Role::User, "from another cwd").unwrap();

        let store =
            FileStore::new(&dir_b, std::path::Path::new("/home/u/proj-b"))
                .with_root(&root);
        let entries = store
            .load(&s.id)
            .await
            .expect("the sibling-dir conversation loads by id");
        assert_eq!(
            entries,
            vec![Entry::Message {
                role: Role::User,
                text: "from another cwd".into()
            }],
            "lookup spans the sessions root, not just the store's own dir",
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_rooted_store_appends_to_the_conversation_where_it_lives() {
        // Appending to a cross-dir conversation must reach its file — a
        // duplicate created in the store's own dir would fork the history.
        let root = tmp("root-append");
        let dir_a = root.join("proj-a");
        let dir_b = root.join("proj-b");

        let s = crate::session::Session::create(
            &dir_a,
            std::path::Path::new("/home/u/proj-a"),
            None,
        )
        .unwrap();
        s.append_message(Role::User, "question").unwrap();

        let store =
            FileStore::new(&dir_b, std::path::Path::new("/home/u/proj-b"))
                .with_root(&root);
        store
            .append(
                &s.id,
                &Entry::Message {
                    role: Role::Assistant,
                    text: "answer".into(),
                },
            )
            .await
            .expect("the cross-dir conversation accepts appends");

        let loaded = crate::session::load(&s.path)
            .expect("the original file still parses");
        assert!(
            matches!(
                loaded.items.last(),
                Some(crate::session::LoadedItem::Message { text, .. }) if text == "answer"
            ),
            "the append landed in the conversation's own file",
        );
        assert_eq!(
            std::fs::read_dir(&dir_b).map(|d| d.flatten().count()).unwrap_or(0),
            0,
            "no duplicate conversation was created in the store's own dir",
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_rooted_store_creates_unknown_conversations_in_its_own_dir() {
        // The widened lookup must not change where new conversations are
        // born: an unknown id is created in the store's dir (the cwd's
        // folder), and resolves back through the rooted lookup.
        let root = tmp("root-create");
        let dir_b = root.join("proj-b");

        let store =
            FileStore::new(&dir_b, std::path::Path::new("/home/u/proj-b"))
                .with_root(&root);
        store
            .append(
                "conv-new",
                &Entry::Message { role: Role::User, text: "first".into() },
            )
            .await
            .expect("first append creates the conversation");

        assert!(
            std::fs::read_dir(&dir_b).unwrap().flatten().any(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.ends_with("_conv-new.jsonl"))
            }),
            "the new conversation was created in the store's own dir",
        );
        let entries = store
            .load("conv-new")
            .await
            .expect("the new conversation loads back");
        assert_eq!(
            entries,
            vec![Entry::Message { role: Role::User, text: "first".into() }]
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_file_stored_conversation_resumes_through_the_runtime() {
        // The criterion round-trip: append to the file store, reload, resume.
        let dir = tmp("resume");
        let cwd = std::path::Path::new("/home/u/proj");
        let store = std::sync::Arc::new(FileStore::new(&dir, cwd));

        store
            .append(
                "conv-r",
                &Entry::Message { role: Role::User, text: "question".into() },
            )
            .await
            .unwrap();
        store
            .append(
                "conv-r",
                &Entry::Message {
                    role: Role::Assistant,
                    text: "answer".into(),
                },
            )
            .await
            .unwrap();

        let rt = crate::runtime::Runtime::builder().store(store).build();
        let agent = rt
            .resume(crate::runtime::SessionOptions::new(".".into()), "conv-r")
            .await
            .expect("the file-stored conversation resumes");
        assert_eq!(
            agent.history().len(),
            2,
            "both stored messages re-enter the history"
        );
        assert!(
            matches!(agent.history().last(), Some(crate::providers::Step::Assistant { text, .. }) if text == "answer"),
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
