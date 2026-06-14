//! Session persistence.
//!
//! Each session is a JSONL file under `<config>/sessions/--<cwd>--/`, named
//! `<sortable-timestamp>_<id>.jsonl`. The first line is a header
//! (`{"type":"session", id, timestamp, cwd, parentSession?}`); the rest are
//! `message` entries (`{type, id, parentId, timestamp, message}`) and
//! `session_info` entries (the user-set `name`). The cwd is encoded into the
//! directory name so the on-disk layout is stable.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::Role;

const VERSION: u32 = 3;

/// A piece of a message's content (stored as an array; we only emit text).
#[derive(Serialize, Deserialize)]
struct TextContent {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

/// One item recovered from a session, in file order.
pub enum LoadedItem {
    Message {
        role: Role,
        text: String,
    },
    /// An auto-compaction checkpoint (older items are replaced by this summary).
    Compaction {
        summary: String,
    },
    /// A `!`/`!!` shell command and its output.
    Bash {
        command: String,
        output: String,
        exit_code: Option<i32>,
        cancelled: bool,
        exclude_from_context: bool,
    },
    /// An agent tool call and its result.
    Tool {
        name: String,
        title: String,
        output: String,
        is_error: bool,
        took_ms: u64,
    },
}

/// A session loaded from disk.
pub struct Loaded {
    pub id: String,
    pub name: Option<String>,
    pub parent: Option<String>,
    pub items: Vec<LoadedItem>,
}

/// Summary of a session file, for `/session` and `/tree`.
#[derive(Clone)]
pub struct Meta {
    pub path: PathBuf,
    pub id: String,
    pub name: Option<String>,
    pub timestamp: String,
    pub parent: Option<String>,
    /// First user message, for a preview.
    pub preview: String,
    /// Number of conversation messages.
    pub messages: usize,
}

impl Meta {
    /// A label for the picker: the name, or the preview, or the id.
    pub fn label(&self) -> String {
        self.name
            .clone()
            .filter(|n| !n.is_empty())
            .or_else(|| {
                (!self.preview.is_empty()).then(|| self.preview.clone())
            })
            .unwrap_or_else(|| self.id.clone())
    }
}

/// The active session: where it's stored and its identity.
pub struct Session {
    pub id: String,
    pub name: Option<String>,
    /// The session this was forked from, if any. Persisted for the fork tree
    /// (which reads it via [`Meta`]); kept on the live session for completeness.
    pub parent: Option<String>,
    pub path: PathBuf,
}

/// Encode `cwd` into a safe directory name: `--home-user-proj--`.
fn encode_cwd(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let trimmed = s.trim_start_matches(['/', '\\']);
    let safe: String = trimmed
        .chars()
        .map(|c| if matches!(c, '/' | '\\' | ':') { '-' } else { c })
        .collect();
    format!("--{safe}--")
}

/// The sessions storage root: `$TAPIR_SESSION_DIR` when set, else
/// `<config_dir>/sessions`.
pub fn sessions_root(config_dir: &Path) -> PathBuf {
    std::env::var_os("TAPIR_SESSION_DIR")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| config_dir.join("sessions"))
}

/// The sessions directory for `cwd` under the storage root (created on demand).
pub fn dir_for(config_dir: &Path, cwd: &Path) -> PathBuf {
    sessions_root(config_dir).join(encode_cwd(cwd))
}

fn new_id() -> String {
    let nanos = Local::now().timestamp_nanos_opt().unwrap_or(0);
    format!("{:x}{:x}", std::process::id(), nanos as u64)
}

fn now_rfc3339() -> String {
    Local::now().to_rfc3339()
}

fn file_stamp() -> String {
    Local::now().format("%Y%m%dT%H%M%S").to_string()
}

impl Session {
    /// Start a fresh session file in `dir` for `cwd` (header written immediately).
    pub fn create(
        dir: &Path,
        cwd: &Path,
        parent: Option<String>,
    ) -> Result<Self> {
        Self::create_with_id(dir, cwd, parent, new_id())
    }

    /// Like [`create`](Self::create) but for a caller-supplied id — the file
    /// store creates conversations keyed by an external identifier while
    /// keeping the header and layout identical.
    pub fn create_with_id(
        dir: &Path,
        cwd: &Path,
        parent: Option<String>,
        id: String,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        crate::config::restrict_dir(dir);
        let path = dir.join(format!("{}_{}.jsonl", file_stamp(), id));
        let mut header = json!({
            "type": "session",
            "version": VERSION,
            "id": id,
            "timestamp": now_rfc3339(),
            "cwd": cwd.to_string_lossy(),
        });
        if let Some(p) = &parent {
            header["parentSession"] = json!(p);
        }
        write_line(&path, &header)?;
        // Sessions can contain pasted secrets — keep them owner-only.
        crate::config::restrict_file(&path);
        Ok(Self { id, name: None, parent, path })
    }

    /// Append a conversation message.
    pub fn append_message(&self, role: Role, text: &str) -> Result<()> {
        let role_str = match role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let entry = json!({
            "type": "message",
            "id": new_id(),
            "parentId": Value::Null,
            "timestamp": now_rfc3339(),
            "message": {
                "role": role_str,
                "content": [TextContent { kind: "text".into(), text: text.to_string() }],
            },
        });
        write_line(&self.path, &entry)
    }

    /// Append a `!`/`!!` shell command and its output.
    #[allow(clippy::too_many_arguments)]
    pub fn append_bash(
        &self,
        command: &str,
        output: &str,
        exit_code: Option<i32>,
        cancelled: bool,
        exclude_from_context: bool,
    ) -> Result<()> {
        let entry = json!({
            "type": "bash",
            "id": new_id(),
            "parentId": Value::Null,
            "timestamp": now_rfc3339(),
            "command": command,
            "output": output,
            "exitCode": exit_code,
            "cancelled": cancelled,
            "excludeFromContext": exclude_from_context,
        });
        write_line(&self.path, &entry)
    }

    /// Append an agent tool call and its result.
    pub fn append_tool(
        &self,
        name: &str,
        title: &str,
        output: &str,
        is_error: bool,
        took_ms: u64,
    ) -> Result<()> {
        let entry = json!({
            "type": "tool",
            "id": new_id(),
            "parentId": Value::Null,
            "timestamp": now_rfc3339(),
            "name": name,
            "title": title,
            "output": output,
            "isError": is_error,
            "tookMs": took_ms,
        });
        write_line(&self.path, &entry)
    }

    /// Append an auto-compaction checkpoint, so resuming rebuilds the summary.
    pub fn append_compaction(&self, summary: &str) -> Result<()> {
        let entry = json!({
            "type": "compaction",
            "id": new_id(),
            "parentId": Value::Null,
            "timestamp": now_rfc3339(),
            "summary": summary,
        });
        write_line(&self.path, &entry)
    }

    /// Set (or clear) the session's display name via a `session_info` entry.
    pub fn set_name(&mut self, name: &str) -> Result<()> {
        let name = name.trim().to_string();
        let entry = json!({
            "type": "session_info",
            "id": new_id(),
            "parentId": Value::Null,
            "timestamp": now_rfc3339(),
            "name": name,
        });
        write_line(&self.path, &entry)?;
        self.name = (!name.is_empty()).then_some(name);
        Ok(())
    }
}

fn write_line(path: &Path, value: &Value) -> Result<()> {
    use std::io::Write;
    let mut f =
        std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{value}")?;
    Ok(())
}

/// Parse the content array (or string) of a message into plain text.
fn content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Read every entry line of a session file.
fn read_entries(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Load a session file into its messages + metadata.
pub fn load(path: &Path) -> Result<Loaded> {
    let entries = read_entries(path);
    let header = entries
        .iter()
        .find(|e| e["type"] == "session")
        .ok_or_else(|| anyhow!("not a session file (no header)"))?;
    let id = header["id"].as_str().unwrap_or_default().to_string();
    let parent = header["parentSession"].as_str().map(str::to_string);

    let mut name = None;
    let mut items = Vec::new();
    for e in &entries {
        match e["type"].as_str() {
            Some("session_info") => {
                name = e["name"]
                    .as_str()
                    .map(str::to_string)
                    .filter(|n| !n.is_empty());
            }
            Some("message") => {
                let role = match e["message"]["role"].as_str() {
                    Some("assistant") => Role::Assistant,
                    _ => Role::User,
                };
                let text = content_text(&e["message"]);
                if !text.is_empty() {
                    items.push(LoadedItem::Message { role, text });
                }
            }
            Some("compaction") => {
                if let Some(summary) = e["summary"].as_str() {
                    items.push(LoadedItem::Compaction {
                        summary: summary.to_string(),
                    });
                }
            }
            Some("bash") => {
                items.push(LoadedItem::Bash {
                    command: e["command"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    output: e["output"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    exit_code: e["exitCode"].as_i64().map(|c| c as i32),
                    cancelled: e["cancelled"].as_bool().unwrap_or(false),
                    exclude_from_context: e["excludeFromContext"]
                        .as_bool()
                        .unwrap_or(false),
                });
            }
            Some("tool") => {
                items.push(LoadedItem::Tool {
                    name: e["name"].as_str().unwrap_or_default().to_string(),
                    title: e["title"].as_str().unwrap_or_default().to_string(),
                    output: e["output"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    is_error: e["isError"].as_bool().unwrap_or(false),
                    took_ms: e["tookMs"].as_u64().unwrap_or(0),
                });
            }
            _ => {}
        }
    }
    Ok(Loaded { id, name, parent, items })
}

/// All sessions for `cwd`, newest first (by filename, which is timestamp-sorted).
pub fn list(config_dir: &Path, cwd: &Path) -> Vec<Meta> {
    list_dir(&dir_for(config_dir, cwd))
}

/// All sessions in a specific sessions directory, newest first.
pub fn list_dir(dir: &Path) -> Vec<Meta> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    paths.sort();
    paths.reverse();

    paths.iter().filter_map(|p| meta(p)).collect()
}

fn meta(path: &Path) -> Option<Meta> {
    let entries = read_entries(path);
    let header = entries.iter().find(|e| e["type"] == "session")?;
    let id = header["id"].as_str()?.to_string();
    let mut name = None;
    let mut preview = String::new();
    let mut messages = 0;
    for e in &entries {
        match e["type"].as_str() {
            Some("session_info") => {
                name = e["name"]
                    .as_str()
                    .map(str::to_string)
                    .filter(|n| !n.is_empty());
            }
            Some("message") => {
                messages += 1;
                if preview.is_empty() && e["message"]["role"] == "user" {
                    preview = content_text(&e["message"])
                        .lines()
                        .next()
                        .unwrap_or("")
                        .to_string();
                }
            }
            _ => {}
        }
    }
    Some(Meta {
        path: path.to_path_buf(),
        id,
        name,
        timestamp: header["timestamp"].as_str().unwrap_or_default().to_string(),
        parent: header["parentSession"].as_str().map(str::to_string),
        preview,
        messages,
    })
}

/// Every session across all folders (scope = "All"), newest first.
pub fn list_all(config_dir: &Path) -> Vec<Meta> {
    let root = sessions_root(config_dir);
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut metas: Vec<Meta> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| list_dir(&d))
        .collect();
    // Newest first across folders (filenames embed a sortable timestamp).
    metas.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    metas
}

/// A short "2h ago" style label for an RFC-3339 timestamp.
pub fn relative_time(rfc3339: &str) -> String {
    let Ok(then) = chrono::DateTime::parse_from_rfc3339(rfc3339) else {
        return String::new();
    };
    let secs = (Local::now().timestamp() - then.timestamp()).max(0);
    if secs < 60 {
        "just now".into()
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

/// Copy `src`'s entries into a new session file in `dir`, with a fresh id and
/// (optionally) a `parentSession`. Used by `/fork` and `/clone`.
pub fn copy_from(
    dir: &Path,
    src: &Path,
    parent: Option<String>,
) -> Result<Session> {
    std::fs::create_dir_all(dir)?;
    let entries = read_entries(src);
    let id = new_id();
    let path = dir.join(format!("{}_{}.jsonl", file_stamp(), id));
    let cwd = entries
        .iter()
        .find(|e| e["type"] == "session")
        .and_then(|h| h["cwd"].as_str())
        .unwrap_or_default()
        .to_string();
    let mut name = None;
    let mut header = json!({
        "type": "session", "version": VERSION, "id": id,
        "timestamp": now_rfc3339(), "cwd": cwd,
    });
    if let Some(p) = &parent {
        header["parentSession"] = json!(p);
    }
    write_line(&path, &header)?;
    // Re-emit the non-header entries (messages, names) unchanged.
    for e in &entries {
        if e["type"] == "session" {
            continue;
        }
        if e["type"] == "session_info" {
            name = e["name"]
                .as_str()
                .map(str::to_string)
                .filter(|n| !n.is_empty());
        }
        write_line(&path, e)?;
    }
    Ok(Session { id, name, parent, path })
}

/// Import an external `.jsonl` into `dir` as a new session (fresh id).
pub fn import(dir: &Path, src: &Path) -> Result<Session> {
    if !src.is_file() {
        return Err(anyhow!("no such file: {}", src.display()));
    }
    copy_from(dir, src, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(label: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("tapir-sess-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn encodes_cwd_like_pi() {
        assert_eq!(encode_cwd(Path::new("/home/u/proj")), "--home-u-proj--");
    }

    #[test]
    fn sessions_root_honors_env_override() {
        let cfg = Path::new("/cfg");
        // Default: `<config>/sessions`.
        assert_eq!(sessions_root(cfg), cfg.join("sessions"));
        // SAFETY: single-threaded test owning its own scratch var.
        unsafe {
            std::env::set_var("TAPIR_SESSION_DIR", "/tmp/tapir-sess-override")
        };
        assert_eq!(sessions_root(cfg), Path::new("/tmp/tapir-sess-override"));
        assert_eq!(
            dir_for(cfg, Path::new("/home/u/proj")),
            Path::new("/tmp/tapir-sess-override/--home-u-proj--")
        );
        unsafe { std::env::remove_var("TAPIR_SESSION_DIR") };
    }

    #[test]
    fn create_append_and_load_roundtrip() {
        let cfg = tmp("rt");
        let cwd = Path::new("/home/u/proj");
        let dir = dir_for(&cfg, cwd);
        let mut s = Session::create(&dir, cwd, None).unwrap();
        s.append_message(Role::User, "hello").unwrap();
        s.append_message(Role::Assistant, "hi there").unwrap();
        s.set_name("My chat").unwrap();

        let loaded = load(&s.path).unwrap();
        assert_eq!(loaded.id, s.id);
        assert_eq!(loaded.name.as_deref(), Some("My chat"));
        assert_eq!(loaded.items.len(), 2);
        assert!(
            matches!(&loaded.items[0], LoadedItem::Message { text, .. } if text == "hello")
        );
        assert!(matches!(
            &loaded.items[1],
            LoadedItem::Message { role: Role::Assistant, .. }
        ));

        let listed = list(&cfg, cwd);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].label(), "My chat");
        let _ = std::fs::remove_dir_all(&cfg);
    }

    #[test]
    fn fork_links_to_parent_and_copies_messages() {
        let cfg = tmp("fork");
        let cwd = Path::new("/home/u/proj");
        let dir = dir_for(&cfg, cwd);
        let s = Session::create(&dir, cwd, None).unwrap();
        s.append_message(Role::User, "original").unwrap();

        let forked = copy_from(&dir, &s.path, Some(s.id.clone())).unwrap();
        assert_ne!(forked.id, s.id);
        let loaded = load(&forked.path).unwrap();
        assert_eq!(loaded.parent.as_deref(), Some(s.id.as_str()));
        assert!(
            matches!(&loaded.items[0], LoadedItem::Message { text, .. } if text == "original")
        );
        let _ = std::fs::remove_dir_all(&cfg);
    }

    #[test]
    fn bash_and_tool_entries_persist_and_load() {
        let cfg = tmp("btool");
        let cwd = Path::new("/home/u/proj");
        let dir = dir_for(&cfg, cwd);
        let s = Session::create(&dir, cwd, None).unwrap();
        s.append_bash("ls", "a.txt\n", Some(0), false, true).unwrap();
        s.append_tool("read", "read src/main.rs", "fn main(){}", false, 12)
            .unwrap();

        let loaded = load(&s.path).unwrap();
        assert!(matches!(
            &loaded.items[0],
            LoadedItem::Bash { command, exclude_from_context: true, .. } if command == "ls"
        ));
        assert!(matches!(
            &loaded.items[1],
            LoadedItem::Tool { name, took_ms: 12, .. } if name == "read"
        ));
        let _ = std::fs::remove_dir_all(&cfg);
    }

    #[test]
    fn compaction_entry_persists_and_loads_in_order() {
        let cfg = tmp("compact");
        let cwd = Path::new("/home/u/proj");
        let dir = dir_for(&cfg, cwd);
        let s = Session::create(&dir, cwd, None).unwrap();
        s.append_message(Role::User, "old q").unwrap();
        s.append_compaction("SUMMARY").unwrap();
        s.append_message(Role::User, "new q").unwrap();

        let loaded = load(&s.path).unwrap();
        assert!(
            matches!(&loaded.items[0], LoadedItem::Message { text, .. } if text == "old q")
        );
        assert!(
            matches!(&loaded.items[1], LoadedItem::Compaction { summary } if summary == "SUMMARY")
        );
        assert!(
            matches!(&loaded.items[2], LoadedItem::Message { text, .. } if text == "new q")
        );
        let _ = std::fs::remove_dir_all(&cfg);
    }
}
