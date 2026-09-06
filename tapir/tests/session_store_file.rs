// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Tests for the reference `FileSessionStore` backend (feature `store-file`):
//! JSONL round-trip in order, a torn trailing line dropped while a mid-file
//! corrupt or unknown-tag line fails closed with the log left intact, a second
//! writer contending the `flock` failing closed, and the on-disk permissions
//! (`0o600` file, `0o700` parent dirs). Each test runs against a real,
//! per-test temp directory.

#![cfg(feature = "store-file")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use tapir::message::{AgentMessage, CustomMessage, NoCustom};
use tapir::store::SessionStore;
use tapir::store::file::FileSessionStore;
use tapir_provider::Message;

/// A per-test temp directory, removed on drop. Names are unique across the
/// process so parallel tests never collide.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tapir-file-store-{}-{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Build a plain user message for terse round-trip fixtures.
fn user(text: &str) -> AgentMessage {
    AgentMessage::Llm(Message::user(text))
}

/// Assert an `AgentMessage` is a user message carrying `expected` text.
fn assert_user(message: &AgentMessage, expected: &str) {
    match message {
        AgentMessage::Llm(Message::User { content }) => {
            let text = content.iter().fold(String::new(), |mut acc, part| {
                if let tapir_provider::ContentPart::Text(t) = part {
                    acc.push_str(t);
                }
                acc
            });
            assert_eq!(text, expected);
        }
        other => panic!("expected a user message, got {other:?}"),
    }
}

#[tokio::test]
async fn append_and_reload_round_trips_in_order() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    let store = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("open");

    store.append(&user("one")).await.expect("append one");
    store.append(&user("two")).await.expect("append two");
    store.append(&user("three")).await.expect("append three");

    let history = store.load().await.expect("load");
    assert_eq!(history.len(), 3);
    assert_user(&history[0], "one");
    assert_user(&history[1], "two");
    assert_user(&history[2], "three");
}

#[tokio::test]
async fn reopen_reloads_persisted_history() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    {
        let store = FileSessionStore::<NoCustom>::open(&path)
            .await
            .expect("open");
        store.append(&user("persisted")).await.expect("append");
    } // drop releases the flock

    let reopened = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("reopen");
    let history = reopened.load().await.expect("load");
    assert_eq!(history.len(), 1);
    assert_user(&history[0], "persisted");
}

#[tokio::test]
async fn missing_and_empty_logs_load_empty() {
    let dir = TempDir::new();

    // Missing: never appended to.
    let missing = FileSessionStore::<NoCustom>::open(dir.join("missing.jsonl"))
        .await
        .expect("open missing");
    assert!(missing.load().await.expect("load missing").is_empty());

    // Empty: opened (created) but nothing appended.
    let empty = FileSessionStore::<NoCustom>::open(dir.join("empty.jsonl"))
        .await
        .expect("open empty");
    assert!(empty.load().await.expect("load empty").is_empty());
}

#[tokio::test]
async fn torn_trailing_line_is_dropped() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    // Two committed records (each LF-terminated) followed by a half-written,
    // unterminated partial append — a torn last write.
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n{}",
            serde_json::to_string(&user("first")).unwrap(),
            serde_json::to_string(&user("second")).unwrap(),
            r#"{"Llm":{"User":{"content":[{"Text":"tor"#,
        ),
    )
    .expect("write torn log");

    let store = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("open");
    let history = store.load().await.expect("load drops torn tail");
    assert_eq!(history.len(), 2);
    assert_user(&history[0], "first");
    assert_user(&history[1], "second");
}

#[tokio::test]
async fn mid_file_corrupt_line_fails_closed_with_log_intact() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    // A committed corrupt line (terminated with '\n') between two good ones.
    let contents = format!(
        "{}\n{}\n{}\n",
        serde_json::to_string(&user("good-one")).unwrap(),
        "{ not json at all",
        serde_json::to_string(&user("good-two")).unwrap(),
    );
    std::fs::write(&path, &contents).expect("write corrupt log");

    let store = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("open");
    store.load().await.expect_err("corrupt line fails closed");

    // The log is left byte-for-byte intact — load never rewrites.
    let after = std::fs::read_to_string(&path).expect("reread log");
    assert_eq!(after, contents);
}

#[tokio::test]
async fn unknown_tag_line_fails_closed() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    // A committed record with an unknown external tag.
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&user("good")).unwrap(),
            r#"{"Bogus":{"x":1}}"#,
        ),
    )
    .expect("write log");

    let store = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("open");
    store.load().await.expect_err("unknown tag fails closed");
}

#[tokio::test]
async fn second_writer_contending_the_lock_fails_closed() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    let first = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("first open");

    // A second opener on the held log must fail closed rather than interleave.
    let contended = FileSessionStore::<NoCustom>::open(&path).await;
    assert!(
        contended.is_err(),
        "a second writer must fail closed on the held flock"
    );

    // After the first writer drops, the lock is free again.
    drop(first);
    FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("reopen after release");
}

#[cfg(unix)]
#[tokio::test]
async fn on_disk_permissions_are_private() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new();
    // A nested parent so the created directory's mode is observable.
    let parent = dir.join("nested/dir");
    let path = parent.join("session.jsonl");

    let store = FileSessionStore::<NoCustom>::open(&path)
        .await
        .expect("open");
    store.append(&user("x")).await.expect("append");

    let file_mode = std::fs::metadata(&path)
        .expect("stat file")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(file_mode, 0o600, "log file must be 0o600");

    let dir_mode = std::fs::metadata(&parent)
        .expect("stat dir")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(dir_mode, 0o700, "created parent dirs must be 0o700");
}

/// A caller-defined message type: one variant reaches the model, one stays
/// UI-only. Serde-derived so it round-trips through the file backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Note {
    /// A UI-only note; never converted for the model.
    Ui(String),
    /// A note that converts to a user turn for the model.
    Say(String),
}

impl CustomMessage for Note {
    fn to_llm(&self) -> Option<Message> {
        match self {
            Note::Ui(_) => None,
            Note::Say(text) => Some(Message::user(text.clone())),
        }
    }
}

#[tokio::test]
async fn custom_messages_round_trip_through_the_file_backend() {
    let dir = TempDir::new();
    let path = dir.join("session.jsonl");

    let originals: Vec<AgentMessage<Note>> = vec![
        AgentMessage::Llm(Message::user("plain")),
        AgentMessage::Custom(Note::Ui("banner".to_string())),
        AgentMessage::Custom(Note::Say("spoken".to_string())),
    ];

    let store = FileSessionStore::<Note>::open(&path).await.expect("open");
    for message in &originals {
        store.append(message).await.expect("append");
    }

    let loaded = store.load().await.expect("load");
    assert_eq!(
        serde_json::to_value(&originals).unwrap(),
        serde_json::to_value(&loaded).unwrap(),
    );
}
