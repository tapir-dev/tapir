//! Message queue — steering / follow-up messages.
//!
//! While the agent is working you can queue input:
//! - **steering** (Enter) is delivered after the current round's tool calls,
//!   before the next model round — it nudges the work in progress,
//! - **follow-up** (Alt+Enter) waits until the agent is fully idle, then runs.
//!
//! The queue is an `Arc<Mutex<…>>` shared with the background chat task: the app
//! pushes/clears it (key handling, display) and the task drains it between
//! rounds. Delivery honors a per-kind [`Mode`] (one-at-a-time vs all at once),
//! configured in Settings (`steeringMode` / `followUpMode`).

use std::sync::{Arc, Mutex};

use crate::agent::Image;

/// Which delivery slot a queued message belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Steer,
    FollowUp,
}

/// A queued message: what's shown/restored (`display`), what the model receives
/// (`model_text`, with `@path` references expanded), and any image attachments.
#[derive(Clone)]
pub struct Queued {
    pub display: String,
    pub model_text: String,
    pub images: Vec<Image>,
    pub kind: Kind,
}

/// The queue, shared between the app and the background chat task.
pub type Shared = Arc<Mutex<Vec<Queued>>>;

/// A fresh, empty queue.
pub fn new() -> Shared {
    Arc::new(Mutex::new(Vec::new()))
}

/// How many queued messages of a kind to deliver at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Deliver one message, await the reply, then the next (the default).
    OneAtATime,
    /// Deliver all queued messages at once before the next reply.
    All,
}

impl Mode {
    /// Parse a settings value (`"all"` / `"one-at-a-time"`).
    pub fn from_setting(s: &str) -> Mode {
        if s == "all" { Mode::All } else { Mode::OneAtATime }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::All => "all",
            Mode::OneAtATime => "one-at-a-time",
        }
    }
}

/// Remove and return the messages of `kind` to deliver now, honoring `mode`
/// (one, or all of that kind), preserving the rest in order.
pub fn drain(q: &Shared, kind: Kind, mode: Mode) -> Vec<Queued> {
    let mut guard = q.lock().unwrap();
    let take = match mode {
        Mode::All => usize::MAX,
        Mode::OneAtATime => 1,
    };
    let mut out = Vec::new();
    let mut kept = Vec::with_capacity(guard.len());
    for m in guard.drain(..) {
        if m.kind == kind && out.len() < take {
            out.push(m);
        } else {
            kept.push(m);
        }
    }
    *guard = kept;
    out
}

/// Clear the queue and return every message's display text, steering first then
/// follow-up — for Escape / dequeue restoration.
pub fn take_for_restore(q: &Shared) -> Vec<String> {
    let mut guard = q.lock().unwrap();
    let mut out: Vec<String> = guard
        .iter()
        .filter(|m| m.kind == Kind::Steer)
        .map(|m| m.display.clone())
        .collect();
    out.extend(
        guard
            .iter()
            .filter(|m| m.kind == Kind::FollowUp)
            .map(|m| m.display.clone()),
    );
    guard.clear();
    out
}

/// Whether the queue holds no messages.
#[cfg(test)]
pub fn is_empty(q: &Shared) -> bool {
    q.lock().unwrap().is_empty()
}

/// Clear the queue and return every message in full (steering first, then
/// follow-up) — used to flush a race-window leftover into a fresh turn.
pub fn drain_all_full(q: &Shared) -> Vec<Queued> {
    let mut guard = q.lock().unwrap();
    let (mut steer, mut follow): (Vec<Queued>, Vec<Queued>) =
        guard.drain(..).partition(|m| m.kind == Kind::Steer);
    steer.append(&mut follow);
    steer
}

/// Display snapshot: (steering messages, follow-up messages), in order.
pub fn snapshot(q: &Shared) -> (Vec<String>, Vec<String>) {
    let guard = q.lock().unwrap();
    let steer = guard
        .iter()
        .filter(|m| m.kind == Kind::Steer)
        .map(|m| m.display.clone())
        .collect();
    let follow = guard
        .iter()
        .filter(|m| m.kind == Kind::FollowUp)
        .map(|m| m.display.clone())
        .collect();
    (steer, follow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(kind: Kind, text: &str) -> Queued {
        Queued {
            display: text.into(),
            model_text: text.into(),
            images: vec![],
            kind,
        }
    }

    #[test]
    fn one_at_a_time_takes_a_single_message_of_the_kind() {
        let shared = new();
        shared.lock().unwrap().extend([
            q(Kind::Steer, "a"),
            q(Kind::FollowUp, "f"),
            q(Kind::Steer, "b"),
        ]);
        let got = drain(&shared, Kind::Steer, Mode::OneAtATime);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].display, "a");
        // The other steer + the follow-up remain.
        let (steer, follow) = snapshot(&shared);
        assert_eq!(steer, ["b"]);
        assert_eq!(follow, ["f"]);
    }

    #[test]
    fn all_mode_takes_every_message_of_the_kind() {
        let shared = new();
        shared.lock().unwrap().extend([
            q(Kind::Steer, "a"),
            q(Kind::Steer, "b"),
            q(Kind::FollowUp, "f"),
        ]);
        let got = drain(&shared, Kind::Steer, Mode::All);
        assert_eq!(
            got.iter().map(|m| m.display.clone()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(snapshot(&shared).1, ["f"], "follow-up untouched");
    }

    #[test]
    fn restore_orders_steering_then_follow_up_and_clears() {
        let shared = new();
        shared.lock().unwrap().extend([
            q(Kind::FollowUp, "f1"),
            q(Kind::Steer, "s1"),
            q(Kind::FollowUp, "f2"),
        ]);
        let restored = take_for_restore(&shared);
        assert_eq!(restored, ["s1", "f1", "f2"]);
        assert!(is_empty(&shared));
    }
}
