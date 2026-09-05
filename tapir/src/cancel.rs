// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Batch-scoped cooperative cancellation.
//!
//! A run mints one [`CancelTrigger`]/[`Cancel`] pair. The trigger is held by the
//! driver (the abort ticket wires it onto the [`RunHandle`](crate::agent::RunHandle));
//! the observer half is cloned into the batch executor and into every
//! [`ToolCtx`](crate::tool::ToolCtx). Firing the trigger fans out to all clones at
//! once, so the executor stops launching queued calls and a long-running tool that
//! selects on [`Cancel::cancelled`] can bail — the cascade that leaves no orphaned
//! work.
//!
//! Cancellation is cooperative for the tool's own body but pre-emptive for the
//! task: the executor aborts the in-flight join, so a tool that never checks its
//! `ctx` is still dropped at its next await point.

use std::future;

use tokio::sync::watch;

/// The observer half of a cancellation pair, cloned wherever a batch or a tool
/// needs to watch for cancellation. Cheap to clone.
#[derive(Debug, Clone)]
pub(crate) struct Cancel {
    rx: watch::Receiver<bool>,
}

/// The trigger half, held by the run's driver. Dropping it without firing leaves
/// the paired [`Cancel`] permanently un-cancelled.
///
/// Scaffolding: the run holds one but never fires it yet — the abort ticket wires
/// [`cancel`](Self::cancel) onto [`RunHandle`](crate::agent::RunHandle). Exercised
/// by the executor's cancellation tests meanwhile.
#[derive(Debug)]
#[allow(dead_code, reason = "trigger fired by the abort ticket; tested here")]
pub(crate) struct CancelTrigger {
    tx: watch::Sender<bool>,
}

/// Mint a fresh trigger/observer pair, initially un-cancelled.
pub(crate) fn cancel_pair() -> (CancelTrigger, Cancel) {
    let (tx, rx) = watch::channel(false);
    (CancelTrigger { tx }, Cancel { rx })
}

#[allow(dead_code, reason = "trigger fired by the abort ticket; tested here")]
impl CancelTrigger {
    /// Fire cancellation. Idempotent; every live [`Cancel`] clone observes it.
    pub(crate) fn cancel(&self) {
        // Ignore a send error: it only means every observer was already dropped,
        // so there is nothing left to cancel.
        let _ = self.tx.send(true);
    }
}

impl Cancel {
    /// A token that can never fire — its trigger is dropped on the spot. The
    /// default for a [`ToolCtx`](crate::tool::ToolCtx) built outside a run.
    pub(crate) fn never() -> Self {
        // Drop the sender immediately: `is_cancelled` stays false and `cancelled`
        // parks forever.
        let (_tx, rx) = watch::channel(false);
        Self { rx }
    }

    /// Whether cancellation has already fired. A cheap, non-blocking check for a
    /// tool to poll between steps.
    pub(crate) fn is_cancelled(&self) -> bool {
        *self.rx.borrow()
    }

    /// Resolve once cancellation fires. If the trigger was dropped without firing
    /// (e.g. [`never`](Self::never)), this parks forever, so it is safe to
    /// `select!` on without spuriously waking.
    pub(crate) async fn cancelled(&self) {
        let mut rx = self.rx.clone();
        loop {
            if *rx.borrow_and_update() {
                return;
            }
            if rx.changed().await.is_err() {
                // The trigger is gone and the value is still false: cancellation
                // can never fire, so never resolve.
                future::pending::<()>().await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn fires_and_fans_out_to_clones() {
        let (trigger, cancel) = cancel_pair();
        let clone = cancel.clone();
        assert!(!cancel.is_cancelled());

        trigger.cancel();

        assert!(cancel.is_cancelled());
        assert!(clone.is_cancelled());
        // Both the original and the clone resolve their `cancelled` future.
        cancel.cancelled().await;
        clone.cancelled().await;
    }

    #[tokio::test]
    async fn cancelled_awaits_a_later_fire() {
        let (trigger, cancel) = cancel_pair();
        let waiter = tokio::spawn(async move { cancel.cancelled().await });
        // The future is still pending until the trigger fires.
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!waiter.is_finished());
        trigger.cancel();
        waiter.await.expect("waiter task");
    }

    #[tokio::test]
    async fn never_stays_pending_and_uncancelled() {
        let cancel = Cancel::never();
        assert!(!cancel.is_cancelled());
        // `cancelled` must not resolve; a timeout is our only observable proof.
        let timed =
            tokio::time::timeout(Duration::from_millis(20), cancel.cancelled())
                .await;
        assert!(timed.is_err(), "never() must never resolve cancelled()");
    }

    #[tokio::test]
    async fn dropped_trigger_never_fires() {
        let (trigger, cancel) = cancel_pair();
        drop(trigger);
        assert!(!cancel.is_cancelled());
        let timed =
            tokio::time::timeout(Duration::from_millis(20), cancel.cancelled())
                .await;
        assert!(timed.is_err(), "a dropped trigger must not fire");
    }
}
