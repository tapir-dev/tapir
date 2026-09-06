// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The `.model("id")` builder convenience: resolve a provider offline from the
//! compiled-in catalog, with `.model` and `.provider` mutually exclusive and
//! validated at `build()`. Gated on a provider feature, which is what makes the
//! convenience compile at all. nextest runs each test in its own process, so
//! setting the credential env var is race-free.
#![cfg(feature = "anthropic")]

use async_trait::async_trait;
use tapir::{Agent, Error};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    Provider, StreamEvents,
};

/// A do-nothing provider, only ever used to occupy the `.provider` seam in the
/// mutual-exclusion test; it is never called.
struct Noop;

#[async_trait]
impl Provider for Noop {
    async fn complete(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        unreachable!("never called")
    }

    async fn complete_stream(
        &self,
        _ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        unreachable!("never called")
    }
}

#[test]
fn model_resolves_offline_with_env_credential() {
    // SAFETY: nextest runs each test in its own process, so this mutation is not
    // observed by any concurrent test.
    unsafe {
        std::env::set_var("ANTHROPIC_API_KEY", "sk-test-offline");
    }
    let agent = Agent::builder()
        .model("claude-haiku-4-5")
        .system("You are helpful.")
        .build();
    assert!(
        agent.is_ok(),
        "a known id with an env credential must resolve offline: {:?}",
        agent.err()
    );
}

/// Assert a builder configuration fails at `build()` with [`Error::Build`].
/// Hand-rolled rather than `expect_err` because [`Agent`] is not `Debug`.
fn assert_build_error(result: Result<Agent, Error>) {
    match result {
        Ok(_) => panic!("expected a build error, got a built agent"),
        Err(err) => assert!(
            matches!(err, Error::Build(_)),
            "expected Error::Build, got {err:?}"
        ),
    }
}

#[test]
fn model_and_provider_are_mutually_exclusive() {
    assert_build_error(
        Agent::builder()
            .model("claude-haiku-4-5")
            .provider(Noop)
            .build(),
    );
}

#[test]
fn neither_model_nor_provider_fails() {
    assert_build_error(Agent::builder().build());
}

#[test]
fn unknown_model_id_fails_at_build() {
    assert_build_error(Agent::builder().model("no-such-model-xyz").build());
}
