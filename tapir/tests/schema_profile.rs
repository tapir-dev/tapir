// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The send-path `SchemaProfile` seam: the active profile normalizes each
//! offered tool's argument schema before the provider call. `NonStrict` (the
//! default) is a passthrough; a custom profile's rewrite is what the provider
//! actually receives. Both run against an in-crate scripted [`Provider`] that
//! records the schema it was handed.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tapir::Agent;
use tapir::schema::SchemaProfile;
use tapir::tool::{
    Concurrency, ErasedTool, Tool, ToolCtx, ToolError, UpdateSink,
};
use tapir_provider::{
    AssistantMessage, CompletionOptions, Context, Error as ProviderError,
    FinishReason, Provider, StreamAccumulator, StreamEvent, StreamEvents,
    Usage,
};

/// A `Provider` that answers with a fixed tool-free reply and records the
/// argument schema of the first tool it was offered, so a test can prove the
/// profile ran on the send path.
struct RecordingProvider {
    seen: Arc<Mutex<Option<Value>>>,
}

impl RecordingProvider {
    fn new() -> (Self, Arc<Mutex<Option<Value>>>) {
        let seen = Arc::new(Mutex::new(None));
        (Self { seen: seen.clone() }, seen)
    }

    fn record(&self, ctx: &Context) -> Vec<StreamEvent> {
        if let Some(def) = ctx.tools.first() {
            *self.seen.lock().unwrap() = Some(def.input_schema.clone());
        }
        vec![
            StreamEvent::MessageStart,
            StreamEvent::TextStart { index: 0 },
            StreamEvent::TextDelta {
                index: 0,
                text: "done".to_string(),
            },
            StreamEvent::TextEnd { index: 0 },
            StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            },
        ]
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    async fn complete(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, ProviderError> {
        Ok(StreamAccumulator::fold(&self.record(ctx)))
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<StreamEvents, ProviderError> {
        let events = self.record(ctx);
        Ok(Box::pin(futures_util::stream::iter(
            events.into_iter().map(Ok::<StreamEvent, ProviderError>),
        )))
    }
}

/// A trivial tool with a known argument schema, so a test can compare what the
/// provider was sent against the tool's own definition.
struct Echo;

#[async_trait]
impl Tool for Echo {
    type Args = ();
    type Output = String;
    type Error = ToolError;

    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echoes"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }

    async fn execute(
        &self,
        (): (),
        _ctx: &ToolCtx,
        _on_update: &mut UpdateSink<'_>,
    ) -> Result<String, ToolError> {
        Ok("ok".to_string())
    }
}

/// A profile that stamps a marker key into every schema it normalizes, so a
/// test can prove the rewrite reached the provider.
struct Marker;

impl SchemaProfile for Marker {
    fn normalize(&self, schema: &mut Value) {
        if let Value::Object(map) = schema {
            map.insert("x-normalized".to_string(), json!(true));
        }
    }
}

#[tokio::test]
async fn non_strict_is_passthrough() {
    let (provider, seen) = RecordingProvider::new();
    let agent = Agent::builder()
        .provider(provider)
        .tool(Echo)
        .build()
        .expect("build");

    let _ = agent.prompt("hi").await.expect("run");

    let sent = seen.lock().unwrap().clone().expect("a tool was offered");
    let original = ErasedTool::definition(&Echo).input_schema;
    assert_eq!(
        sent, original,
        "the default NonStrict profile must send the schema verbatim"
    );
}

#[tokio::test]
async fn custom_profile_rewrites_on_send_path() {
    let (provider, seen) = RecordingProvider::new();
    let agent = Agent::builder()
        .provider(provider)
        .schema_profile(Marker)
        .tool(Echo)
        .build()
        .expect("build");

    let _ = agent.prompt("hi").await.expect("run");

    let sent = seen.lock().unwrap().clone().expect("a tool was offered");
    assert_eq!(
        sent.get("x-normalized"),
        Some(&json!(true)),
        "the custom profile's rewrite must reach the provider: {sent}"
    );
}
