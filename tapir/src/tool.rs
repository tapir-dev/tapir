// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The typed tool system: the author-facing [`Tool`] trait, the object-safe
//! [`ErasedTool`] the run loop stores, and the blanket erasure that turns every
//! `Tool` into an `Arc<dyn ErasedTool>`.
//!
//! An author writes a typed `Tool` (usually via the [`#[tool]`](macro@crate::tool)
//! attribute) with a `Deserialize + JsonSchema` argument type. Erasure gives the
//! run loop a uniform, object-safe handle whose [`invoke`](ErasedTool::invoke) is
//! the single dispatch boundary: it JSON-validates raw arguments into the typed
//! `Args` and normalizes any author error into a framework [`ToolError`].

use std::error::Error as StdError;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tapir_provider::{ContentPart, ToolDefinition, ToolResultMessage};

use crate::cancel::Cancel;

/// The concurrency class governing how a batch of tool calls executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Concurrency {
    /// Parallelizable reads: may run alongside any other tool.
    Safe,
    /// Serialized mutations, run behind a barrier against other tools.
    Exclusive,
}

/// Per-call context handed to [`Tool::execute`].
///
/// A `ctx` parameter on a `#[tool]` fn is what makes the tool contextual. It
/// carries the call id and the cancellation seam; a later ticket grows it into a
/// handle back to the agent and session.
pub struct ToolCtx {
    id: String,
    cancel: Cancel,
}

impl ToolCtx {
    /// Build a context for a call, identified by the model-supplied call id. The
    /// context is un-cancellable — the run loop wires the live batch token
    /// internally; this plain constructor suits standalone tool tests.
    pub fn new(call_id: impl Into<String>) -> Self {
        Self {
            id: call_id.into(),
            cancel: Cancel::never(),
        }
    }

    /// Build a context wired to a live batch's cancellation token.
    pub(crate) fn with_cancel(
        call_id: impl Into<String>,
        cancel: Cancel,
    ) -> Self {
        Self {
            id: call_id.into(),
            cancel,
        }
    }

    /// The id of the tool call this context serves.
    #[must_use]
    pub fn call_id(&self) -> &str {
        &self.id
    }

    /// Whether this call's batch has been cancelled. A cheap, non-blocking poll a
    /// long-running tool can check between steps to bail early.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Resolves when this call's batch is cancelled. A tool with a blocking wait
    /// can `select!` on this to abort cooperatively; the executor also aborts the
    /// task itself, so a tool that ignores this is still dropped at its next
    /// await.
    pub async fn cancelled(&self) {
        self.cancel.cancelled().await;
    }
}

/// A flat progress update streamed while a tool executes.
///
/// Deliberately a single flat enum rather than a per-tool associated type, so
/// [`ErasedTool`] stays object-safe and the run loop can forward every update
/// onto one `AgentEvent` stream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolUpdate {
    /// A human-readable progress line.
    Progress(String),
    /// A structured partial result.
    Partial(Value),
}

/// The sink a tool writes progress updates to during [`Tool::execute`].
pub type UpdateSink<'a> = dyn FnMut(ToolUpdate) + Send + 'a;

/// A tool's successful output, normalized to provider content.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    /// The content parts sent back to the model as the tool result.
    pub content: Vec<ContentPart>,
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self {
            content: vec![ContentPart::text(text)],
        }
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::from(text.to_owned())
    }
}

impl From<Vec<ContentPart>> for ToolOutput {
    fn from(content: Vec<ContentPart>) -> Self {
        Self { content }
    }
}

impl From<()> for ToolOutput {
    fn from((): ()) -> Self {
        Self {
            content: Vec::new(),
        }
    }
}

/// A tool failure, split into two deliberately separate channels.
///
/// - `model_message` rides back to the model in the tool result and is the only
///   text the model ever sees; it drives any retry.
/// - `operator_detail` is for logs and telemetry and is never shown to the
///   model.
///
/// There is no `retryable` flag: a failure's model-visibility is the only axis
/// the run loop needs.
#[derive(Debug, thiserror::Error)]
#[error("{model_message}")]
pub struct ToolError {
    /// The model-visible failure text.
    pub model_message: String,
    /// Operator-only detail, never sent to the model.
    #[source]
    pub operator_detail: Option<Box<dyn StdError + Send + Sync>>,
}

impl ToolError {
    /// A model-visible error with no operator detail.
    pub fn model(message: impl Into<String>) -> Self {
        Self {
            model_message: message.into(),
            operator_detail: None,
        }
    }

    /// Attach operator-only detail, kept out of the model's view.
    #[must_use]
    pub fn with_operator(
        mut self,
        detail: impl Into<Box<dyn StdError + Send + Sync>>,
    ) -> Self {
        self.operator_detail = Some(detail.into());
        self
    }

    /// The one error the dispatch boundary mints itself: raw JSON arguments
    /// failed to deserialize into the tool's `Args`.
    fn invalid_args(tool: &str, err: &serde_json::Error) -> Self {
        Self::model(format!("invalid arguments for `{tool}`: {err}"))
    }
}

impl From<String> for ToolError {
    fn from(message: String) -> Self {
        Self::model(message)
    }
}

impl From<&str> for ToolError {
    fn from(message: &str) -> Self {
        Self::model(message)
    }
}

impl From<std::convert::Infallible> for ToolError {
    fn from(never: std::convert::Infallible) -> Self {
        match never {}
    }
}

/// A typed capability the agent can invoke.
///
/// All the ergonomics live here; the object-safe half is derived for free by
/// the blanket [`ErasedTool`] impl. Authors usually reach this trait through the
/// [`#[tool]`](macro@crate::tool) attribute rather than implementing it by hand.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// The argument type. `Deserialize` makes it the validator and `JsonSchema`
    /// derives the parameter schema, so the two can never drift.
    type Args: DeserializeOwned + JsonSchema + Send;
    /// Whatever the tool produces; anything convertible into [`ToolOutput`].
    type Output: Into<ToolOutput> + Send;
    /// The author-chosen error, normalized to [`ToolError`] only at the
    /// dispatch boundary.
    type Error: Into<ToolError> + Send;

    /// The tool's name, as the model references it.
    fn name(&self) -> &str;

    /// A natural-language description of what the tool does.
    fn description(&self) -> &str;

    /// The portable JSON Schema for [`Args`](Tool::Args). The default derives
    /// and post-processes it, so an author never writes schema by hand.
    fn parameters(&self) -> Value {
        portable_schema::<Self::Args>()
    }

    /// The scheduling class. The safe default is [`Exclusive`](Concurrency::Exclusive)
    /// (assume a mutation); a read-only tool overrides to
    /// [`Safe`](Concurrency::Safe).
    fn concurrency(&self) -> Concurrency {
        Concurrency::Exclusive
    }

    /// Run the tool. `on_update` streams partials; `ctx` carries the call id and
    /// the cancellation seam.
    async fn execute(
        &self,
        args: Self::Args,
        ctx: &ToolCtx,
        on_update: &mut UpdateSink<'_>,
    ) -> Result<Self::Output, Self::Error>;
}

/// The object-safe, erased form of a [`Tool`] the run loop stores as
/// `Arc<dyn ErasedTool>`.
///
/// Authors never implement this: the blanket impl erases every `Tool` into it.
/// [`invoke`](ErasedTool::invoke) is the single dispatch boundary above which
/// everything deals only in erased tools.
#[async_trait]
pub trait ErasedTool: Send + Sync + 'static {
    /// The tool's name, as the model references it.
    fn name(&self) -> &str;

    /// A natural-language description of what the tool does.
    fn description(&self) -> &str;

    /// The provider-ready definition: name, description, and portable schema.
    fn definition(&self) -> ToolDefinition;

    /// The scheduling class governing how this tool batches against others.
    fn concurrency(&self) -> Concurrency;

    /// Whether this tool is read-only, derived from its concurrency class: a
    /// tool is read-only exactly when it is [`Safe`](Concurrency::Safe).
    fn read_only(&self) -> bool;

    /// Dispatch a raw model-generated call. This is the boundary: it validates
    /// the raw JSON into the typed `Args`, runs the author code, and normalizes
    /// any author error into a [`ToolError`].
    async fn invoke(
        &self,
        raw_args: Value,
        ctx: &ToolCtx,
        on_update: &mut UpdateSink<'_>,
    ) -> Result<ToolOutput, ToolError>;
}

#[async_trait]
impl<T: Tool> ErasedTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn description(&self) -> &str {
        Tool::description(self)
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            Tool::name(self),
            Tool::description(self),
            self.parameters(),
        )
    }

    fn concurrency(&self) -> Concurrency {
        Tool::concurrency(self)
    }

    fn read_only(&self) -> bool {
        Tool::concurrency(self) == Concurrency::Safe
    }

    async fn invoke(
        &self,
        raw_args: Value,
        ctx: &ToolCtx,
        on_update: &mut UpdateSink<'_>,
    ) -> Result<ToolOutput, ToolError> {
        let args: T::Args = serde_json::from_value(raw_args)
            .map_err(|e| ToolError::invalid_args(Tool::name(self), &e))?;
        let out = self
            .execute(args, ctx, on_update)
            .await
            .map_err(Into::into)?;
        Ok(out.into())
    }
}

/// One model-requested tool call, lifted out of the reply into owned fields so
/// the batch can run while history is mutated without holding a borrow on the
/// message. `Clone` so each spawned call task owns its copy.
///
/// Handed by reference to the [`before_tool_call`](crate::agent::AgentBuilder::before_tool_call)
/// gate and the [`after_tool_call`](crate::agent::AgentBuilder::after_tool_call)
/// observer; the fields are read through [`id`](Self::id), [`name`](Self::name),
/// and [`arguments`](Self::arguments).
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

impl ToolCall {
    /// The model-supplied id of this call, stable across the call's lifetime.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The name of the tool the model asked to invoke.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The call's arguments, as the raw JSON the model produced (or the
    /// rewritten value a [`ToolDecision::Modify`] installed).
    #[must_use]
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }
}

/// The decision returned by the pre-batch approval gate for a single call.
#[non_exhaustive]
pub enum ToolDecision {
    /// Run the call as requested.
    Proceed,
    /// Rewrite the call's arguments, re-running validation.
    Modify {
        /// The replacement arguments.
        arguments: Value,
    },
    /// Reject the call with a model-visible message.
    Deny {
        /// The rejection message surfaced to the model.
        message: String,
    },
}

/// A boxed future whose borrow of the hook's argument is scoped to `'a`. The
/// hook signatures are HRTB (`for<'a>`) so a closure may borrow the call (and
/// result) it is handed straight into the returned future without cloning.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The pre-batch tool-call approval gate. Awaited once per call in model order
/// before the batch executes — and so before the concurrency window opens, so a
/// human wait never holds a concurrency slot. Returns a [`ToolDecision`] that
/// [`Proceed`](ToolDecision::Proceed)s, [`Modify`](ToolDecision::Modify)s the
/// arguments, or [`Deny`](ToolDecision::Deny)s the call. `None` on the builder
/// is a no-op that admits every call.
pub type BeforeToolCall = Arc<
    dyn for<'a> Fn(&'a ToolCall) -> BoxFuture<'a, ToolDecision> + Send + Sync,
>;

/// The post-batch, observe-only tool-result hook. Awaited once per executed
/// result (never for a gate-denied call), in model order, after the batch
/// settles. `None` on the builder is a no-op.
pub type AfterToolCall = Arc<
    dyn for<'a> Fn(&'a ToolCall, &'a ToolResultMessage) -> BoxFuture<'a, ()>
        + Send
        + Sync,
>;

/// Derive the canonical, portable schema for `T`.
///
/// Raw schemars output is post-processed into one schema that survives every
/// provider strict subset: `$ref`s are inlined, meta noise (`$schema`, `title`)
/// is dropped, fieldless enums are flattened from a `oneOf`/`anyOf` of single
/// `const` branches into one flat `{ type, enum }`, and objects get
/// `additionalProperties: false`.
fn portable_schema<T: JsonSchema>() -> Value {
    let settings = SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let schema = SchemaGenerator::new(settings).into_root_schema_for::<T>();
    let mut value =
        serde_json::to_value(schema).expect("schema serializes to JSON");
    normalize(&mut value);
    value
}

/// Recursive post-processing pass over a raw schema value.
fn normalize(value: &mut Value) {
    if let Value::Array(items) = value {
        for item in items {
            normalize(item);
        }
        return;
    }
    let Value::Object(map) = value else { return };
    // Drop meta noise on every object, not just the root: with inlined
    // subschemas a nested struct carries its own `title`/`$schema`.
    map.remove("$schema");
    map.remove("title");

    // Flatten a fieldless enum: schemars renders it as a `oneOf`/`anyOf` of
    // single-`const` branches; strict subsets reject `oneOf`/`const`, so
    // collapse to one flat `{ type, enum }`.
    for key in ["oneOf", "anyOf"] {
        let flat = map
            .get(key)
            .and_then(Value::as_array)
            .and_then(|branches| flatten_const_enum(branches));
        if let Some((ty, values)) = flat {
            map.remove(key);
            map.insert("type".into(), Value::String(ty));
            map.insert("enum".into(), Value::Array(values));
        }
    }

    if map.get("type").and_then(Value::as_str) == Some("object")
        && map.contains_key("properties")
    {
        map.entry("additionalProperties")
            .or_insert(Value::Bool(false));
    }

    for child in map.values_mut() {
        normalize(child);
    }
}

/// If every branch is a single-value `const`/`enum` with no `properties`, return
/// `(json-type, values)` for the flattened form; otherwise `None`.
fn flatten_const_enum(branches: &[Value]) -> Option<(String, Vec<Value>)> {
    let mut values = Vec::new();
    let mut ty: Option<String> = None;
    for branch in branches {
        let obj = branch.as_object()?;
        if obj.contains_key("properties") {
            return None;
        }
        let value = obj.get("const").or_else(|| {
            obj.get("enum")
                .and_then(Value::as_array)
                .filter(|a| a.len() == 1)
                .map(|a| &a[0])
        })?;
        ty.get_or_insert_with(|| json_type_name(value));
        values.push(value.clone());
    }
    if values.is_empty() {
        return None;
    }
    Some((ty.unwrap_or_else(|| "string".into()), values))
}

/// The JSON Schema `type` name for a scalar value.
fn json_type_name(value: &Value) -> String {
    match value {
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        _ => "string",
    }
    .to_owned()
}
