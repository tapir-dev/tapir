// PROTOTYPE — THROWAWAY. Answers tapir-dev/tapir#5: "Design the typed Tool
// trait and #[tool] derive macro". Not production code; discard with the branch.
// Run: cargo run --example proto_tool_trait
//
// Question this prototype settles: does the typed `Tool` trait + object-safe
// `ErasedTool` erase + `#[tool]` macro shape actually feel right, and does a
// useful tool land in well under 15 lines?
//
// What is real here: the trait/erase shapes, the dispatch-boundary validation,
// the operator-vs-model error split, and the portable-schema post-processing.
// What is faked: `ContentPart` / `ToolDefinition` are 1-field stand-ins for the
// `tapir_provider` types (kept local so this runs anywhere), and the `#[tool]`
// macro is hand-expanded in `mod expanded` (a throwaway file can't host a real
// proc-macro crate).

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

// ===========================================================================
// Provider stand-ins. In the real crate these ARE `tapir_provider::ContentPart`
// and `tapir_provider::ToolDefinition`, reused verbatim (see #2's surface map).
// ===========================================================================

/// One block of tool output. The real enum also has Image/ToolCall/Thinking.
#[derive(Debug, Clone)]
pub enum ContentPart {
    Text(String),
}

/// Provider-neutral tool description carried verbatim to the backend. The schema
/// is shipped as-is and never validated by the provider — portability is ours.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ===========================================================================
// Tool-system types.
// ===========================================================================

/// Scheduling class. Governs how the run loop batches a tool against others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concurrency {
    /// Parallel-safe (a read): may run alongside any other tool.
    Safe,
    /// Serialize against other Exclusive tools (a mutation).
    Exclusive,
}

/// Per-call context handed to `execute`. The seam for cancellation and (later)
/// a handle back to the agent/session. Kept tiny on purpose.
pub struct ToolCtx {
    pub call_id: String,
    cancel: Arc<AtomicBool>,
}

impl ToolCtx {
    fn new(call_id: impl Into<String>) -> Self {
        Self { call_id: call_id.into(), cancel: Arc::new(AtomicBool::new(false)) }
    }
    /// Long-running tools poll this between steps.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A streaming partial emitted mid-`execute`. The run loop forwards these onto
/// the agent's `Stream<AgentEvent>`. A single flat enum (not a per-tool
/// associated type) keeps `ErasedTool` object-safe.
#[derive(Debug, Clone)]
pub enum ToolUpdate {
    /// Human-readable progress line.
    Progress(String),
    /// Structured partial output.
    Partial(Value),
}

/// A tool's successful result, normalized to provider content.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: Vec<ContentPart>,
}

impl From<String> for ToolOutput {
    fn from(s: String) -> Self {
        Self { content: vec![ContentPart::Text(s)] }
    }
}
impl From<&str> for ToolOutput {
    fn from(s: &str) -> Self {
        Self::from(s.to_owned())
    }
}

/// The framework error every author error normalizes into at the dispatch
/// boundary. Two channels, deliberately separate:
/// - `model_message` rides back in the `ToolResult` (`is_error: true`) and is
///   the only text the model ever sees — it drives the retry.
/// - `operator_detail` is for logs/telemetry and is NEVER shown to the model.
#[derive(Debug)]
pub struct ToolError {
    pub model_message: String,
    pub operator_detail: Option<Box<dyn Error + Send + Sync>>,
    /// Whether the run loop may auto-retry (bounded). Default false.
    pub retryable: bool,
}

impl ToolError {
    /// Model-visible error, no operator detail.
    pub fn model(msg: impl Into<String>) -> Self {
        Self { model_message: msg.into(), operator_detail: None, retryable: false }
    }
    /// Attach operator-only detail (kept out of the model's view).
    #[must_use]
    pub fn with_operator(mut self, e: impl Into<Box<dyn Error + Send + Sync>>) -> Self {
        self.operator_detail = Some(e.into());
        self
    }
    #[must_use]
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }
    /// The one error the boundary mints itself: raw JSON failed to become `Args`.
    fn invalid_args(tool: &str, e: serde_json::Error) -> Self {
        Self::model(format!("invalid arguments for `{tool}`: {e}"))
    }
}

// Escape hatches so a `String`/`&str` author error normalizes for free.
impl From<String> for ToolError {
    fn from(s: String) -> Self {
        Self::model(s)
    }
}
impl From<&str> for ToolError {
    fn from(s: &str) -> Self {
        Self::model(s)
    }
}

/// The typed tool an author writes. All the ergonomics live here; the object-safe
/// half is derived for free below.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Argument type. `Deserialize` makes it the validator; `JsonSchema` derives
    /// the parameter schema — the two can never drift.
    type Args: DeserializeOwned + JsonSchema + Send;
    /// Whatever the tool produces; anything convertible into provider content.
    type Output: Into<ToolOutput> + Send;
    /// Author-chosen error, normalized to `ToolError` only at the boundary.
    type Error: Into<ToolError> + Send;

    fn name(&self) -> &str;
    fn description(&self) -> &str;

    /// Portable JSON Schema for `Args`. Default derives + post-processes it, so
    /// an author never writes schema by hand.
    fn parameters(&self) -> Value {
        portable_schema::<Self::Args>()
    }

    /// Metadata. Safe defaults: serialize (Exclusive) and assume mutation.
    fn concurrency(&self) -> Concurrency {
        Concurrency::Exclusive
    }
    fn read_only(&self) -> bool {
        false
    }

    /// Run the tool. `on_update` streams partials; `ctx` carries cancellation.
    async fn execute(
        &self,
        args: Self::Args,
        ctx: &ToolCtx,
        on_update: &mut (dyn FnMut(ToolUpdate) + Send),
    ) -> Result<Self::Output, Self::Error>;
}

/// The object-safe face the run loop stores as `Arc<dyn ErasedTool>`. Authors
/// never implement this — the blanket impl erases every `Tool` into it.
#[async_trait]
pub trait ErasedTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Provider-ready definition (name + description + portable schema).
    fn definition(&self) -> ToolDefinition;
    fn concurrency(&self) -> Concurrency;
    fn read_only(&self) -> bool;

    /// Dispatch a raw model-generated call. This is THE boundary: it validates
    /// JSON into `Args`, runs the author code, and normalizes any author error
    /// into a `ToolError`. Everything above it deals only in erased tools.
    async fn invoke(
        &self,
        raw_args: Value,
        ctx: &ToolCtx,
        on_update: &mut (dyn FnMut(ToolUpdate) + Send),
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
        ToolDefinition {
            name: Tool::name(self).to_owned(),
            description: Tool::description(self).to_owned(),
            input_schema: self.parameters(),
        }
    }
    fn concurrency(&self) -> Concurrency {
        Tool::concurrency(self)
    }
    fn read_only(&self) -> bool {
        Tool::read_only(self)
    }
    async fn invoke(
        &self,
        raw_args: Value,
        ctx: &ToolCtx,
        on_update: &mut (dyn FnMut(ToolUpdate) + Send),
    ) -> Result<ToolOutput, ToolError> {
        // 1. validate JSON -> typed Args (a parse failure IS a validation error)
        let args: T::Args = serde_json::from_value(raw_args)
            .map_err(|e| ToolError::invalid_args(Tool::name(self), e))?;
        // 2. run author code; 3. normalize author Output/Error to framework types
        let out = self.execute(args, ctx, on_update).await.map_err(Into::into)?;
        Ok(out.into())
    }
}

// ===========================================================================
// Portable schema: what the `#[tool]` derive does to raw schemars output to
// survive all three provider strict subsets (see #3's dialect survey).
// ===========================================================================

/// Derive a portable schema for `T`: inline refs, drop meta noise, flatten
/// fieldless enums (`oneOf`-of-`const` -> flat `enum`), and set
/// `additionalProperties: false` on objects. (Per-provider nullability/required
/// differences would be a send-time pass; this is the one canonical schema.)
fn portable_schema<T: JsonSchema>() -> Value {
    let settings = SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true; // no $ref — Gemini/Anthropic strict reject
        s.meta_schema = None; // drop $schema
    });
    let schema = SchemaGenerator::new(settings).into_root_schema_for::<T>();
    let mut v = serde_json::to_value(schema).expect("schema serializes to JSON");
    normalize(&mut v);
    if let Some(obj) = v.as_object_mut() {
        obj.remove("title");
        obj.remove("description");
        obj.remove("$schema");
    }
    v
}

/// Recursive post-processing pass.
fn normalize(v: &mut Value) {
    if let Value::Array(arr) = v {
        for item in arr {
            normalize(item);
        }
        return;
    }
    let Value::Object(map) = v else { return };
    map.remove("$schema");

    // Flatten a fieldless enum: schemars renders it as oneOf/anyOf of single
    // `const` branches; every strict subset rejects oneOf/const, so collapse to
    // one flat `{ "type": ..., "enum": [...] }`.
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

    // additionalProperties: false on every object with properties.
    if map.get("type").and_then(Value::as_str) == Some("object")
        && map.contains_key("properties")
    {
        map.entry("additionalProperties").or_insert(Value::Bool(false));
    }

    for (_k, child) in map.iter_mut() {
        normalize(child);
    }
}

/// If every branch is a single-value `const`/`enum` with no `properties`, return
/// `(json-type, values)` for the flattened form; else `None`.
fn flatten_const_enum(branches: &[Value]) -> Option<(String, Vec<Value>)> {
    let mut values = Vec::new();
    let mut ty: Option<String> = None;
    for b in branches {
        let obj = b.as_object()?;
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

fn json_type_name(v: &Value) -> String {
    match v {
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        _ => "string",
    }
    .to_owned()
}

// ===========================================================================
// A hand-written tool. Proves the trait is usable directly, no macro involved.
// ===========================================================================

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WeatherArgs {
    /// City name or coordinates.
    city: String,
    /// Temperature unit; defaults to Celsius.
    #[serde(default)]
    units: Units,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Units {
    #[default]
    Celsius,
    Fahrenheit,
}

/// Author-chosen error type.
#[derive(Debug)]
enum WeatherError {
    EmptyCity,
}

impl From<WeatherError> for ToolError {
    fn from(e: WeatherError) -> Self {
        match e {
            // model sees a terse, retry-able hint; operator log keeps the detail.
            WeatherError::EmptyCity => ToolError::model("city must not be empty")
                .with_operator("WeatherError::EmptyCity at geo lookup"),
        }
    }
}

struct GetWeather;

#[async_trait]
impl Tool for GetWeather {
    type Args = WeatherArgs;
    type Output = String;
    type Error = WeatherError;

    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Get current weather for a location"
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Safe
    }
    fn read_only(&self) -> bool {
        true
    }

    async fn execute(
        &self,
        args: WeatherArgs,
        _ctx: &ToolCtx,
        on_update: &mut (dyn FnMut(ToolUpdate) + Send),
    ) -> Result<String, WeatherError> {
        on_update(ToolUpdate::Progress(format!("looking up {}", args.city)));
        if args.city.is_empty() {
            return Err(WeatherError::EmptyCity);
        }
        let (temp, unit) = match args.units {
            Units::Celsius => (18, 'C'),
            Units::Fahrenheit => (64, 'F'),
        };
        Ok(format!("It is {temp}\u{b0}{unit} and clear in {}.", args.city))
    }
}

// ===========================================================================
// Sketched `#[tool]` output. The author writes the fn; the macro emits the rest.
//
// Author source (target: a useful tool in well under 15 lines):
//
//     /// Search the web and return the top results.
//     #[tool]
//     async fn web_search(args: SearchArgs, ctx: &ToolCtx) -> Result<String, SearchError> {
//         Ok(format!("results for {}", args.query))
//     }
//
// What `#[tool]` reads and emits:
//   - fn doc comment       -> description
//   - fn name              -> tool name ("web_search")
//   - the `args:` type      -> `type Args` (schema derived from it; per-field docs
//                             come from the Args struct, where schemars reads them)
//   - a `ctx: &ToolCtx` param PRESENT -> contextual dispatch (below);
//     ABSENT -> generated execute takes the same shape but ignores ctx.
//   - it emits a unit struct shadowing the fn name + this `Tool` impl, so
//     `.tool(web_search)` registers a value.
// ===========================================================================

mod expanded {
    use super::*;

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct SearchArgs {
        /// The search query.
        pub query: String,
        /// Max results to return.
        #[serde(default)]
        pub limit: Option<u32>,
    }

    #[derive(Debug)]
    pub enum SearchError {}
    impl From<SearchError> for ToolError {
        fn from(_: SearchError) -> Self {
            ToolError::model("unreachable")
        }
    }

    // ---- generated by #[tool] ----
    #[allow(non_camel_case_types)]
    pub struct web_search;

    #[async_trait]
    impl Tool for web_search {
        type Args = SearchArgs;
        type Output = String;
        type Error = SearchError;

        fn name(&self) -> &str {
            "web_search"
        }
        fn description(&self) -> &str {
            "Search the web and return the top results."
        }

        async fn execute(
            &self,
            args: SearchArgs,
            ctx: &ToolCtx, // <- present because the source fn had a ctx param
            _on_update: &mut (dyn FnMut(ToolUpdate) + Send),
        ) -> Result<String, SearchError> {
            // body of the author's fn, verbatim
            let n = args.limit.unwrap_or(3);
            Ok(format!("[{}] {n} results for {:?}", ctx.call_id, args.query))
        }
    }
}

// ===========================================================================
// Dispatch harness. Everything below deals only in `Arc<dyn ErasedTool>`.
// ===========================================================================

async fn dispatch(
    tools: &[Arc<dyn ErasedTool>],
    name: &str,
    raw_args: Value,
    ctx: &ToolCtx,
    on_update: &mut (dyn FnMut(ToolUpdate) + Send),
) {
    println!("\n>>> call {name}({raw_args})");
    let Some(tool) = tools.iter().find(|t| t.name() == name) else {
        println!("    no such tool");
        return;
    };
    match tool.invoke(raw_args, ctx, on_update).await {
        Ok(out) => println!("    ok  -> ToolResult(is_error=false) {:?}", out.content),
        Err(e) => {
            // this is how the run loop would build the ToolResult:
            println!("    err -> ToolResult(is_error=true) model={:?}", e.model_message);
            if let Some(op) = &e.operator_detail {
                println!("           operator-only (never sent to model): {op}");
            }
            println!("           retryable={}", e.retryable);
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let tools: Vec<Arc<dyn ErasedTool>> =
        vec![Arc::new(GetWeather), Arc::new(expanded::web_search)];

    println!("=== registered tool definitions (schema is what ships to providers) ===");
    for t in &tools {
        let def = t.definition();
        println!(
            "\n- {} [{:?}, read_only={}]: {}",
            def.name,
            t.concurrency(),
            t.read_only(),
            def.description
        );
        println!(
            "  schema: {}",
            serde_json::to_string(&def.input_schema).unwrap()
        );
    }

    println!("\n=== dispatch through Arc<dyn ErasedTool> ===");
    let ctx = ToolCtx::new("call_abc");
    let mut sink = |u: ToolUpdate| println!("    [update] {u:?}");

    // 1. valid call
    dispatch(&tools, "get_weather", json!({"city": "Paris", "units": "fahrenheit"}), &ctx, &mut sink).await;
    // 2. boundary validation failure (missing required `city`)
    dispatch(&tools, "get_weather", json!({"units": "celsius"}), &ctx, &mut sink).await;
    // 3. author error, normalized -> model vs operator channels
    dispatch(&tools, "get_weather", json!({"city": "", "units": "celsius"}), &ctx, &mut sink).await;
    // 4. the macro-shaped tool, dispatched identically
    dispatch(&tools, "web_search", json!({"query": "tapir sdk", "limit": 5}), &ctx, &mut sink).await;

    println!("\n(cancellation seam present: ctx.is_cancelled() == {})", ctx.is_cancelled());
}
