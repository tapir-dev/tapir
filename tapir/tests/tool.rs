// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The typed tool system, standalone: an annotated `async fn` and a hand-written
//! `Tool` both erase to `Arc<dyn ErasedTool>`, report their concurrency class,
//! validate raw JSON at the dispatch boundary, split author errors into the
//! model/operator channels, and emit the canonical portable schema.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tapir::tool::{Concurrency, ErasedTool, ToolCtx, ToolError, ToolUpdate};
use tapir::{__private::async_trait, tool};

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

/// Author-chosen error type, normalized to `ToolError` at the boundary.
#[derive(Debug)]
enum WeatherError {
    EmptyCity,
}

impl From<WeatherError> for ToolError {
    fn from(err: WeatherError) -> Self {
        match err {
            WeatherError::EmptyCity => {
                ToolError::model("city must not be empty")
                    .with_operator("WeatherError::EmptyCity at geo lookup")
            }
        }
    }
}

/// A hand-written tool: proves the trait is usable with no macro involved.
struct GetWeather;

#[async_trait]
impl tapir::tool::Tool for GetWeather {
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

    async fn execute(
        &self,
        args: WeatherArgs,
        _ctx: &ToolCtx,
        on_update: &mut tapir::tool::UpdateSink<'_>,
    ) -> Result<String, WeatherError> {
        on_update(ToolUpdate::Progress(format!("looking up {}", args.city)));
        if args.city.is_empty() {
            return Err(WeatherError::EmptyCity);
        }
        let (temp, unit) = match args.units {
            Units::Celsius => (18, 'C'),
            Units::Fahrenheit => (64, 'F'),
        };
        Ok(format!("It is {temp}{unit} and clear in {}.", args.city))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct EchoArgs {
    /// The message to echo.
    message: String,
}

// A macro-authored tool with a `ctx` parameter (contextual), read-only.
#[tool(read_only)]
/// Echo the message back, tagged with the call id.
async fn echo(args: EchoArgs, ctx: &ToolCtx) -> String {
    format!("[{}] {}", ctx.call_id(), args.message)
}

fn sink() -> impl FnMut(ToolUpdate) + Send {
    |_u| {}
}

#[tokio::test]
async fn erases_and_reports_concurrency() {
    let tools: Vec<Arc<dyn ErasedTool>> =
        vec![Arc::new(GetWeather), Arc::new(echo)];

    assert_eq!(tools[0].name(), "get_weather");
    assert_eq!(tools[0].concurrency(), Concurrency::Safe);
    assert!(tools[0].read_only());

    assert_eq!(tools[1].name(), "echo");
    assert_eq!(
        tools[1].description(),
        "Echo the message back, tagged with the call id."
    );
    assert_eq!(tools[1].concurrency(), Concurrency::Safe);
    assert!(tools[1].read_only());
}

#[tokio::test]
async fn invoke_validates_and_runs() {
    let tool = GetWeather;
    let ctx = ToolCtx::new("call_1");
    let mut on_update = sink();

    let out = ErasedTool::invoke(
        &tool,
        json!({"city": "Paris", "units": "fahrenheit"}),
        &ctx,
        &mut on_update,
    )
    .await
    .expect("valid call succeeds");
    assert_eq!(
        out.content,
        vec![tapir_provider::ContentPart::text(
            "It is 64F and clear in Paris."
        )]
    );
}

#[tokio::test]
async fn invoke_surfaces_bad_args() {
    let tool = GetWeather;
    let ctx = ToolCtx::new("call_2");
    let mut on_update = sink();

    // Missing required `city`.
    let err = ErasedTool::invoke(
        &tool,
        json!({"units": "celsius"}),
        &ctx,
        &mut on_update,
    )
    .await
    .expect_err("missing city fails validation");
    assert!(
        err.model_message
            .contains("invalid arguments for `get_weather`"),
        "unexpected message: {}",
        err.model_message
    );
    assert!(err.operator_detail.is_none());
}

#[tokio::test]
async fn invoke_splits_author_error() {
    let tool = GetWeather;
    let ctx = ToolCtx::new("call_3");
    let mut on_update = sink();

    let err = ErasedTool::invoke(
        &tool,
        json!({"city": "", "units": "celsius"}),
        &ctx,
        &mut on_update,
    )
    .await
    .expect_err("empty city is an author error");
    assert_eq!(err.model_message, "city must not be empty");
    assert_eq!(
        err.operator_detail
            .expect("operator detail attached")
            .to_string(),
        "WeatherError::EmptyCity at geo lookup"
    );
}

#[tokio::test]
async fn contextual_tool_sees_call_id() {
    let ctx = ToolCtx::new("abc");
    let mut on_update = sink();
    let out = ErasedTool::invoke(
        &echo,
        json!({"message": "hi"}),
        &ctx,
        &mut on_update,
    )
    .await
    .expect("echo succeeds");
    assert_eq!(
        out.content,
        vec![tapir_provider::ContentPart::text("[abc] hi")]
    );
}

#[test]
fn emits_canonical_portable_schema() {
    let schema = ErasedTool::definition(&GetWeather).input_schema;
    let expected = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "city": {
                "type": "string",
                "description": "City name or coordinates."
            },
            "units": {
                "type": "string",
                "description": "Temperature unit; defaults to Celsius.",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["city"]
    });
    assert_eq!(schema, expected, "schema drifted from golden");
}
