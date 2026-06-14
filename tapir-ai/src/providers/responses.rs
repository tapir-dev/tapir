//! GitHub Copilot — the OpenAI **Responses** API (`/responses`, SSE).
//!
//! Ported from tapir's original single-provider `agent.rs`. The conversation is
//! re-serialized from the neutral [`Step`] history into Responses `input` items
//! each round; for the assistant turn we echo Copilot's verbatim output items
//! (kept in `Step::Assistant::raw`) so encrypted reasoning round-trips intact.

use serde_json::{Value, json};

use super::{
    Creds, RoundOutcome, SseAccumulator, Step, base_url, model_reasons,
    send_err, status_err,
};
use crate::auth::copilot;
use crate::message::{RoundError, ToolCall, ToolDef, TurnEvent, Usage};

/// Build the Responses `input` items from the neutral history.
pub fn build_input(history: &[Step]) -> Vec<Value> {
    let mut items = Vec::new();
    for step in history {
        match step {
            Step::User { text, images } => {
                let mut content = vec![json!({ "type": "input_text", "text": text })];
                for img in images {
                    content.push(json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", img.mime, img.data),
                    }));
                }
                items.push(json!({ "role": "user", "content": content }));
            }
            Step::Assistant { text, raw, .. } => match raw {
                // Echo Copilot's exact items (text + encrypted reasoning + calls).
                Some(raw) => items.extend(raw.iter().cloned()),
                None => items.push(json!({
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": text, "annotations": [] }],
                })),
            },
            Step::ToolResult { call_id, output, .. } => {
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
        }
    }
    items
}

/// The `tools` array (Responses function shape) for the active tool set.
fn tools_param(active: &[ToolDef]) -> Value {
    Value::Array(
        active
            .iter()
            .map(|d| {
                json!({
                    "type": "function",
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters.clone(),
                    "strict": false,
                })
            })
            .collect(),
    )
}

/// Folds a Responses SSE stream into text / thinking / the verbatim output
/// items (from which tool calls and the echo `raw` are derived). Pure (no I/O)
/// so the parse can be unit-tested; `apply` returns the deltas to stream.
#[derive(Default)]
struct ResponsesAcc {
    usage: Usage,
    text: String,
    thinking: String,
    output_items: Vec<Value>,
}

impl SseAccumulator for ResponsesAcc {
    fn apply(&mut self, event: &Value) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        match event.get("type").and_then(|t| t.as_str()) {
            Some("response.output_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(|d| d.as_str())
                {
                    self.text.push_str(delta);
                    out.push(TurnEvent::Text { delta: delta.to_string() });
                }
            }
            Some("response.reasoning_summary_text.delta")
            | Some("response.reasoning_text.delta") => {
                if let Some(delta) = event.get("delta").and_then(|d| d.as_str())
                {
                    self.thinking.push_str(delta);
                    out.push(TurnEvent::Thinking { delta: delta.to_string() });
                }
            }
            Some("response.completed") => {
                self.usage = parse_usage(event);
                if let Some(items) = event
                    .get("response")
                    .and_then(|r| r.get("output"))
                    .and_then(|o| o.as_array())
                {
                    self.output_items = items.clone();
                }
            }
            _ => {}
        }
        out
    }
}

pub async fn stream(
    base: Option<&str>,
    ctx: &super::RoundCtx<'_>,
    history: &[Step],
) -> Result<RoundOutcome, RoundError> {
    let super::RoundCtx {
        client,
        provider,
        creds,
        model,
        instructions,
        tools,
        effort,
        tx,
    } = *ctx;
    let is_copilot = provider == "copilot";
    // Copilot uses its short-lived OAuth token; OpenAI uses the API key.
    let bearer = match creds {
        Creds::Copilot { access } => access.as_str(),
        Creds::ApiKey { key } => key.as_str(),
    };
    // An explicit endpoint (a custom provider) wins; else Copilot's base comes
    // from the token's proxy endpoint and OpenAI's is fixed.
    let base = match base {
        Some(b) => b.to_string(),
        None if is_copilot => copilot::api_base(bearer),
        None => base_url(provider).to_string(),
    };

    let mut body = json!({
        "model": model,
        "instructions": instructions,
        "input": build_input(history),
        "stream": true,
        "store": false,
    });
    if !tools.is_empty() {
        body["tools"] = tools_param(tools);
    }
    // `TAPIR_CACHE_RETENTION=long` → extend OpenAI's prompt cache to 24h.
    if super::long_cache_retention() {
        body["prompt_cache_retention"] = json!("24h");
    }
    // Copilot accepts reasoning on its models; for OpenAI gate it on the model
    // (non-reasoning models reject a `reasoning` block).
    if let Some(effort) = effort
        && (is_copilot || model_reasons(provider, model))
    {
        body["reasoning"] = json!({ "effort": effort, "summary": "auto" });
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    let mut req = client
        .post(format!("{base}/responses"))
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {bearer}"));
    if is_copilot {
        req = req
            .header(reqwest::header::USER_AGENT, copilot::USER_AGENT)
            .header("Editor-Version", copilot::EDITOR_VERSION)
            .header("Editor-Plugin-Version", copilot::PLUGIN_VERSION)
            .header("Copilot-Integration-Id", copilot::INTEGRATION_ID)
            .header("X-Initiator", "user")
            .header("Openai-Intent", "conversation-edits");
    }
    let resp = req.json(&body).send().await.map_err(send_err)?;
    if !resp.status().is_success() {
        return Err(status_err(resp).await);
    }

    let mut acc = ResponsesAcc::default();
    super::drive_sse(resp, &mut acc, tx).await?;

    let tool_calls = extract_tool_calls(&acc.output_items);
    Ok(RoundOutcome {
        usage: acc.usage,
        tool_calls: tool_calls.clone(),
        assistant: Step::Assistant {
            text: acc.text,
            thinking: acc.thinking,
            tool_calls,
            raw: Some(acc.output_items),
        },
    })
}

fn extract_tool_calls(output_items: &[Value]) -> Vec<ToolCall> {
    output_items
        .iter()
        .filter(|item| {
            item.get("type").and_then(|t| t.as_str()) == Some("function_call")
        })
        .filter_map(|item| {
            let call_id =
                item.get("call_id").and_then(|v| v.as_str())?.to_string();
            let name = item.get("name").and_then(|v| v.as_str())?.to_string();
            let args_str =
                item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
            let args =
                serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
            Some(ToolCall { call_id, name, args })
        })
        .collect()
}

fn parse_usage(event: &Value) -> Usage {
    let u = event.get("response").and_then(|r| r.get("usage"));
    let get = |key: &str| -> u64 {
        u.and_then(|u| u.get(key)).and_then(|v| v.as_u64()).unwrap_or(0)
    };
    let cache_read = u
        .and_then(|u| u.get("input_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Usage {
        // OpenAI includes cached tokens in input_tokens; subtract to get fresh.
        input: get("input_tokens").saturating_sub(cache_read),
        output: get("output_tokens"),
        cache_read,
        total: get("total_tokens"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Image, Role};

    #[test]
    fn input_uses_responses_content_shapes() {
        let input = build_input(&[
            Step::User {
                text: "hi".into(),
                images: vec![Image {
                    mime: "image/png".into(),
                    data: "QUJD".into(),
                }],
            },
            Step::Assistant {
                text: "hello".into(),
                thinking: String::new(),
                tool_calls: vec![],
                raw: None,
            },
            Step::ToolResult {
                call_id: "c1".into(),
                name: "read".into(),
                output: "ok".into(),
                is_error: false,
            },
        ]);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert_eq!(
            input[0]["content"][1]["image_url"],
            "data:image/png;base64,QUJD"
        );
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "c1");
        let _ = Role::User;
    }

    #[test]
    fn assistant_raw_items_are_echoed_verbatim() {
        let raw =
            vec![json!({"type":"reasoning","x":1}), json!({"type":"message"})];
        let input = build_input(&[Step::Assistant {
            text: "ignored".into(),
            thinking: String::new(),
            tool_calls: vec![],
            raw: Some(raw.clone()),
        }]);
        assert_eq!(input, raw);
    }

    #[test]
    fn extracts_tool_calls_from_output() {
        let items = vec![
            json!({"type": "reasoning"}),
            json!({"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{\"path\":\"x\"}"}),
        ];
        let calls = extract_tool_calls(&items);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].args["path"], "x");
    }

    #[test]
    fn accumulates_responses_stream_into_outcome() {
        let mut acc = ResponsesAcc::default();
        let d = acc.apply(
            &json!({ "type": "response.output_text.delta", "delta": "Hi " }),
        );
        assert!(
            matches!(d.as_slice(), [TurnEvent::Text { delta }] if delta == "Hi ")
        );
        let d = acc.apply(&json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" }));
        assert!(
            matches!(d.as_slice(), [TurnEvent::Thinking { delta }] if delta == "think")
        );
        acc.apply(
            &json!({ "type": "response.output_text.delta", "delta": "there" }),
        );
        // `response.completed` carries usage and the verbatim output items.
        acc.apply(&json!({
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                    "input_tokens_details": { "cached_tokens": 30 }
                },
                "output": [
                    { "type": "reasoning" },
                    { "type": "function_call", "call_id": "c1", "name": "read", "arguments": "{\"path\":\"x\"}" }
                ]
            }
        }));
        assert_eq!(acc.text, "Hi there");
        assert_eq!(acc.thinking, "think");
        assert_eq!(acc.usage.input, 70, "cached subtracted from input");
        assert_eq!(acc.usage.output, 20);
        let calls = extract_tool_calls(&acc.output_items);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].args["path"], "x");
        assert_eq!(
            acc.output_items.len(),
            2,
            "verbatim items become the `raw` echo"
        );
    }

    #[test]
    fn usage_subtracts_cached_from_input() {
        let event = json!({
            "type": "response.completed",
            "response": { "usage": {
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                "input_tokens_details": { "cached_tokens": 30 }
            }}
        });
        let u = parse_usage(&event);
        assert_eq!(u.input, 70);
        assert_eq!(u.output, 20);
        assert_eq!(u.cache_read, 30);
        assert_eq!(u.total, 120);
    }
}
