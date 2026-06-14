//! Anthropic **Messages** API (`/v1/messages`, SSE).
//!
//! `system` is a top-level field; tool calls are `tool_use` content blocks and
//! their results come back as a `user` message of `tool_result` blocks — so we
//! coalesce consecutive [`Step::ToolResult`]s into one message to keep the
//! required user/assistant alternation.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{
    Creds, RoundOutcome, SseAccumulator, Step, base_url, send_err, status_err,
};
use crate::message::{RoundError, ToolCall, ToolDef, TurnEvent, Usage};

const VERSION: &str = "2023-06-01";

/// Build the `messages` array, coalescing consecutive tool results.
pub fn build_messages(history: &[Step]) -> Vec<Value> {
    let mut msgs = Vec::new();
    let mut i = 0;
    while i < history.len() {
        match &history[i] {
            Step::User { text, images } => {
                let mut content = vec![json!({ "type": "text", "text": text })];
                for img in images {
                    content.push(json!({
                        "type": "image",
                        "source": { "type": "base64", "media_type": img.mime, "data": img.data },
                    }));
                }
                msgs.push(json!({ "role": "user", "content": content }));
                i += 1;
            }
            Step::Assistant { text, tool_calls, .. } => {
                let mut content = Vec::new();
                if !text.is_empty() {
                    content.push(json!({ "type": "text", "text": text }));
                }
                for tc in tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": tc.call_id,
                        "name": tc.name,
                        "input": tc.args,
                    }));
                }
                msgs.push(json!({ "role": "assistant", "content": content }));
                i += 1;
            }
            Step::ToolResult { .. } => {
                // Gather the run of tool results into a single user message.
                let mut blocks = Vec::new();
                while let Some(Step::ToolResult {
                    call_id,
                    output,
                    is_error,
                    ..
                }) = history.get(i)
                {
                    blocks.push(json!({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                        "is_error": is_error,
                    }));
                    i += 1;
                }
                msgs.push(json!({ "role": "user", "content": blocks }));
            }
        }
    }
    msgs
}

/// Folds an Anthropic Messages SSE stream into text / thinking / tool calls /
/// usage. Pure (no I/O) so the parse can be unit-tested with captured events;
/// `apply` returns the text/thinking deltas the caller streams.
#[derive(Default)]
struct AnthropicAcc {
    usage: Usage,
    text: String,
    thinking: String,
    /// Tool-use blocks keyed by content-block index: (id, name, partial json).
    blocks: BTreeMap<usize, (String, String, String)>,
}

impl SseAccumulator for AnthropicAcc {
    fn apply(&mut self, ev: &Value) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        match ev.get("type").and_then(|t| t.as_str()) {
            Some("message_start") => {
                if let Some(u) = ev.pointer("/message/usage") {
                    self.usage.input = u
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    self.usage.cache_read = u
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                }
            }
            Some("content_block_start") => {
                let idx = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0)
                    as usize;
                if let Some(cb) = ev.get("content_block")
                    && cb.get("type").and_then(|t| t.as_str())
                        == Some("tool_use")
                {
                    let id = cb
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = cb
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.blocks.insert(idx, (id, name, String::new()));
                }
            }
            Some("content_block_delta") => {
                let idx = ev.get("index").and_then(|v| v.as_u64()).unwrap_or(0)
                    as usize;
                let delta = ev.get("delta");
                match delta.and_then(|d| d.get("type")).and_then(|t| t.as_str())
                {
                    Some("text_delta") => {
                        if let Some(t) = delta
                            .and_then(|d| d.get("text"))
                            .and_then(|v| v.as_str())
                        {
                            self.text.push_str(t);
                            out.push(TurnEvent::Text { delta: t.to_string() });
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(t) = delta
                            .and_then(|d| d.get("thinking"))
                            .and_then(|v| v.as_str())
                        {
                            self.thinking.push_str(t);
                            out.push(TurnEvent::Thinking {
                                delta: t.to_string(),
                            });
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(p) = delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|v| v.as_str())
                            && let Some(slot) = self.blocks.get_mut(&idx)
                        {
                            slot.2.push_str(p);
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(o) =
                    ev.pointer("/usage/output_tokens").and_then(|v| v.as_u64())
                {
                    self.usage.output = o;
                }
            }
            _ => {}
        }
        out
    }
}

impl AnthropicAcc {
    /// Finalize: total = input + cache_read + output, plus the assembled calls.
    fn finish(mut self) -> (Usage, String, String, Vec<ToolCall>) {
        self.usage.total =
            self.usage.input + self.usage.cache_read + self.usage.output;
        let tool_calls = self
            .blocks
            .into_values()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, args_json)| {
                let args = serde_json::from_str(&args_json)
                    .unwrap_or_else(|_| json!({}));
                ToolCall { call_id: id, name, args }
            })
            .collect();
        (self.usage, self.text, self.thinking, tool_calls)
    }
}

/// The `tools` array (Anthropic `input_schema` shape) for the active tool set.
fn tools_param(active: &[ToolDef]) -> Value {
    Value::Array(
        active
            .iter()
            .map(|d| {
                json!({
                    "name": d.name,
                    "description": d.description,
                    "input_schema": d.parameters.clone(),
                })
            })
            .collect(),
    )
}

pub async fn stream(
    base: Option<&str>,
    ctx: &super::RoundCtx<'_>,
    history: &[Step],
) -> Result<RoundOutcome, RoundError> {
    let super::RoundCtx {
        client,
        creds,
        model,
        instructions,
        tools,
        effort,
        tx,
        ..
    } = *ctx;
    let Creds::ApiKey { key } = creds else {
        return Err(RoundError { message: "missing API key".into() });
    };
    // `effort` is accepted for signature parity; Anthropic's thinking knob varies
    // by model version (`enabled`+budget vs `adaptive`+`output_config`), so we
    // leave it off to keep requests valid across all Anthropic models.
    let _ = effort;
    // An explicit endpoint (a custom provider) wins; else the fixed default.
    let base =
        base.map(str::to_string).unwrap_or_else(|| base_url("anthropic"));

    // `max_tokens` is required by the Messages API.
    let max_tokens = crate::catalog::models::get("anthropic", model)
        .map(|m| m.max_output)
        .filter(|&m| m > 0)
        .unwrap_or(8_192);

    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "system": instructions,
        "messages": build_messages(history),
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = tools_param(tools);
    }
    // `TAPIR_CACHE_RETENTION=long` → cache the (large) system prompt with a 1h
    // TTL; the extended TTL needs an opt-in beta header.
    let long_cache = super::long_cache_retention();
    if long_cache {
        body["system"] = json!([{
            "type": "text",
            "text": instructions,
            "cache_control": { "type": "ephemeral", "ttl": "1h" },
        }]);
    }

    let mut req = client
        .post(format!("{base}/v1/messages"))
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("x-api-key", key.as_str())
        .header("anthropic-version", VERSION);
    if long_cache {
        req = req.header("anthropic-beta", "extended-cache-ttl-2025-04-11");
    }
    let resp = req.json(&body).send().await.map_err(send_err)?;
    if !resp.status().is_success() {
        return Err(status_err(resp).await);
    }

    let mut acc = AnthropicAcc::default();
    super::drive_sse(resp, &mut acc, tx).await?;

    let (usage, text, thinking, tool_calls) = acc.finish();
    Ok(RoundOutcome {
        usage,
        tool_calls: tool_calls.clone(),
        assistant: Step::Assistant { text, thinking, tool_calls, raw: None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_tool_results_into_one_user_message() {
        let msgs = build_messages(&[
            Step::User { text: "go".into(), images: vec![] },
            Step::Assistant {
                text: "doing".into(),
                thinking: String::new(),
                tool_calls: vec![
                    ToolCall {
                        call_id: "a".into(),
                        name: "read".into(),
                        args: json!({}),
                    },
                    ToolCall {
                        call_id: "b".into(),
                        name: "ls".into(),
                        args: json!({}),
                    },
                ],
                raw: None,
            },
            Step::ToolResult {
                call_id: "a".into(),
                name: "read".into(),
                output: "A".into(),
                is_error: false,
            },
            Step::ToolResult {
                call_id: "b".into(),
                name: "ls".into(),
                output: "B".into(),
                is_error: true,
            },
        ]);
        assert_eq!(
            msgs.len(),
            3,
            "user, assistant, then one merged tool-result user"
        );
        assert_eq!(msgs[1]["content"][0]["type"], "text");
        assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][1]["is_error"], true);
    }

    #[test]
    fn accumulates_messages_stream_into_outcome() {
        let mut acc = AnthropicAcc::default();
        acc.apply(&json!({
            "type": "message_start",
            "message": { "usage": { "input_tokens": 20, "cache_read_input_tokens": 5 } }
        }));
        let d = acc.apply(&json!({
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "text_delta", "text": "Hi" }
        }));
        assert!(
            matches!(d.as_slice(), [TurnEvent::Text { delta }] if delta == "Hi")
        );
        let d = acc.apply(&json!({
            "type": "content_block_delta", "index": 0,
            "delta": { "type": "thinking_delta", "thinking": "...." }
        }));
        assert!(
            matches!(d.as_slice(), [TurnEvent::Thinking { delta }] if delta == "....")
        );
        // A tool_use block: it starts, then its input JSON streams in pieces.
        acc.apply(&json!({
            "type": "content_block_start", "index": 1,
            "content_block": { "type": "tool_use", "id": "toolu_1", "name": "read" }
        }));
        acc.apply(&json!({
            "type": "content_block_delta", "index": 1,
            "delta": { "type": "input_json_delta", "partial_json": "{\"path\":" }
        }));
        acc.apply(&json!({
            "type": "content_block_delta", "index": 1,
            "delta": { "type": "input_json_delta", "partial_json": "\"a\"}" }
        }));
        acc.apply(&json!({ "type": "message_delta", "usage": { "output_tokens": 7 } }));

        let (usage, text, thinking, calls) = acc.finish();
        assert_eq!(text, "Hi");
        assert_eq!(thinking, "....");
        assert_eq!(usage.input, 20);
        assert_eq!(usage.cache_read, 5);
        assert_eq!(usage.output, 7);
        assert_eq!(usage.total, 32, "total = input + cache_read + output");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "toolu_1");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].args["path"], "a");
    }

    #[test]
    fn user_image_becomes_base64_source() {
        use crate::message::Image;
        let msgs = build_messages(&[Step::User {
            text: "look".into(),
            images: vec![Image {
                mime: "image/png".into(),
                data: "QUJD".into(),
            }],
        }]);
        assert_eq!(msgs[0]["content"][1]["type"], "image");
        assert_eq!(msgs[0]["content"][1]["source"]["data"], "QUJD");
        assert_eq!(msgs[0]["content"][1]["source"]["media_type"], "image/png");
    }
}
