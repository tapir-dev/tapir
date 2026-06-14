//! OpenAI **Chat Completions** (`/chat/completions`, SSE) — shared by OpenAI,
//! DeepSeek and OpenRouter (all OpenAI-compatible).
//!
//! The neutral history becomes a `messages` array; tool calls stream in
//! incrementally per `index` and are assembled before being returned.

use serde_json::{Value, json};

use super::{
    Creds, RoundOutcome, SseAccumulator, Step, base_url, model_reasons,
    send_err, status_err,
};
use crate::message::{RoundError, ToolCall, ToolDef, TurnEvent, Usage};

/// Build the `messages` array from the system prompt and neutral history.
pub fn build_messages(instructions: &str, history: &[Step]) -> Vec<Value> {
    let mut msgs = vec![json!({ "role": "system", "content": instructions })];
    for step in history {
        match step {
            Step::User { text, images } => {
                if images.is_empty() {
                    msgs.push(json!({ "role": "user", "content": text }));
                } else {
                    let mut parts =
                        vec![json!({ "type": "text", "text": text })];
                    for img in images {
                        parts.push(json!({
                            "type": "image_url",
                            "image_url": { "url": format!("data:{};base64,{}", img.mime, img.data) },
                        }));
                    }
                    msgs.push(json!({ "role": "user", "content": parts }));
                }
            }
            Step::Assistant { text, tool_calls, .. } => {
                let mut m = json!({ "role": "assistant", "content": text });
                if !tool_calls.is_empty() {
                    m["tool_calls"] = Value::Array(
                        tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.args.to_string(),
                                    },
                                })
                            })
                            .collect(),
                    );
                }
                msgs.push(m);
            }
            Step::ToolResult { call_id, output, .. } => {
                msgs.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }));
            }
        }
    }
    msgs
}

/// The `tools` array (Chat Completions function shape) for the active tool set.
fn tools_param(active: &[ToolDef]) -> Value {
    Value::Array(
        active
            .iter()
            .map(|d| {
                json!({
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description,
                        "parameters": d.parameters.clone(),
                    },
                })
            })
            .collect(),
    )
}

/// Streaming accumulator for a tool call assembled across deltas.
#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    args: String,
}

/// Folds a Chat Completions SSE stream into text / thinking / tool calls / usage.
/// Pure (no I/O) so the parse logic can be unit-tested by feeding it the parsed
/// events directly; `apply` returns the text/thinking deltas the caller streams.
#[derive(Default)]
struct ChatAcc {
    text: String,
    thinking: String,
    calls: Vec<PartialCall>,
    usage: Usage,
}

impl SseAccumulator for ChatAcc {
    fn apply(&mut self, ev: &Value) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
            self.usage = parse_usage(u);
        }
        let Some(delta) = ev.pointer("/choices/0/delta") else {
            return out;
        };
        if let Some(c) = delta.get("content").and_then(|c| c.as_str())
            && !c.is_empty()
        {
            self.text.push_str(c);
            out.push(TurnEvent::Text { delta: c.to_string() });
        }
        // DeepSeek / OpenRouter expose reasoning under varying keys.
        for key in ["reasoning_content", "reasoning"] {
            if let Some(r) = delta.get(key).and_then(|r| r.as_str())
                && !r.is_empty()
            {
                self.thinking.push_str(r);
                out.push(TurnEvent::Thinking { delta: r.to_string() });
            }
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0)
                    as usize;
                if self.calls.len() <= idx {
                    self.calls.resize_with(idx + 1, PartialCall::default);
                }
                let slot = &mut self.calls[idx];
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    slot.id = id.to_string();
                }
                if let Some(f) = tc.get("function") {
                    if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                        slot.name.push_str(n);
                    }
                    if let Some(a) = f.get("arguments").and_then(|v| v.as_str())
                    {
                        slot.args.push_str(a);
                    }
                }
            }
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
    let Creds::ApiKey { key } = creds else {
        return Err(RoundError { message: "missing API key".into() });
    };
    // An explicit endpoint (a custom provider) wins; else the id's default.
    let base = base.map(str::to_string).unwrap_or_else(|| base_url(provider));

    let mut body = json!({
        "model": model,
        "messages": build_messages(instructions, history),
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if !tools.is_empty() {
        body["tools"] = tools_param(tools);
    }
    // DeepSeek's reasoner thinks automatically and rejects an effort knob;
    // OpenAI (o-series / gpt-5) and OpenRouter take `reasoning_effort`.
    if let Some(effort) = effort
        && model_reasons(provider, model)
        && provider != "deepseek"
    {
        body["reasoning_effort"] = json!(effort);
    }

    let mut req = client
        .post(format!("{base}/chat/completions"))
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {key}"));
    if provider == "openrouter" {
        // OpenRouter asks integrators to identify themselves (optional headers).
        req = req
            .header("HTTP-Referer", "https://github.com/")
            .header("X-Title", "tapir");
    }
    let resp = req.json(&body).send().await.map_err(send_err)?;
    if !resp.status().is_success() {
        return Err(status_err(resp).await);
    }

    let mut acc = ChatAcc::default();
    super::drive_sse(resp, &mut acc, tx).await?;

    let tool_calls = finalize_calls(acc.calls);
    Ok(RoundOutcome {
        usage: acc.usage,
        tool_calls: tool_calls.clone(),
        assistant: Step::Assistant {
            text: acc.text,
            thinking: acc.thinking,
            tool_calls,
            raw: None,
        },
    })
}

fn finalize_calls(calls: Vec<PartialCall>) -> Vec<ToolCall> {
    calls
        .into_iter()
        .filter(|c| !c.name.is_empty())
        .enumerate()
        .map(|(i, c)| {
            let args =
                serde_json::from_str(&c.args).unwrap_or_else(|_| json!({}));
            let call_id =
                if c.id.is_empty() { format!("call_{i}") } else { c.id };
            ToolCall { call_id, name: c.name, args }
        })
        .collect()
}

fn parse_usage(u: &Value) -> Usage {
    let get = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let cache_read = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Usage {
        input: get("prompt_tokens").saturating_sub(cache_read),
        output: get("completion_tokens"),
        cache_read,
        total: get("total_tokens"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Image;

    #[test]
    fn serializes_messages_tools_and_results() {
        let msgs = build_messages(
            "SYS",
            &[
                Step::User {
                    text: "hi".into(),
                    images: vec![Image {
                        mime: "image/png".into(),
                        data: "QUJD".into(),
                    }],
                },
                Step::Assistant {
                    text: String::new(),
                    thinking: String::new(),
                    tool_calls: vec![ToolCall {
                        call_id: "c1".into(),
                        name: "read".into(),
                        args: json!({ "path": "x" }),
                    }],
                    raw: None,
                },
                Step::ToolResult {
                    call_id: "c1".into(),
                    name: "read".into(),
                    output: "data".into(),
                    is_error: false,
                },
            ],
        );
        assert_eq!(msgs[0]["role"], "system");
        // User with an image becomes a content-parts array.
        assert_eq!(msgs[1]["content"][0]["type"], "text");
        assert_eq!(msgs[1]["content"][1]["type"], "image_url");
        // Assistant carries the tool call; arguments are a JSON string.
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"x\"}"
        );
        // Tool result is a `tool` message keyed by call id.
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "c1");
    }

    #[test]
    fn finalizes_streamed_tool_calls() {
        let calls = vec![PartialCall {
            id: "c1".into(),
            name: "read".into(),
            args: "{\"path\":\"a\"}".into(),
        }];
        let out = finalize_calls(calls);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read");
        assert_eq!(out[0].args["path"], "a");
    }

    #[test]
    fn synthesizes_missing_call_id() {
        let out = finalize_calls(vec![PartialCall {
            id: String::new(),
            name: "ls".into(),
            args: "{}".into(),
        }]);
        assert_eq!(out[0].call_id, "call_0");
    }

    #[test]
    fn accumulates_streamed_text_thinking_and_tool_calls() {
        let mut acc = ChatAcc::default();
        // A content delta yields a Text event and accumulates.
        let d = acc.apply(
            &json!({ "choices": [{ "delta": { "content": "Hello " } }] }),
        );
        assert!(
            matches!(d.as_slice(), [TurnEvent::Text { delta }] if delta == "Hello ")
        );
        // Reasoning yields a Thinking event.
        let d = acc.apply(&json!({ "choices": [{ "delta": { "reasoning_content": "hmm" } }] }));
        assert!(
            matches!(d.as_slice(), [TurnEvent::Thinking { delta }] if delta == "hmm")
        );
        // A tool call streamed across deltas: id+name first, arguments in pieces.
        acc.apply(&json!({ "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "id": "call_1", "function": { "name": "read" } }
        ] } }] }));
        acc.apply(&json!({ "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "{\"path\":" } }
        ] } }] }));
        acc.apply(&json!({ "choices": [{ "delta": { "tool_calls": [
            { "index": 0, "function": { "arguments": "\"a\"}" } }
        ] } }] }));
        acc.apply(&json!({ "choices": [{ "delta": { "content": "world" } }] }));
        // The usage trailer arrives last (stream_options.include_usage).
        acc.apply(&json!({ "usage": { "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13 } }));

        assert_eq!(acc.text, "Hello world");
        assert_eq!(acc.thinking, "hmm");
        assert_eq!(acc.usage.output, 3);
        let calls = finalize_calls(acc.calls);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].args["path"], "a");
    }

    #[test]
    fn parses_chat_usage() {
        let u = json!({
            "prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60,
            "prompt_tokens_details": { "cached_tokens": 5 }
        });
        let usage = parse_usage(&u);
        assert_eq!(usage.input, 45);
        assert_eq!(usage.output, 10);
        assert_eq!(usage.cache_read, 5);
        assert_eq!(usage.total, 60);
    }
}
