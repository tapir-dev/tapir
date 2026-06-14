//! Google **Gemini** — `…/models/{model}:streamGenerateContent?alt=sse`.
//!
//! `systemInstruction` is top-level; roles are `user`/`model`; tool calls are
//! `functionCall` parts and their results `functionResponse` parts (coalesced
//! into one `user` turn, like Anthropic). Thinking arrives as parts flagged
//! `thought: true`. The API key rides in the `x-goog-api-key` header (not the
//! URL) so it never lands in logs.

use serde_json::{Value, json};

use super::{
    Creds, RoundOutcome, SseAccumulator, Step, base_url, budget_for,
    model_reasons, send_err, status_err,
};
use crate::message::{RoundError, ToolCall, ToolDef, TurnEvent, Usage};

/// Build the `contents` array, coalescing consecutive tool results.
pub fn build_contents(history: &[Step]) -> Vec<Value> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < history.len() {
        match &history[i] {
            Step::User { text, images } => {
                let mut parts = vec![json!({ "text": text })];
                for img in images {
                    parts.push(json!({
                        "inlineData": { "mimeType": img.mime, "data": img.data },
                    }));
                }
                out.push(json!({ "role": "user", "parts": parts }));
                i += 1;
            }
            Step::Assistant { text, tool_calls, raw, .. } => {
                let mut parts = Vec::new();
                if !text.is_empty() {
                    parts.push(json!({ "text": text }));
                }
                match raw {
                    // Echo the model's verbatim functionCall parts — they carry
                    // the `thoughtSignature` Gemini requires on replay.
                    Some(fc_parts) => parts.extend(fc_parts.iter().cloned()),
                    None => {
                        for tc in tool_calls {
                            parts.push(json!({ "functionCall": { "name": tc.name, "args": tc.args } }));
                        }
                    }
                }
                out.push(json!({ "role": "model", "parts": parts }));
                i += 1;
            }
            Step::ToolResult { .. } => {
                let mut parts = Vec::new();
                while let Some(Step::ToolResult { name, output, .. }) =
                    history.get(i)
                {
                    parts.push(json!({
                        "functionResponse": { "name": name, "response": { "output": output } },
                    }));
                    i += 1;
                }
                out.push(json!({ "role": "user", "parts": parts }));
            }
        }
    }
    out
}

/// Folds a Gemini `streamGenerateContent` SSE stream into text / thinking /
/// tool calls / usage. Pure (no I/O) so the parse can be unit-tested; `apply`
/// returns the text/thinking deltas the caller streams.
#[derive(Default)]
struct GeminiAcc {
    usage: Usage,
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
    /// Verbatim `functionCall` parts (carry `thoughtSignature`), echoed on replay.
    raw_parts: Vec<Value>,
}

impl SseAccumulator for GeminiAcc {
    fn apply(&mut self, ev: &Value) -> Vec<TurnEvent> {
        let mut out = Vec::new();
        if let Some(parts) =
            ev.pointer("/candidates/0/content/parts").and_then(|p| p.as_array())
        {
            for part in parts {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    if part
                        .get("thought")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        self.thinking.push_str(t);
                        out.push(TurnEvent::Thinking { delta: t.to_string() });
                    } else {
                        self.text.push_str(t);
                        out.push(TurnEvent::Text { delta: t.to_string() });
                    }
                } else if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args =
                        fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    let call_id = fc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| {
                            format!("call_{}", self.tool_calls.len())
                        });
                    if !name.is_empty() {
                        // Keep the whole part (functionCall + thoughtSignature).
                        self.raw_parts.push(part.clone());
                        self.tool_calls.push(ToolCall { call_id, name, args });
                    }
                }
            }
        }
        if let Some(u) = ev.get("usageMetadata") {
            self.usage = parse_usage(u);
        }
        out
    }
}

/// The `tools` array (Gemini `functionDeclarations` shape) for the active set.
fn tools_param(active: &[ToolDef]) -> Value {
    let decls: Vec<Value> = active
        .iter()
        .map(|d| {
            json!({
                "name": d.name,
                "description": d.description,
                "parametersJsonSchema": d.parameters.clone(),
            })
        })
        .collect();
    json!([{ "functionDeclarations": decls }])
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
    // An explicit endpoint (a custom provider) wins; else the fixed default.
    let base = base.map(str::to_string).unwrap_or_else(|| base_url("google"));

    let mut body = json!({
        "systemInstruction": { "parts": [{ "text": instructions }] },
        "contents": build_contents(history),
    });
    if !tools.is_empty() {
        body["tools"] = tools_param(tools);
    }
    if let Some(effort) = effort
        && model_reasons("google", model)
    {
        body["generationConfig"] = json!({
            "thinkingConfig": { "thinkingBudget": budget_for(effort), "includeThoughts": true },
        });
    }

    let url = format!("{base}/models/{model}:streamGenerateContent?alt=sse");
    let resp = client
        .post(&url)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("x-goog-api-key", key.as_str())
        .json(&body)
        .send()
        .await
        .map_err(send_err)?;
    if !resp.status().is_success() {
        return Err(status_err(resp).await);
    }

    let mut acc = GeminiAcc::default();
    super::drive_sse(resp, &mut acc, tx).await?;

    let raw = (!acc.raw_parts.is_empty()).then_some(acc.raw_parts);
    let tool_calls = acc.tool_calls;
    Ok(RoundOutcome {
        usage: acc.usage,
        tool_calls: tool_calls.clone(),
        assistant: Step::Assistant {
            text: acc.text,
            thinking: acc.thinking,
            tool_calls,
            raw,
        },
    })
}

fn parse_usage(u: &Value) -> Usage {
    let get = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let cache_read = get("cachedContentTokenCount");
    Usage {
        input: get("promptTokenCount").saturating_sub(cache_read),
        output: get("candidatesTokenCount"),
        cache_read,
        total: get("totalTokenCount"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_contents_with_function_call_and_response() {
        let contents = build_contents(&[
            Step::User { text: "hi".into(), images: vec![] },
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
                output: "DATA".into(),
                is_error: false,
            },
        ]);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["functionCall"]["name"], "read");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"][0]["functionResponse"]["name"], "read");
        assert_eq!(
            contents[2]["parts"][0]["functionResponse"]["response"]["output"],
            "DATA"
        );
    }

    #[test]
    fn accumulates_parts_into_text_thinking_and_calls() {
        let mut acc = GeminiAcc::default();
        let d = acc.apply(&json!({ "candidates": [{ "content": { "parts": [
            { "text": "reasoning", "thought": true },
            { "text": "answer" }
        ] } }] }));
        assert!(
            matches!(&d[0], TurnEvent::Thinking { delta } if delta == "reasoning")
        );
        assert!(
            matches!(&d[1], TurnEvent::Text { delta } if delta == "answer")
        );
        // A functionCall part is captured verbatim in `raw` (thoughtSignature).
        acc.apply(&json!({ "candidates": [{ "content": { "parts": [
            { "functionCall": { "name": "read", "args": { "path": "a" } }, "thoughtSignature": "sig" }
        ] } }] }));
        acc.apply(&json!({ "usageMetadata": {
            "promptTokenCount": 100, "candidatesTokenCount": 25,
            "totalTokenCount": 125, "cachedContentTokenCount": 10
        } }));

        assert_eq!(acc.text, "answer");
        assert_eq!(acc.thinking, "reasoning");
        assert_eq!(acc.usage.output, 25);
        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].name, "read");
        assert_eq!(
            acc.tool_calls[0].call_id, "call_0",
            "synthesized when no id"
        );
        assert_eq!(acc.tool_calls[0].args["path"], "a");
        assert_eq!(acc.raw_parts.len(), 1);
        assert_eq!(
            acc.raw_parts[0]["thoughtSignature"], "sig",
            "whole part kept for replay"
        );
    }

    #[test]
    fn parses_gemini_usage() {
        let u = json!({
            "promptTokenCount": 100, "candidatesTokenCount": 25,
            "totalTokenCount": 125, "cachedContentTokenCount": 10
        });
        let usage = parse_usage(&u);
        assert_eq!(usage.input, 90);
        assert_eq!(usage.output, 25);
        assert_eq!(usage.cache_read, 10);
        assert_eq!(usage.total, 125);
    }
}
