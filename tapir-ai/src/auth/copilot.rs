//! GitHub Copilot credential **resolution** — the non-interactive half of
//! Copilot auth, consumed by the engine's file-backed credential resolver and
//! the Responses wire shape.
//!
//! [`exchange_for_copilot_token`] trades a saved GitHub token for a fresh
//! Copilot bearer ([`OauthCreds`]); [`api_base`] derives the chat endpoint from
//! that bearer; [`fetch_available_models`] reads the account's enabled model
//! ids. None of this is interactive: *acquiring* the GitHub token (the RFC 8628
//! device flow, browser and all) is frontend work and lives in the TUI
//! (`tapir-tui`'s `device_flow`).
//!
//! Only `github.com` is supported for now (no GitHub Enterprise prompt).

use anyhow::{Result, anyhow};
use serde::Deserialize;

use super::OauthCreds;

/// Shared with the TUI's interactive device flow (`tapir-tui`), which speaks
/// to the same GitHub endpoints.
pub const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";

const COPILOT_TOKEN_URL: &str =
    "https://api.github.com/copilot_internal/v2/token";

/// Editor identification headers required by the Copilot API.
pub(crate) const EDITOR_VERSION: &str = "vscode/1.107.0";
pub(crate) const PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub(crate) const INTEGRATION_ID: &str = "vscode-chat";

#[derive(Deserialize)]
struct CopilotTokenResp {
    token: String,
    expires_at: i64,
}

/// Exchange the (saved) GitHub access token for a Copilot bearer token.
pub async fn exchange_for_copilot_token(
    client: &reqwest::Client,
    github_token: &str,
) -> Result<OauthCreds> {
    let resp = client
        .get(COPILOT_TOKEN_URL)
        .header(reqwest::header::ACCEPT, "application/json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {github_token}"),
        )
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .header("Editor-Version", EDITOR_VERSION)
        .header("Editor-Plugin-Version", PLUGIN_VERSION)
        .header("Copilot-Integration-Id", INTEGRATION_ID)
        .send()
        .await?;
    // Surface GitHub's response body on failure — it names the actual reason
    // (no Copilot access on the authorized account, bad credentials, …), which
    // a bare status code hides.
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Err(anyhow!(
            "Copilot token exchange failed: HTTP {status}: {body}"
        ));
    }
    parse_copilot_token(github_token, &body)
}

fn parse_copilot_token(github_token: &str, body: &str) -> Result<OauthCreds> {
    let r: CopilotTokenResp = serde_json::from_str(body)
        .map_err(|e| anyhow!("invalid Copilot token response: {e}"))?;
    Ok(OauthCreds {
        refresh: github_token.to_string(),
        // Expire 5 minutes early so we never present a token mid-expiry.
        expires: r.expires_at * 1000 - 5 * 60 * 1000,
        access: r.token,
    })
}

/// Default Copilot API host, used if the token carries no `proxy-ep`.
const DEFAULT_API_BASE: &str = "https://api.individual.githubcopilot.com";

/// The Copilot API base URL for a given access token (for chat requests).
pub fn api_base(token: &str) -> String {
    base_url_from_token(token)
}

/// Derive the API base URL from the token's `proxy-ep` (e.g.
/// `tid=…;proxy-ep=proxy.individual.githubcopilot.com;` → the `api.` host).
fn base_url_from_token(token: &str) -> String {
    let host =
        token.split(';').find_map(|kv| kv.trim().strip_prefix("proxy-ep="));
    match host {
        Some(proxy) => {
            let api =
                proxy.strip_prefix("proxy.").map(|rest| format!("api.{rest}"));
            format!("https://{}", api.as_deref().unwrap_or(proxy))
        }
        None => DEFAULT_API_BASE.to_string(),
    }
}

/// Fetch the models the account can actually use (the `model_picker_enabled`
/// ids from the Copilot `/models` endpoint), for availability marking.
pub async fn fetch_available_models(
    client: &reqwest::Client,
    access_token: &str,
) -> Result<Vec<String>> {
    let base = base_url_from_token(access_token);
    let resp = client
        .get(format!("{base}/models"))
        .header(reqwest::header::ACCEPT, "application/json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {access_token}"),
        )
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .header("Editor-Version", EDITOR_VERSION)
        .header("Editor-Plugin-Version", PLUGIN_VERSION)
        .header("Copilot-Integration-Id", INTEGRATION_ID)
        .send()
        .await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Err(anyhow!(
            "Copilot models probe failed: HTTP {status}: {body}"
        ));
    }
    parse_models(&body)
}

fn parse_models(body: &str) -> Result<Vec<String>> {
    let v: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| anyhow!("invalid models response: {e}"))?;
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow!("models response missing `data`"))?;
    let ids = data
        .iter()
        .filter(|m| {
            m.get("model_picker_enabled")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
        })
        .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
        .collect();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_derives_from_proxy_ep() {
        assert_eq!(
            base_url_from_token(
                "tid=1;exp=2;proxy-ep=proxy.individual.githubcopilot.com;"
            ),
            "https://api.individual.githubcopilot.com"
        );
        assert_eq!(base_url_from_token("tid=1;exp=2"), DEFAULT_API_BASE);
    }

    #[test]
    fn parse_models_keeps_only_picker_enabled() {
        let body = r#"{
            "data": [
                {"id": "gpt-5.4", "model_picker_enabled": true},
                {"id": "gpt-5.5", "model_picker_enabled": false},
                {"id": "claude-opus-4.8", "model_picker_enabled": true},
                {"id": "no-flag"}
            ]
        }"#;
        let ids = parse_models(body).unwrap();
        assert_eq!(ids, vec!["gpt-5.4", "claude-opus-4.8"]);
    }

    #[test]
    fn copilot_token_applies_expiry_margin() {
        let creds = parse_copilot_token(
            "ghu_abc",
            r#"{"token":"tid=1;exp=2;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":1700000000}"#,
        )
        .unwrap();
        assert_eq!(creds.refresh, "ghu_abc");
        assert!(creds.access.contains("proxy-ep="));
        assert_eq!(creds.expires, 1_700_000_000 * 1000 - 5 * 60 * 1000);
    }
}
