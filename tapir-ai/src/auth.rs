//! Persisted authentication (`~/.config/tapir/auth.toml`).
//!
//! Two kinds of credential live side by side, keyed by provider id:
//!   * `mock = true` — placeholder sign-in for providers without a real flow yet
//!     (anthropic / openai / google).
//!   * an `[oauth]` sub-table — real OAuth credentials, currently produced by the
//!     GitHub Copilot device flow (see [`copilot`]).
//!
//! On startup the saved provider is restored. The shape is stable so a real
//! backend can swap in without changing the UI.

pub mod copilot;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Auth {
    /// The provider to sign in to on launch. With several providers configured
    /// (now that keys are supported), this is the one the user last chose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<String>,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderAuth>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ProviderAuth {
    #[serde(default, skip_serializing_if = "is_false")]
    pub mock: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oauth: Option<OauthCreds>,
    /// Model ids the provider reports as enabled for this account (from its
    /// `/models` endpoint). Empty when unknown — then the UI shows all models.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// The model last selected for this provider (via `/model` or Ctrl+P),
    /// restored on the next launch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// API key for providers authenticated by key (Anthropic, OpenAI, DeepSeek,
    /// Gemini, OpenRouter). An environment variable, when set, takes precedence
    /// (see `providers::api_key_with`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// OAuth credentials. `refresh` is the long-lived token used to mint a fresh
/// `access` token; `expires` is epoch milliseconds (with a safety margin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OauthCreds {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Auth {
    /// The signed-in provider id, if any: the explicit `active` one when it's
    /// still configured, otherwise the first configured provider.
    pub fn signed_in(&self) -> Option<&str> {
        if let Some(a) = &self.active
            && self.providers.contains_key(a)
        {
            return Some(a.as_str());
        }
        self.providers.keys().next().map(String::as_str)
    }
}

fn auth_dir() -> Option<PathBuf> {
    // Share the config dir (honors `$TAPIR_CONFIG_DIR`) so `auth.toml` follows it.
    crate::config::dir()
}

/// The on-disk path of `auth.toml` (for user-facing messages).
pub fn path() -> Option<PathBuf> {
    auth_dir().map(|d| d.join("auth.toml"))
}

/// Load `auth.toml` (defaults — no providers — if missing or invalid).
pub fn load() -> Auth {
    auth_dir().map(|d| load_in(&d)).unwrap_or_default()
}

/// Persist real OAuth credentials for `provider_id` (preserving its model list).
pub fn set_oauth(provider_id: &str, creds: OauthCreds) {
    if let Some(dir) = auth_dir() {
        let _ = set_oauth_in(&dir, provider_id, creds);
    }
}

/// Persist the provider's enabled model ids (preserving its credentials).
pub fn set_models(provider_id: &str, models: Vec<String>) {
    if let Some(dir) = auth_dir() {
        let _ = set_models_in(&dir, provider_id, models);
    }
}

/// Persist the provider's last-selected model (preserving its credentials).
pub fn set_model_in(dir: &Path, provider_id: &str, model: &str) -> Result<()> {
    upsert_in(dir, provider_id, |e| e.model = Some(model.to_string()))
}

/// Persist an API key for `provider_id` and mark it configured (preserving
/// siblings). Used by `/login` for key-authenticated providers.
pub fn set_api_key_in(dir: &Path, provider_id: &str, key: &str) -> Result<()> {
    upsert_in(dir, provider_id, |e| {
        e.mock = false;
        e.api_key = Some(key.to_string());
    })
}

/// Mark `provider_id` as the active provider (ensuring it has an entry, so it's
/// restored on launch even when its key lives only in the environment).
pub fn set_active_in(dir: &Path, provider_id: &str) -> Result<()> {
    let mut auth = load_in(dir);
    auth.providers.entry(provider_id.to_string()).or_default();
    auth.active = Some(provider_id.to_string());
    write_in(dir, &auth)
}

/// Remove a provider's credentials (and clear `active` if it pointed there).
/// The new active provider, if any, is whatever remains first.
pub fn remove_provider_in(dir: &Path, provider_id: &str) -> Result<()> {
    let mut auth = load_in(dir);
    auth.providers.remove(provider_id);
    if auth.active.as_deref() == Some(provider_id) {
        auth.active = auth.providers.keys().next().cloned();
    }
    write_in(dir, &auth)
}

/// Load `auth.toml` from a specific directory (the app passes its config dir, so
/// tests stay hermetic — see `App::config_dir`).
pub fn load_in(dir: &Path) -> Auth {
    std::fs::read_to_string(dir.join("auth.toml"))
        .ok()
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_in(dir: &Path, auth: &Auth) -> Result<()> {
    // auth.toml holds plaintext API keys and OAuth tokens — keep it owner-only.
    std::fs::create_dir_all(dir)?;
    crate::config::restrict_dir(dir);
    let path = dir.join("auth.toml");
    std::fs::write(&path, toml::to_string(auth)?)?;
    crate::config::restrict_file(&path);
    Ok(())
}

/// Load, mutate one provider entry, and write back — preserving sibling fields.
fn upsert_in(
    dir: &Path,
    provider_id: &str,
    edit: impl FnOnce(&mut ProviderAuth),
) -> Result<()> {
    let mut auth = load_in(dir);
    edit(auth.providers.entry(provider_id.to_string()).or_default());
    write_in(dir, &auth)
}

#[cfg(test)]
fn add_provider_in(dir: &Path, provider_id: &str) -> Result<()> {
    upsert_in(dir, provider_id, |e| e.mock = true)
}

fn set_oauth_in(
    dir: &Path,
    provider_id: &str,
    creds: OauthCreds,
) -> Result<()> {
    upsert_in(dir, provider_id, |e| {
        e.mock = false;
        e.oauth = Some(creds);
    })
}

fn set_models_in(
    dir: &Path,
    provider_id: &str,
    models: Vec<String>,
) -> Result<()> {
    upsert_in(dir, provider_id, |e| e.models = models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_provider_persists_and_restores() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        assert!(load_in(&dir).signed_in().is_none());
        add_provider_in(&dir, "anthropic").unwrap();

        let auth = load_in(&dir);
        assert_eq!(auth.signed_in(), Some("anthropic"));
        assert!(auth.providers["anthropic"].mock);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oauth_creds_persist_and_restore() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-oauth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        set_oauth_in(
            &dir,
            "copilot",
            OauthCreds {
                refresh: "ghu_xxx".into(),
                access:
                    "tid=1;exp=2;proxy-ep=proxy.individual.githubcopilot.com;"
                        .into(),
                expires: 1_700_000_000_000,
            },
        )
        .unwrap();

        let auth = load_in(&dir);
        assert_eq!(auth.signed_in(), Some("copilot"));
        let entry = &auth.providers["copilot"];
        assert!(!entry.mock);
        let oauth = entry.oauth.as_ref().unwrap();
        assert_eq!(oauth.refresh, "ghu_xxx");
        assert_eq!(oauth.expires, 1_700_000_000_000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oauth_and_models_are_preserved_independently() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        set_oauth_in(
            &dir,
            "copilot",
            OauthCreds {
                refresh: "ghu".into(),
                access: "tid=1".into(),
                expires: 1,
            },
        )
        .unwrap();
        set_models_in(
            &dir,
            "copilot",
            vec!["gpt-5.4".into(), "claude-opus-4.8".into()],
        )
        .unwrap();

        let entry = &load_in(&dir).providers["copilot"];
        // set_models kept the credentials; set_oauth kept (empty) models before.
        assert!(entry.oauth.is_some());
        assert_eq!(entry.models, vec!["gpt-5.4", "claude-opus-4.8"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selected_model_persists_alongside_credentials() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-defmodel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        set_oauth_in(
            &dir,
            "copilot",
            OauthCreds {
                refresh: "ghu".into(),
                access: "tid=1".into(),
                expires: 1,
            },
        )
        .unwrap();
        set_model_in(&dir, "copilot", "claude-opus-4-8").unwrap();

        let entry = &load_in(&dir).providers["copilot"];
        assert_eq!(entry.model.as_deref(), Some("claude-opus-4-8"));
        assert!(entry.oauth.is_some(), "credentials preserved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn api_key_persists_and_marks_configured() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-apikey-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        set_api_key_in(&dir, "anthropic", "sk-ant-123").unwrap();
        set_model_in(&dir, "anthropic", "claude-opus-4-8").unwrap();

        let entry = &load_in(&dir).providers["anthropic"];
        assert_eq!(entry.api_key.as_deref(), Some("sk-ant-123"));
        assert!(!entry.mock, "an API key is real auth, not a mock sign-in");
        assert_eq!(
            entry.model.as_deref(),
            Some("claude-opus-4-8"),
            "model preserved"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn auth_file_is_written_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir()
            .join(format!("tapir-perms-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        set_api_key_in(&dir, "anthropic", "sk-secret").unwrap();
        let mode = std::fs::metadata(dir.join("auth.toml"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "auth.toml must not be group/world readable"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn active_provider_overrides_alphabetical_first() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-active-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        add_provider_in(&dir, "anthropic").unwrap();
        add_provider_in(&dir, "openai").unwrap();
        // Without an explicit active, signed_in() is the first key (anthropic).
        assert_eq!(load_in(&dir).signed_in(), Some("anthropic"));
        // Setting active flips it, even though openai sorts after anthropic.
        set_active_in(&dir, "openai").unwrap();
        assert_eq!(load_in(&dir).signed_in(), Some("openai"));
        // set_active ensures an entry exists even for an env-only provider.
        set_active_in(&dir, "deepseek").unwrap();
        assert_eq!(load_in(&dir).signed_in(), Some("deepseek"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
