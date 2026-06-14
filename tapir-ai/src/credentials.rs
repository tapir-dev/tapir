//! Credential **resolution**: the [`CredentialProvider`] trait and its built-in
//! resolvers.
//!
//! Resolution is split from acquisition. A resolver is injected on the
//! [`RuntimeBuilder`](crate::runtime::RuntimeBuilder) and consulted by each
//! [`Provider`](crate::providers::Provider)'s default credential path per turn;
//! the engine never acquires credentials interactively — obtaining them (a
//! device flow, a browser login) is a frontend's job, which then writes what
//! [`FileCreds`] reads.
//!
//! Two resolvers ship:
//! - [`FileCreds`] (the default) — the engine's existing chain over `auth.toml`:
//!   a runtime `--api-key`, then the provider's environment variable, then the
//!   saved key; Copilot mints a fresh bearer from its saved OAuth refresh
//!   (a non-interactive exchange). The on-disk format is unchanged.
//! - [`EnvCreds`] — one shared service credential for every provider id, for a
//!   headless adapter (a bot) running non-interactively with a single key.

use async_trait::async_trait;

use crate::providers::Creds;

/// Resolves credentials by provider id. Object-safe (`async-trait` boxes the
/// future) and async so an implementation can consult an external service (a
/// secret manager, an OAuth token mint).
#[async_trait]
pub trait CredentialProvider: Send + Sync {
    /// The credentials for `provider`, or an error explaining what's missing.
    /// `client` serves resolvers that need HTTP (Copilot's bearer mint); most
    /// ignore it.
    async fn resolve(
        &self,
        client: &reqwest::Client,
        provider: &str,
    ) -> anyhow::Result<Creds>;
}

/// The default resolver: the engine's existing credential chain over
/// `auth.toml` (format unchanged). A runtime `--api-key` wins, then the
/// provider's environment variable, then the saved key; Copilot exchanges its
/// saved OAuth refresh for a fresh bearer — non-interactive resolution only.
#[derive(Default)]
pub struct FileCreds {
    /// Where `auth.toml` lives; `None` is the process config dir.
    dir: Option<std::path::PathBuf>,
}

impl FileCreds {
    /// A resolver over the process config dir (where `auth.toml` lives).
    pub fn new() -> Self {
        Self::default()
    }

    /// A resolver over an explicit directory's `auth.toml` — an embedder (or a
    /// hermetic test) points it at its own credential file.
    // Embedder-facing; the TUI uses the config-dir default, so only the seam
    // tests construct this in this binary.
    pub fn in_dir(dir: impl Into<std::path::PathBuf>) -> Self {
        Self { dir: Some(dir.into()) }
    }

    /// The credential file's contents, from the explicit dir or the config dir.
    fn auth(&self) -> crate::auth::Auth {
        match &self.dir {
            Some(dir) => crate::auth::load_in(dir),
            None => crate::auth::load(),
        }
    }
}

/// A single shared service credential for every provider id, read from one
/// environment variable — the headless operator's resolver: a bot deploys with
/// one key and no browser-based login, fully non-interactive.
// Adapter-facing (the TUI keeps the FileCreds default); exercised by the seam
// tests until a headless adapter (#14) injects it.
pub struct EnvCreds {
    var: String,
}

impl EnvCreds {
    /// A resolver serving the value of `var` as the API key for any provider.
    pub fn var(var: impl Into<String>) -> Self {
        Self { var: var.into() }
    }
}

#[async_trait]
impl CredentialProvider for EnvCreds {
    async fn resolve(
        &self,
        _client: &reqwest::Client,
        _provider: &str,
    ) -> anyhow::Result<Creds> {
        match std::env::var(&self.var) {
            Ok(key) if !key.trim().is_empty() => {
                Ok(Creds::ApiKey { key: key.trim().into() })
            }
            _ => {
                Err(anyhow::anyhow!("no service credential — set {}", self.var))
            }
        }
    }
}

#[async_trait]
impl CredentialProvider for FileCreds {
    async fn resolve(
        &self,
        client: &reqwest::Client,
        provider: &str,
    ) -> anyhow::Result<Creds> {
        // Copilot is the only OAuth provider; everyone else (incl. OpenAI, which
        // also speaks the Responses API) authenticates with an API key.
        if provider == "copilot" {
            let refresh = self
                .auth()
                .providers
                .get(provider)
                .and_then(|p| p.oauth.as_ref())
                .map(|o| o.refresh.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("no saved credentials for {provider}")
                })?;
            let creds = crate::auth::copilot::exchange_for_copilot_token(
                client, &refresh,
            )
            .await?;
            Ok(Creds::Copilot { access: creds.access })
        } else {
            let key = crate::providers::api_key_with(&self.auth(), provider)
                .ok_or_else(|| {
                    let var = crate::providers::env_var(provider)
                        .unwrap_or("the API key");
                    anyhow::anyhow!(
                        "no API key for {provider} — set {var} or run /login"
                    )
                })?;
            Ok(Creds::ApiKey { key })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_creds_reads_the_existing_auth_file_format() {
        // The resolver reads what the frontend's login wrote — via the existing
        // helper, so the on-disk format is the existing one, unchanged. An id
        // with no env-var mapping keeps the test hermetic (file part only).
        let dir = std::env::temp_dir()
            .join(format!("tapir-filecreds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        crate::auth::set_api_key_in(&dir, "myai", "sk-from-file").unwrap();

        let resolver = FileCreds::in_dir(&dir);
        let client = crate::providers::base_client(5);
        let creds = resolver
            .resolve(&client, "myai")
            .await
            .expect("the saved key resolves");
        assert!(
            matches!(creds, Creds::ApiKey { ref key } if key == "sk-from-file"),
            "the key written in the existing auth.toml format is what resolves",
        );

        // A provider with nothing saved fails with the actionable hint — the
        // resolver never acquires interactively. (`.err()`, not `unwrap_err()`:
        // Creds deliberately has no Debug, so secrets can't leak into logs.)
        let err = resolver
            .resolve(&client, "nokey")
            .await
            .err()
            .expect("missing creds error");
        assert!(
            err.to_string().contains("no API key"),
            "missing creds are an error: {err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn env_creds_serves_one_shared_credential_for_any_provider() {
        // The headless-operator case: one service key, non-interactive, the
        // same for every provider id a session might use.
        // SAFETY: a test-only variable nothing else reads (env prior art:
        // providers::tests::api_key_prefers_runtime_then_env_then_file).
        unsafe { std::env::set_var("TAPIR_TEST_SERVICE_KEY", "sk-service") };

        let resolver = EnvCreds::var("TAPIR_TEST_SERVICE_KEY");
        let client = crate::providers::base_client(5);
        for id in ["deepseek", "anthropic", "myai"] {
            let creds = resolver
                .resolve(&client, id)
                .await
                .expect("the shared key resolves");
            assert!(
                matches!(creds, Creds::ApiKey { ref key } if key == "sk-service"),
                "every provider id gets the one shared credential ({id})",
            );
        }

        // An unset variable is an actionable error naming the variable.
        let unset = EnvCreds::var("TAPIR_TEST_UNSET_KEY");
        let err = unset
            .resolve(&client, "deepseek")
            .await
            .err()
            .expect("unset var errors");
        assert!(
            err.to_string().contains("TAPIR_TEST_UNSET_KEY"),
            "the error names the var: {err}"
        );

        unsafe { std::env::remove_var("TAPIR_TEST_SERVICE_KEY") };
    }

    #[tokio::test]
    async fn an_explicit_provider_key_wins_over_the_injected_resolver() {
        use crate::providers::{Api, Provider, WireProvider};

        // A resolver that would hand out a different key — the explicit key on
        // the provider must shadow it (the #08 convenience-constructor contract,
        // preserved through the resolver injection).
        struct OtherKey;
        #[async_trait]
        impl CredentialProvider for OtherKey {
            async fn resolve(
                &self,
                _: &reqwest::Client,
                _: &str,
            ) -> anyhow::Result<Creds> {
                Ok(Creds::ApiKey { key: "sk-from-resolver".into() })
            }
        }

        let p = WireProvider::new("myai", Api::Chat).api_key("sk-explicit");
        let client = crate::providers::base_client(5);
        let creds = p.creds(&client, &OtherKey).await.expect("resolves");
        assert!(
            matches!(creds, Creds::ApiKey { ref key } if key == "sk-explicit"),
            "the provider's explicit key shadows the injected resolver",
        );
    }
}
