//! The provider and model catalog — the list of providers `/login` offers and
//! the per-model metadata (context window, pricing, reasoning support) baked in
//! at build time (see [`models`]).

pub mod models;

// Re-exported at the `catalog` root (also at its canonical `catalog::models::`
// path) — the curated default model for a freshly signed-in provider.
pub use models::default_model;

/// Providers the `/login` flow offers: (id, display name).
pub const PROVIDERS: &[(&str, &str)] = &[
    ("copilot", "GitHub Copilot"),
    ("anthropic", "Anthropic"),
    ("openai", "OpenAI"),
    ("google", "Google"),
    ("deepseek", "DeepSeek"),
    ("openrouter", "OpenRouter"),
];

/// Model ids offered by a provider, from the baked-in [`models`] catalog.
pub fn models_for(provider_id: &str) -> Vec<String> {
    models::for_provider(provider_id)
        .into_iter()
        .map(|m| m.id.to_string())
        .collect()
}
