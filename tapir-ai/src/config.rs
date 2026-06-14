//! Persisted user settings (`~/.config/tapir/config.toml`).
//!
//! Global settings persist as follows: changing a setting writes it, and it
//! is loaded on startup. Auth lives separately in `auth.toml` (slice 05).
//!
//! The writers come in `_in(dir, …)` form so the app can target a specific
//! directory (and tests a tempdir, or `None` to skip persistence entirely —
//! see `App::config_dir`).

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    /// Name of the selected theme, if the user has chosen one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Theme appearance mode: `auto` (match the terminal), `dark`, or `light`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub appearance: Option<String>,
    /// Reasoning depth for thinking-capable models (off..xhigh).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
    /// Whether context auto-compaction is enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_compact: Option<bool>,
    /// Whether the model's reasoning ("thinking") is shown in the transcript.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_thinking: Option<bool>,
    /// Provider HTTP idle timeout, in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_timeout_secs: Option<u64>,
    /// How many times to retry a provider request on a transient failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_retries: Option<u32>,
    /// Delivery mode for queued steering messages (`one-at-a-time` / `all`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steering_mode: Option<String>,
    /// Delivery mode for queued follow-up messages (`one-at-a-time` / `all`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow_up_mode: Option<String>,
}

/// The config directory: `$TAPIR_CONFIG_DIR` when set, else `~/.config/tapir`.
/// Holds `config.toml`, `auth.toml`, `themes/`, `skills/`, `prompts/`, and
/// (unless overridden) `sessions/`.
pub fn dir() -> Option<PathBuf> {
    if let Some(d) =
        std::env::var_os("TAPIR_CONFIG_DIR").filter(|s| !s.is_empty())
    {
        return Some(PathBuf::from(d));
    }
    directories::ProjectDirs::from("", "", "tapir")
        .map(|d| d.config_dir().to_path_buf())
}

/// Load `config.toml` (defaults if missing or invalid).
pub fn load() -> Config {
    dir().map(|d| load_in(&d)).unwrap_or_default()
}

/// Extra text appended to the system prompt, from `system.md` in the config dir
/// (empty when absent) — the user's way to extend the agent's instructions.
pub fn system_append() -> String {
    dir()
        .and_then(|d| std::fs::read_to_string(d.join("system.md")).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// The full system-prompt append: the global `system.md`, then any
/// `--append-system-prompt` pieces, joined by blank lines (empties dropped).
pub fn merge_append(cli_append: &[String]) -> String {
    let mut parts = Vec::new();
    let base = system_append();
    if !base.is_empty() {
        parts.push(base);
    }
    parts.extend(
        cli_append
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    );
    parts.join("\n\n")
}

pub fn load_in(dir: &Path) -> Config {
    std::fs::read_to_string(dir.join("config.toml"))
        .ok()
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_in(dir: &Path, config: &Config) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    restrict_dir(dir);
    let path = dir.join("config.toml");
    std::fs::write(&path, toml::to_string(config)?)?;
    restrict_file(&path);
    Ok(())
}

/// Tighten a file to owner-only (`0600`) on Unix; no-op elsewhere. Used for
/// anything that may hold secrets (auth, sessions) or private settings.
pub fn restrict_file(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(
            path,
            std::fs::Permissions::from_mode(0o600),
        );
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Tighten a directory to owner-only (`0700`) on Unix; no-op elsewhere.
pub fn restrict_dir(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(
            path,
            std::fs::Permissions::from_mode(0o700),
        );
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Load, mutate one field via `f`, and write back — preserving every other
/// config field. The single load/write site for all the `set_*_in` setters.
fn update_in(dir: &Path, f: impl FnOnce(&mut Config)) -> Result<()> {
    let mut config = load_in(dir);
    f(&mut config);
    write_in(dir, &config)
}

/// Persist the selected theme name, preserving any other config fields.
pub fn set_theme_in(dir: &Path, name: &str) -> Result<()> {
    update_in(dir, |c| c.theme = Some(name.to_string()))
}

/// Persist the theme appearance mode, preserving any other config fields.
pub fn set_appearance_in(dir: &Path, mode: &str) -> Result<()> {
    update_in(dir, |c| c.appearance = Some(mode.to_string()))
}

/// Persist the default thinking level, preserving any other config fields.
pub fn set_thinking_level_in(dir: &Path, level: &str) -> Result<()> {
    update_in(dir, |c| c.thinking_level = Some(level.to_string()))
}

/// Persist the auto-compact flag, preserving any other config fields.
pub fn set_auto_compact_in(dir: &Path, enabled: bool) -> Result<()> {
    update_in(dir, |c| c.auto_compact = Some(enabled))
}

/// Persist the show-thinking flag, preserving any other config fields.
pub fn set_show_thinking_in(dir: &Path, enabled: bool) -> Result<()> {
    update_in(dir, |c| c.show_thinking = Some(enabled))
}

/// Persist the provider HTTP timeout, preserving any other config fields.
pub fn set_http_timeout_in(dir: &Path, secs: u64) -> Result<()> {
    update_in(dir, |c| c.http_timeout_secs = Some(secs))
}

/// Persist the provider retry count, preserving any other config fields.
pub fn set_http_retries_in(dir: &Path, retries: u32) -> Result<()> {
    update_in(dir, |c| c.http_retries = Some(retries))
}

/// Persist the steering delivery mode, preserving any other config fields.
pub fn set_steering_mode_in(dir: &Path, mode: &str) -> Result<()> {
    update_in(dir, |c| c.steering_mode = Some(mode.to_string()))
}

/// Persist the follow-up delivery mode, preserving any other config fields.
pub fn set_follow_up_mode_in(dir: &Path, mode: &str) -> Result<()> {
    update_in(dir, |c| c.follow_up_mode = Some(mode.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_persist_and_load() {
        let dir = std::env::temp_dir()
            .join(format!("tapir-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(load_in(&dir).theme, None);
        set_theme_in(&dir, "solarized-dark").unwrap();
        set_thinking_level_in(&dir, "high").unwrap();
        set_auto_compact_in(&dir, false).unwrap();
        set_show_thinking_in(&dir, false).unwrap();
        set_http_timeout_in(&dir, 300).unwrap();
        set_http_retries_in(&dir, 5).unwrap();
        set_appearance_in(&dir, "light").unwrap();

        let cfg = load_in(&dir);
        assert_eq!(cfg.theme.as_deref(), Some("solarized-dark"));
        assert_eq!(cfg.appearance.as_deref(), Some("light"));
        assert_eq!(cfg.thinking_level.as_deref(), Some("high"));
        assert_eq!(cfg.auto_compact, Some(false));
        assert_eq!(cfg.show_thinking, Some(false));
        assert_eq!(cfg.http_timeout_secs, Some(300));
        assert_eq!(cfg.http_retries, Some(5));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_dir_honors_env_override() {
        // SAFETY: single-threaded test owning its own scratch var.
        unsafe {
            std::env::set_var("TAPIR_CONFIG_DIR", "/tmp/tapir-cfg-override")
        };
        assert_eq!(
            dir().as_deref(),
            Some(Path::new("/tmp/tapir-cfg-override"))
        );
        unsafe { std::env::remove_var("TAPIR_CONFIG_DIR") };
    }
}
