//! Prompt templates.
//!
//! A prompt template is a Markdown file whose filename (without `.md`) becomes a
//! `/command`. Typing `/name [args]` expands the file's body — with argument
//! placeholders substituted — into a normal user message that's sent to the
//! model. They're the user's reusable "workflow commands" (e.g. `/review`,
//! `/pr <url>`), distinct from skills (which wrap a body in a `<skill>` block).
//!
//! Discovery mirrors skills: tapir's own `<config>/prompts`, the cross-agent
//! `~/.agents/prompts`, then the project's `.tapir/prompts`. Unlike skills,
//! discovery is **non-recursive** — only direct `.md` children of each
//! directory are templates.
//!
//! Placeholders in the body:
//! - `$1`, `$2`, … positional arguments,
//! - `$@` and `$ARGUMENTS` for all arguments joined,
//! - `${@:N}` for arguments from the Nth (1-indexed) onward,
//! - `${@:N:L}` for `L` arguments starting at N.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct PromptTemplate {
    /// The `/command` name — the filename without `.md`.
    pub name: String,
    /// From frontmatter `description`, or the first non-empty body line.
    pub description: String,
    /// Optional frontmatter `argument-hint`, shown in the menu before the
    /// description (e.g. `<PR-URL>`).
    pub argument_hint: Option<String>,
    /// The template body (frontmatter stripped) — the text that gets expanded.
    pub content: String,
    pub file_path: PathBuf,
}

/// Split a Markdown file into its frontmatter map and body. Returns an empty map
/// (and the whole content as body) when there's no `---`-delimited frontmatter.
fn parse_frontmatter(content: &str) -> (BTreeMap<String, String>, String) {
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (BTreeMap::new(), content.to_string());
    }
    let mut map = BTreeMap::new();
    let mut body = Vec::new();
    let mut in_fm = true;
    for line in lines {
        if in_fm {
            if line.trim() == "---" {
                in_fm = false;
                continue;
            }
            if let Some((k, v)) = line.split_once(':') {
                let v = v.trim().trim_matches('"').trim_matches('\'');
                map.insert(k.trim().to_string(), v.to_string());
            }
        } else {
            body.push(line);
        }
    }
    // Unterminated frontmatter → treat the file as having none.
    if in_fm {
        return (BTreeMap::new(), content.to_string());
    }
    (map, body.join("\n"))
}

/// Parse command arguments respecting `"…"` / `'…'` quoting (bash-style).
pub fn parse_command_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quote: Option<char> = None;
    for ch in s.chars() {
        match in_quote {
            Some(q) => {
                if ch == q {
                    in_quote = None;
                } else {
                    current.push(ch);
                }
            }
            None => {
                if ch == '"' || ch == '\'' {
                    in_quote = Some(ch);
                } else if ch.is_whitespace() {
                    if !current.is_empty() {
                        args.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push(ch);
                }
            }
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

/// Substitute argument placeholders in `content` (see the module docs). A single
/// left-to-right pass, so argument values that themselves contain `$1`/`$@` are
/// never re-substituted.
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let all = args.join(" ");
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            // Copy one full UTF-8 char.
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                i += 1;
            }
            out.push_str(&content[start..i]);
            continue;
        }
        let rest = &content[i + 1..];
        if let Some(after) = rest.strip_prefix("ARGUMENTS") {
            out.push_str(&all);
            i = content.len() - after.len();
        } else if rest.starts_with("{@:") {
            // `${@:N}` or `${@:N:L}` — bash-style slice (1-indexed).
            if let Some(close) = rest.find('}') {
                let inner = &rest[3..close];
                let (start_s, len_s) = match inner.split_once(':') {
                    Some((a, b)) => (a.trim(), Some(b.trim())),
                    None => (inner.trim(), None),
                };
                if let Ok(n) = start_s.parse::<usize>() {
                    let start = n.saturating_sub(1);
                    let slice: Vec<&String> =
                        match len_s.and_then(|l| l.parse::<usize>().ok()) {
                            Some(len) => {
                                args.iter().skip(start).take(len).collect()
                            }
                            None => args.iter().skip(start).collect(),
                        };
                    out.push_str(
                        &slice
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(" "),
                    );
                    i += 1 + close + 1;
                } else {
                    out.push('$');
                    i += 1;
                }
            } else {
                out.push('$');
                i += 1;
            }
        } else if rest.starts_with('@') {
            out.push_str(&all);
            i += 2;
        } else if rest.as_bytes().first().is_some_and(u8::is_ascii_digit) {
            let digits: String =
                rest.chars().take_while(char::is_ascii_digit).collect();
            // `$0` → empty (1-indexed; argument 0 does not exist).
            if let Some(idx) =
                digits.parse::<usize>().ok().and_then(|n| n.checked_sub(1))
                && let Some(v) = args.get(idx)
            {
                out.push_str(v);
            }
            i += 1 + digits.len();
        } else {
            out.push('$');
            i += 1;
        }
    }
    out
}

fn load_file(path: &Path) -> Option<PromptTemplate> {
    let raw = std::fs::read_to_string(path).ok()?;
    let (fm, body) = parse_frontmatter(&raw);
    let name = path.file_stem()?.to_string_lossy().into_owned();
    if name.is_empty() {
        return None;
    }
    // Description: frontmatter, else the first non-empty body line (truncated).
    let description =
        match fm.get("description").map(|s| s.trim()).filter(|s| !s.is_empty())
        {
            Some(d) => d.to_string(),
            None => match body.lines().find(|l| !l.trim().is_empty()) {
                Some(first) if first.chars().count() > 60 => {
                    format!("{}...", first.chars().take(60).collect::<String>())
                }
                Some(first) => first.to_string(),
                None => String::new(),
            },
        };
    let argument_hint = fm
        .get("argument-hint")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    Some(PromptTemplate {
        name,
        description,
        argument_hint,
        content: body,
        file_path: path.to_path_buf(),
    })
}

/// Scan `dir` for `.md` files (non-recursive, following symlinks to files).
fn load_from_dir(dir: &Path, out: &mut Vec<PromptTemplate>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.starts_with('.') || !name.ends_with(".md") {
            continue;
        }
        // `is_file` follows symlinks, so a symlinked `.md` is included.
        if path.is_file()
            && let Some(t) = load_file(&path)
        {
            out.push(t);
        }
    }
}

/// Load templates from each root in order, de-duplicated by name (first
/// wins). Used by tests (incl. frontends', via the `test-util` feature).
#[cfg(any(test, feature = "test-util"))]
pub fn load_from_roots(roots: &[PathBuf]) -> Vec<PromptTemplate> {
    let mut raw = Vec::new();
    for root in roots {
        load_from_dir(root, &mut raw);
    }
    let mut seen = BTreeSet::new();
    raw.into_iter().filter(|t| seen.insert(t.name.clone())).collect()
}

/// The prompt-template directories, in priority order — the same shape as
/// [`crate::skills::roots`], but the `prompts` subdirectory.
#[cfg(test)]
pub fn roots(cwd: &Path, tapir_config: Option<&Path>) -> Vec<PathBuf> {
    roots_with(cwd, tapir_config, true)
}

/// Like `roots`, but the project-local `<cwd>/.tapir/prompts` is included only
/// when `trust_project` (it's dropped by `--no-approve`).
pub fn roots_with(
    cwd: &Path,
    tapir_config: Option<&Path>,
    trust_project: bool,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(config) = tapir_config {
        roots.push(config.join("prompts"));
    }
    if let Ok(home) = etcetera::home_dir() {
        roots.push(home.join(".agents").join("prompts"));
    }
    if trust_project {
        roots.push(cwd.join(".tapir").join("prompts"));
    }
    roots
}

/// All prompt templates visible to tapir, honoring the CLI flags: `extra` are
/// explicit `--prompt-template` paths (a directory of `.md`, or a single `.md`
/// file),
/// additive even when `include_defaults` is false (`--no-prompt-templates` only
/// disables default discovery). `trust_project` gates the project-local root.
pub fn load_with(
    cwd: &Path,
    tapir_config: Option<&Path>,
    extra: &[PathBuf],
    include_defaults: bool,
    trust_project: bool,
) -> Vec<PromptTemplate> {
    let mut raw = Vec::new();
    if include_defaults {
        for root in roots_with(cwd, tapir_config, trust_project) {
            load_from_dir(&root, &mut raw);
        }
    }
    for path in extra {
        if path.is_dir() {
            load_from_dir(path, &mut raw);
        } else if path.is_file()
            && let Some(t) = load_file(path)
        {
            raw.push(t);
        }
    }
    let mut seen = BTreeSet::new();
    raw.into_iter().filter(|t| seen.insert(t.name.clone())).collect()
}

/// Expand `/name [args]` against the loaded templates. Returns the substituted
/// body, or `None` when `text` isn't `/something` or no template matches.
pub fn expand(text: &str, templates: &[PromptTemplate]) -> Option<String> {
    let rest = text.trim_start().strip_prefix('/')?;
    let (name, args_str) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim_start()),
        None => (rest, ""),
    };
    let tpl = templates.iter().find(|t| t.name == name)?;
    let args = parse_command_args(args_str);
    Some(substitute_args(&tpl.content, &args))
}

/// A short, model-facing reminder that tapir supports prompt templates and how
/// to author one — so "build me a /command for X" works (tapir embeds the
/// essentials).
pub const HELP_FOR_PROMPT: &str = "Prompt templates (custom /commands): you can create reusable \
slash commands for the user. A template is a Markdown file under `~/.agents/prompts/` (global) or \
`<cwd>/.tapir/prompts/` (project); the filename without `.md` becomes the command, so `review.md` \
is invoked as `/review`. Optional frontmatter keys: `description` and `argument-hint`. The body may \
reference arguments with `$1`, `$2`, … (positional), `$@` or `$ARGUMENTS` (all joined), `${@:N}` \
(from the Nth onward) and `${@:N:L}` (L args from N). When the user asks you to build a workflow \
command, write such a file with the write tool and tell them how to invoke it.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutes_positional_all_and_slices() {
        let args =
            parse_command_args(r#"Button "click handler" disabled extra"#);
        assert_eq!(args, ["Button", "click handler", "disabled", "extra"]);

        assert_eq!(
            substitute_args("name $1 feats: $@", &args),
            "name Button feats: Button click handler disabled extra"
        );
        assert_eq!(
            substitute_args("$ARGUMENTS", &args),
            "Button click handler disabled extra"
        );
        assert_eq!(
            substitute_args("${@:2}", &args),
            "click handler disabled extra"
        );
        assert_eq!(
            substitute_args("${@:2:2}", &args),
            "click handler disabled"
        );
        // Missing positional → empty; literal `$` left alone.
        assert_eq!(substitute_args("$9 cost $$", &[]), " cost $$");
        // `$0` is empty (1-indexed).
        assert_eq!(substitute_args("[$0]", &args), "[]");
    }

    #[test]
    fn loads_md_with_frontmatter_and_hint() {
        let root = std::env::temp_dir()
            .join(format!("tapir-prompts-{}", std::process::id()));
        let dir = root.join(".tapir/prompts");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("pr.md"),
            "---\ndescription: Review a PR\nargument-hint: \"<PR-URL>\"\n---\nReview $1 carefully.",
        )
        .unwrap();
        // No frontmatter → first non-empty line becomes the description.
        std::fs::write(dir.join("note.md"), "Just jot a note about $@.")
            .unwrap();
        // Hidden files are skipped.
        std::fs::write(dir.join(".hidden.md"), "nope").unwrap();

        let tpls = load_from_roots(&[root.join(".tapir").join("prompts")]);
        assert_eq!(tpls.len(), 2);
        let pr = tpls.iter().find(|t| t.name == "pr").unwrap();
        assert_eq!(pr.description, "Review a PR");
        assert_eq!(pr.argument_hint.as_deref(), Some("<PR-URL>"));
        let note = tpls.iter().find(|t| t.name == "note").unwrap();
        assert_eq!(note.description, "Just jot a note about $@.");

        assert_eq!(
            expand("/pr https://x/1", &tpls).unwrap(),
            "Review https://x/1 carefully."
        );
        assert!(expand("/unknown", &tpls).is_none());
        assert!(expand("hello", &tpls).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn roots_match_the_skills_shape_under_prompts() {
        let rs = roots(Path::new("/tmp/proj"), Some(Path::new("/cfg")));
        let has = |needle: &str| rs.iter().any(|r| r.ends_with(needle));
        assert!(rs.iter().any(|r| r.starts_with("/cfg")));
        assert!(has(".agents/prompts"));
        assert!(has(".tapir/prompts"));
    }
}
