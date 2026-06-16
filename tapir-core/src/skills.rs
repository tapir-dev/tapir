//! Agent skills.
//!
//! A skill is a Markdown file with YAML-ish frontmatter (`name`, `description`,
//! `disable-model-invocation`). Skills are discovered from tapir's own
//! `<config>/skills`, the shared cross-agent `~/.agents/skills`, plus the
//! project's `.tapir/skills`:
//!
//! - a directory containing `SKILL.md` is a skill root (no recursion past it),
//! - otherwise direct `.md` children are skills, and subdirectories are searched
//!   for `SKILL.md`.
//!
//! Two ways to use a skill:
//! - **model-invocation**: skills are listed in the system prompt and the model
//!   loads the file with the `read` tool when a task matches,
//! - **explicit**: `/skill:name [args]` expands into a `<skill>` block.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    pub base_dir: PathBuf,
    /// `disable-model-invocation: true` hides it from the prompt (explicit only).
    pub disable_model_invocation: bool,
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

fn load_file(path: &Path) -> Option<Skill> {
    let content = std::fs::read_to_string(path).ok()?;
    let (fm, _) = parse_frontmatter(&content);
    let description = fm.get("description")?.trim().to_string();
    if description.is_empty() {
        return None; // description is required
    }
    let base_dir = path.parent()?.to_path_buf();
    let name =
        fm.get("name").filter(|s| !s.is_empty()).cloned().unwrap_or_else(
            || {
                base_dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            },
        );
    if name.is_empty() {
        return None;
    }
    Some(Skill {
        name,
        description,
        file_path: path.to_path_buf(),
        base_dir,
        disable_model_invocation: fm
            .get("disable-model-invocation")
            .map(|v| v == "true")
            .unwrap_or(false),
    })
}

/// Bound on recursion, so a symlinked skill dir can't loop forever.
const MAX_DEPTH: usize = 8;

/// Discover skills under `dir` (see the module docs for the rules).
fn load_from_dir(
    dir: &Path,
    include_root_files: bool,
    depth: usize,
    out: &mut Vec<Skill>,
) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let entries: Vec<PathBuf> = read.flatten().map(|e| e.path()).collect();

    // A `SKILL.md` makes this directory a single skill root — stop here.
    if let Some(skill_md) =
        entries.iter().find(|p| p.file_name().is_some_and(|n| n == "SKILL.md"))
        && skill_md.is_file()
    {
        if let Some(s) = load_file(skill_md) {
            out.push(s);
        }
        return;
    }

    for path in &entries {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            load_from_dir(path, false, depth + 1, out);
        } else if include_root_files
            && path.extension().is_some_and(|e| e == "md")
            && let Some(s) = load_file(path)
        {
            out.push(s);
        }
    }
}

/// Load skills from each root directory in order, de-duplicated by name (first
/// wins). Used by tests (incl. frontends', via the `test-util` feature);
/// production goes through [`load_with`].
#[cfg(any(test, feature = "test-util"))]
pub fn load_from_roots(roots: &[PathBuf]) -> Vec<Skill> {
    let mut raw = Vec::new();
    for root in roots {
        load_from_dir(root, true, 0, &mut raw);
    }
    let mut seen = BTreeSet::new();
    raw.into_iter().filter(|s| seen.insert(s.name.clone())).collect()
}

/// The skill root directories, in priority order: tapir's own global, then the
/// shared cross-agent location `~/.agents/skills`, then the project's
/// `.tapir/skills`. The project-local `<cwd>/.tapir/skills` is included only
/// when `trust_project` (dropped by `--no-approve`).
pub fn roots_with(
    cwd: &Path,
    tapir_config: Option<&Path>,
    trust_project: bool,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(config) = tapir_config {
        roots.push(config.join("skills"));
    }
    if let Ok(home) = etcetera::home_dir() {
        // `~/.agents/skills` is the canonical cross-agent location
        // (other agents may symlink their own skills dir into it, so we skip those).
        roots.push(home.join(".agents").join("skills"));
    }
    if trust_project {
        roots.push(cwd.join(".tapir").join("skills"));
    }
    roots
}

/// The skill root directories (delegates to [`roots_with`] with project trust).
#[cfg(test)]
pub fn roots(cwd: &Path, tapir_config: Option<&Path>) -> Vec<PathBuf> {
    roots_with(cwd, tapir_config, true)
}

/// All skills visible to tapir, honoring the CLI flags: `extra` are explicit
/// `--skill` paths
/// (a directory to scan, or a `SKILL.md`/`.md` file) — additive even when
/// `include_defaults` is false (`--no-skills` only disables default discovery).
/// `trust_project` gates the project-local root (`--no-approve` drops it).
pub fn load_with(
    cwd: &Path,
    tapir_config: Option<&Path>,
    extra: &[PathBuf],
    include_defaults: bool,
    trust_project: bool,
) -> Vec<Skill> {
    let mut raw = Vec::new();
    if include_defaults {
        for root in roots_with(cwd, tapir_config, trust_project) {
            load_from_dir(&root, true, 0, &mut raw);
        }
    }
    for path in extra {
        if path.is_dir() {
            load_from_dir(path, true, 0, &mut raw);
        } else if path.is_file()
            && let Some(s) = load_file(path)
        {
            raw.push(s);
        }
    }
    let mut seen = BTreeSet::new();
    raw.into_iter().filter(|s| seen.insert(s.name.clone())).collect()
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// The `<available_skills>` system-prompt section (empty when none are
/// model-invocable).
pub fn format_for_prompt(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> =
        skills.iter().filter(|s| !s.disable_model_invocation).collect();
    if visible.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    s.push_str("The following skills provide specialized instructions for specific tasks.\n");
    s.push_str("Use the read tool to load a skill's file when the task matches its description.\n");
    s.push_str(
        "When a skill file references a relative path, resolve it against the skill directory \
         (the parent of SKILL.md) and use that absolute path in tool commands.\n\n",
    );
    s.push_str("<available_skills>\n");
    for sk in visible {
        s.push_str("  <skill>\n");
        s.push_str(&format!("    <name>{}</name>\n", escape_xml(&sk.name)));
        s.push_str(&format!(
            "    <description>{}</description>\n",
            escape_xml(&sk.description)
        ));
        s.push_str(&format!(
            "    <location>{}</location>\n",
            escape_xml(&sk.file_path.display().to_string())
        ));
        s.push_str("  </skill>\n");
    }
    s.push_str("</available_skills>");
    s
}

/// A resolved `/skill:name [args]` invocation.
pub struct Invocation {
    pub name: String,
    /// The SKILL.md body (frontmatter stripped) — shown when the transcript
    /// entry is expanded.
    pub body: String,
    /// The full `<skill>` block (plus any args) sent to the model.
    pub model_text: String,
}

/// Resolve a `/skill:name [args]` command, or `None` if `text` isn't a skill
/// command (or the skill is unknown / unreadable).
pub fn invoke(text: &str, skills: &[Skill]) -> Option<Invocation> {
    let rest = text.trim().strip_prefix("/skill:")?;
    let (name, args) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    let skill = skills.iter().find(|s| s.name == name)?;
    let content = std::fs::read_to_string(&skill.file_path).ok()?;
    let (_, body) = parse_frontmatter(&content);
    let body = body.trim().to_string();
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{body}\n</skill>",
        skill.name,
        skill.file_path.display(),
        skill.base_dir.display(),
    );
    let model_text =
        if args.is_empty() { block } else { format!("{block}\n\n{args}") };
    Some(Invocation { name: skill.name.clone(), body, model_text })
}

/// Just the `<skill>` block text for `text` (convenience over [`invoke`]).
#[cfg(test)]
pub fn expand_command(text: &str, skills: &[Skill]) -> Option<String> {
    invoke(text, skills).map(|i| i.model_text)
}

/// A short, model-facing reminder that tapir supports skills and how to author
/// one — so "build me a skill for X" works (tapir embeds the essentials).
pub const HELP_FOR_SKILL: &str = "Skills (on-demand capability packages): you can create skills \
for the user. A skill is a directory containing a `SKILL.md` file, placed under \
`~/.agents/skills/<name>/` (global) or `<cwd>/.tapir/skills/<name>/` (project). `SKILL.md` starts \
with frontmatter — required keys `name` (lowercase letters, digits and hyphens, max 64 chars) and \
`description` (be specific: what the skill does AND when to use it, since the description is what \
makes the model load it); optional `disable-model-invocation: true` hides it from the prompt so it \
only runs via `/skill:name`. The rest of the file is freeform instructions; bundle helper scripts \
or reference docs alongside it and link them with paths relative to the skill directory. The model \
loads a matching skill with the read tool, or the user invokes it explicitly with `/skill:name`. \
When the user asks you to build a skill, create the directory and `SKILL.md` with the write tool \
and tell them how to invoke it.";

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(label: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("tapir-skills-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn loads_skill_md_root_and_formats_prompt() {
        let root = tmp("root");
        let skills_dir = root.join(".tapir/skills/pdf");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("SKILL.md"),
            "---\nname: pdf-tools\ndescription: Work with PDF files\n---\nDo the thing.",
        )
        .unwrap();

        let skills = load_from_roots(&[root.join(".tapir").join("skills")]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "pdf-tools");

        let prompt = format_for_prompt(&skills);
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>pdf-tools</name>"));
        assert!(prompt.contains("Work with PDF files"));

        let expanded =
            expand_command("/skill:pdf-tools make a report", &skills).unwrap();
        assert!(expanded.contains("<skill name=\"pdf-tools\""));
        assert!(expanded.contains("Do the thing."));
        assert!(expanded.ends_with("make a report"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn name_falls_back_to_dir_and_description_is_required() {
        let root = tmp("fallback");
        let dir = root.join(".tapir/skills/my-skill");
        std::fs::create_dir_all(&dir).unwrap();
        // No `name` → falls back to the directory name.
        std::fs::write(dir.join("SKILL.md"), "---\ndescription: x\n---\nbody")
            .unwrap();
        // A .md with no description is not a skill.
        let other = root.join(".tapir/skills/notes.md");
        std::fs::write(&other, "just notes, no frontmatter").unwrap();

        let skills = load_from_roots(&[root.join(".tapir").join("skills")]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "my-skill");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn roots_span_own_cross_agent_and_project() {
        let rs = roots(Path::new("/tmp/proj"), Some(Path::new("/cfg")));
        let has = |needle: &str| rs.iter().any(|r| r.ends_with(needle));
        assert!(rs.iter().any(|r| r.starts_with("/cfg")), "tapir's own global");
        assert!(has(".agents/skills"), "the shared cross-agent location");
        assert!(has(".tapir/skills"), "project skills");
        // A foreign agent's own dir is not read (such dirs symlink into ~/.agents).
        assert!(
            !has(".foreign-agent/skills"),
            "a foreign agent's dir is not a root"
        );
    }

    #[test]
    fn explicit_path_is_additive_even_with_defaults_disabled() {
        let root = tmp("explicit");
        let dir = root.join("custom/mover");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: mover\ndescription: move things\n---\nbody",
        )
        .unwrap();

        // `--no-skills` (include_defaults=false) + an explicit `--skill` dir:
        // discovery is off but the explicit skill still loads.
        let skills = load_with(
            Path::new("/nonexistent-cwd"),
            None,
            std::slice::from_ref(&dir),
            false,
            true,
        );
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "mover");

        // An explicit SKILL.md file path also loads.
        let by_file = load_with(
            Path::new("/x"),
            None,
            &[dir.join("SKILL.md")],
            false,
            true,
        );
        assert_eq!(by_file.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn disable_model_invocation_hides_from_prompt_only() {
        let root = tmp("hidden");
        let dir = root.join(".tapir/skills/secret");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: secret\ndescription: hush\ndisable-model-invocation: true\n---\nbody",
        )
        .unwrap();
        let skills = load_from_roots(&[root.join(".tapir").join("skills")]);
        assert_eq!(skills.len(), 1, "still loaded");
        assert_eq!(format_for_prompt(&skills), "", "but not in the prompt");
        assert!(
            expand_command("/skill:secret", &skills).is_some(),
            "explicit still works"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
