//! Project & global context files (`AGENTS.md`/`CLAUDE.md` loading).
//!
//! Per directory we take the first of `CANDIDATES` that exists. We gather the
//! global file (from the config dir) plus the project files walked from the cwd
//! up to the filesystem root, ordered root→cwd so the most specific one is last.
//! They're injected into the system prompt as `<project_instructions>` blocks.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Context file names, in priority order (first match per directory wins).
const CANDIDATES: &[&str] = &["AGENTS.md", "CLAUDE.md", "TAPIR.md"];

pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

/// The first existing, non-empty context file in `dir`.
fn from_dir(dir: &Path) -> Option<ContextFile> {
    for name in CANDIDATES {
        let path = dir.join(name);
        if let Ok(content) = std::fs::read_to_string(&path)
            && !content.trim().is_empty()
        {
            return Some(ContextFile { path, content });
        }
    }
    None
}

/// Global context (from `global_dir`) followed by project context (cwd and its
/// ancestors, ordered root→cwd), with duplicate paths removed.
#[cfg(test)]
pub fn load(cwd: &Path, global_dir: Option<&Path>) -> Vec<ContextFile> {
    load_with(cwd, global_dir, true)
}

/// Like `load`, but the project context (the cwd-and-ancestors files) is
/// collected only when `include_project` — `--no-approve` drops it, keeping just
/// the global context.
pub fn load_with(
    cwd: &Path,
    global_dir: Option<&Path>,
    include_project: bool,
) -> Vec<ContextFile> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();

    if let Some(global) = global_dir.and_then(from_dir) {
        seen.insert(global.path.clone());
        out.push(global);
    }

    if !include_project {
        return out;
    }

    // Walk cwd → root, collecting each directory's context file.
    let mut ancestors = Vec::new();
    let mut dir = Some(cwd);
    while let Some(d) = dir {
        if let Some(cf) = from_dir(d)
            && seen.insert(cf.path.clone())
        {
            ancestors.push(cf);
        }
        dir = d.parent();
    }
    ancestors.reverse(); // root → cwd (most specific last)
    out.extend(ancestors);
    out
}

/// Render context files as the system-prompt `<project_context>` section (empty
/// string when there are none).
pub fn format_for_prompt(files: &[ContextFile]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "<project_context>\nProject-specific instructions and guidelines:\n\n",
    );
    for f in files {
        s.push_str(&format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>\n\n",
            f.path.display(),
            f.content.trim_end()
        ));
    }
    s.push_str("</project_context>");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(label: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("tapir-ctx-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn first_candidate_per_dir_wins() {
        let dir = tmp("first");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("AGENTS.md"), "agents wins").unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "claude loses").unwrap();
        let cf = from_dir(&dir).unwrap();
        assert!(cf.content.contains("agents wins"));
        assert!(cf.path.ends_with("AGENTS.md"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_global_then_ancestors_root_to_cwd() {
        let base = tmp("walk");
        let global = base.join("global");
        let proj = base.join("proj");
        let sub = proj.join("nested");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        std::fs::write(global.join("TAPIR.md"), "global rules").unwrap();
        std::fs::write(proj.join("AGENTS.md"), "project rules").unwrap();
        std::fs::write(sub.join("CLAUDE.md"), "nested rules").unwrap();

        let files = load(&sub, Some(&global));
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        // Global first, then root→cwd (proj before its nested child).
        assert_eq!(names, ["TAPIR.md", "AGENTS.md", "CLAUDE.md"]);

        let prompt = format_for_prompt(&files);
        assert!(prompt.contains("<project_context>"));
        assert!(prompt.contains("global rules"));
        assert!(prompt.contains("nested rules"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn untrusted_keeps_only_global_context() {
        let base = tmp("untrusted");
        let global = base.join("global");
        let proj = base.join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        std::fs::write(global.join("TAPIR.md"), "global rules").unwrap();
        std::fs::write(proj.join("AGENTS.md"), "project rules").unwrap();

        // `--no-approve` (include_project = false) drops the cwd's AGENTS.md.
        let files = load_with(&proj, Some(&global), false);
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["TAPIR.md"], "only the global context survives");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn empty_when_no_files() {
        let dir = tmp("none");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(load(&dir, None).is_empty());
        assert_eq!(format_for_prompt(&[]), "");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
