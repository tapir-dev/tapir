//! Output truncation shared by the tools: keep at most `MAX_LINES` lines or
//! `MAX_BYTES` bytes, whichever comes first.

pub const MAX_LINES: usize = 2000;
pub const MAX_BYTES: usize = 50 * 1024;
/// Per-line cap for grep matches.
pub const GREP_MAX_LINE: usize = 500;

pub struct Truncation {
    pub content: String,
    pub truncated: bool,
    /// "lines" or "bytes" when truncated — computed for completeness; not yet
    /// surfaced in the UI.
    pub by: Option<&'static str>,
    pub total_lines: usize,
    pub output_lines: usize,
}

/// Logical line count: split on `\n`, ignoring a single trailing newline.
fn count_lines(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    let n = s.split('\n').count();
    if s.ends_with('\n') { n - 1 } else { n }
}

/// Keep the first lines that fit within the line and byte limits.
pub fn head(content: &str) -> Truncation {
    let total_lines = count_lines(content);
    let lines: Vec<&str> = content.split('\n').collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut by = None;
    for (i, line) in lines.iter().enumerate() {
        if i >= MAX_LINES {
            by = Some("lines");
            break;
        }
        // +1 for the newline that rejoins.
        let add = line.len() + if i > 0 { 1 } else { 0 };
        if bytes + add > MAX_BYTES && !kept.is_empty() {
            by = Some("bytes");
            break;
        }
        bytes += add;
        kept.push(line);
    }
    let output_lines = kept.len();
    Truncation {
        content: kept.join("\n"),
        truncated: by.is_some(),
        by,
        total_lines,
        output_lines,
    }
}

/// Keep the last lines that fit within the line and byte limits.
pub fn tail(content: &str) -> Truncation {
    let total_lines = count_lines(content);
    let lines: Vec<&str> = content.split('\n').collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let mut by = None;
    for (i, line) in lines.iter().rev().enumerate() {
        if i >= MAX_LINES {
            by = Some("lines");
            break;
        }
        let add = line.len() + if i > 0 { 1 } else { 0 };
        if bytes + add > MAX_BYTES && !kept.is_empty() {
            by = Some("bytes");
            break;
        }
        bytes += add;
        kept.push(line);
    }
    kept.reverse();
    let output_lines = kept.len();
    Truncation {
        content: kept.join("\n"),
        truncated: by.is_some(),
        by,
        total_lines,
        output_lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_keeps_first_lines_when_over_line_limit() {
        let content = (0..MAX_LINES + 50)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let t = head(&content);
        assert!(t.truncated);
        assert_eq!(t.by, Some("lines"));
        assert_eq!(t.output_lines, MAX_LINES);
        assert!(t.content.starts_with("0\n1\n"));
    }

    #[test]
    fn tail_keeps_last_lines() {
        let content = (0..MAX_LINES + 50)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let t = tail(&content);
        assert!(t.truncated);
        assert_eq!(t.output_lines, MAX_LINES);
        assert!(t.content.ends_with(&format!("{}", MAX_LINES + 49)));
    }

    #[test]
    fn no_truncation_for_small_content() {
        let t = head("a\nb\nc");
        assert!(!t.truncated);
        assert_eq!(t.content, "a\nb\nc");
        assert_eq!(t.total_lines, 3);
    }
}
