# Releasing

tapir is released from your machine with
[`release-plz`](https://release-plz.dev/) — no CI yet. Releases are
**GitHub-only**: a git tag and a GitHub release per crate, with no
`cargo publish`. tapir is consumed as a git dependency, not from crates.io.

The three crates share one version (`[workspace.package] version` in
`Cargo.toml`) and are released together.

The flow mirrors the two-step shape oxc uses, but driven locally:

1. **Prepare** — bump versions and write changelogs (`release-plz update`).
2. **Publish** — push tags and cut GitHub releases (`release-plz release`).

## Prerequisites

- `release-plz` installed (`pacman -S release-plz`).
- A GitHub token: `export GIT_TOKEN=$(gh auth token)`. No crates.io token is
  needed, since we do not publish there.
- A clean working tree. `release-plz` refuses to run with uncommitted or
  untracked files. The local-only `Makefile` and `docs/` are listed in
  `.git/info/exclude` so git ignores them without touching the tracked
  `.gitignore`.

## Steps

### 1. Prepare

```sh
release-plz update
```

This bumps `[workspace.package] version` and writes a `CHANGELOG.md` per crate
from the git history. Because we do not use conventional commits, the bump
level is best-effort — review the diff, and if you want a different level
(e.g. a minor bump), edit `version` in `Cargo.toml` by hand before committing.

Then commit and push:

```sh
git add Cargo.toml '**/CHANGELOG.md' CHANGELOG.md
git commit -m "Release v$(grep '^version' Cargo.toml | head -1 | cut -d'\"' -f2)"
git push
```

### 2. Publish

Dry-run first to see what would happen:

```sh
GIT_TOKEN=$(gh auth token) release-plz release --dry-run
```

Then for real:

```sh
GIT_TOKEN=$(gh auth token) release-plz release
```

For each crate at a version without a tag yet, this pushes a
`<crate>-v<version>` tag and creates a GitHub release with the changelog notes.

## Why GitHub-only?

The `tapir` name on crates.io belongs to an unrelated crate (`lcnr/tapir`, a
tapping library), and the project lives under the `tapir` brand (domain,
email) regardless. Publishing only to GitHub keeps the name and skips the
registry entirely; consumers depend on it via git:

```toml
tapir = { git = "https://github.com/tapir-dev/tapir" }
```

To publish to crates.io later, set `publish = true` (and likely
`semver_check = true`) in `release-plz.toml` and add a crates.io token.
