# Releasing

tapir is released from your machine with
[`release-plz`](https://release-plz.dev/) — no CI yet. The three crates share
one version (`[workspace.package] version` in `Cargo.toml`) and are released
together.

The flow mirrors the two-step shape oxc uses, but driven locally:

1. **Prepare** — bump versions and write changelogs (`release-plz update`).
2. **Publish** — push tags, publish to crates.io, cut GitHub releases
   (`release-plz release`).

## Prerequisites

- `release-plz` installed (`pacman -S release-plz`).
- A crates.io token: run `cargo login` once (stored in
  `~/.cargo/credentials.toml`), or export `CARGO_REGISTRY_TOKEN`.
- A GitHub token for the release notes step:
  `export GIT_TOKEN=$(gh auth token)`.
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
git add Cargo.toml Cargo.lock '**/CHANGELOG.md' CHANGELOG.md
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

For each crate not yet on crates.io, this pushes a `<crate>-v<version>` tag,
publishes to crates.io in dependency order (`tapir-ai` → `tapir-core` →
`tapir`), and creates a GitHub release.

## Known blocker: the `tapir` crate name

The `tapir` name on crates.io is already taken by an unrelated crate
(`lcnr/tapir`, a tapping library). `tapir-core` and `tapir-ai` are free. Until
this is resolved, publishing the `tapir` facade will fail. Options:

- Rename the facade crate to a free name (e.g. `tapir-rs`), or
- Mark the facade as unpublished (`publish = false` in `tapir/Cargo.toml`) and
  release only `tapir-core` and `tapir-ai`, or
- Ask the current owner to transfer the name.

`tapir-core` and `tapir-ai` can be released today regardless.
