# Releasing

tapir is released from your machine with
[`release-plz`](https://release-plz.dev/), driven through `just` — no CI yet.

Versioning is **independent per crate**: each crate bumps only when its own
files change, and release-plz propagates a bump to dependents when needed (e.g.
the `tapir` facade re-exports `tapir-ai`, so a new provider bumps both, but not
`tapir-core`). Releases are **GitHub-only**: a git tag and a GitHub release per
changed crate, with no `cargo publish`. tapir is consumed as a git dependency,
not from crates.io.

Bumps and changelogs are derived from [Conventional
Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, …), which this
repo uses.

## Prerequisites

- `just` and `release-plz` installed (`pacman -S just release-plz`).
- Logged in with `gh` — the recipes read the token via `gh auth token`. No
  crates.io token is needed.
- A clean working tree. `release-plz` refuses to run with uncommitted or
  untracked files. The local-only `Makefile` and `docs/` are listed in
  `.git/info/exclude` so git ignores them without touching the tracked
  `.gitignore`.

## Steps

```sh
just release-prepare   # per-crate version bump + CHANGELOG (release-plz update)
# review the diff; adjust a version by hand if you disagree with the bump, then:
git add Cargo.toml Cargo.lock '**/CHANGELOG.md'
git commit -m "chore: release"
git push

just release-dry       # preview which crates would be tagged/released
just release           # tag + cut a GitHub release for each changed crate
```

Per-crate tags follow `<crate>-v<version>` (e.g. `tapir-ai-v0.2.0`). release-plz
uses them as the baseline for the next bump.

## Why GitHub-only?

The `tapir` name on crates.io belongs to an unrelated crate (`lcnr/tapir`, a
tapping library), and the project lives under the `tapir` brand (domain, email)
regardless. Publishing only to GitHub keeps the name and skips the registry;
consumers depend on it via git:

```toml
tapir = { git = "https://github.com/tapir-dev/tapir" }
```

To publish to crates.io later, set `publish = true` (and likely
`semver_check = true`) in `release-plz.toml` and add a crates.io token.
