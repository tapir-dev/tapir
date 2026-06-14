# Releasing

tapir is released from your machine with
[`release-plz`](https://release-plz.dev/), driven through `just` — no CI yet.
Releases are **GitHub-only**: a git tag and a GitHub release per crate, with no
`cargo publish`. tapir is consumed as a git dependency, not from crates.io.

The three crates share one version (`[workspace.package] version` in
`Cargo.toml`) and are released together.

## Prerequisites

- `just`, `release-plz`, and `cargo-about` installed
  (`pacman -S just release-plz cargo-about`).
- A GitHub token: the recipes read it via `gh auth token`, so just stay logged
  in with `gh`. No crates.io token is needed, since we do not publish there.
- A clean working tree. `release-plz` refuses to run with uncommitted or
  untracked files. The local-only `Makefile` and `docs/` are listed in
  `.git/info/exclude` so git ignores them without touching the tracked
  `.gitignore`.

## Steps

```sh
just release-prepare   # bump versions + write CHANGELOGs (release-plz update)
# review the diff; adjust the version level by hand if needed, then:
git add Cargo.toml '**/CHANGELOG.md' CHANGELOG.md
git commit -m "Release v<version>"
git push

just release-dry       # preview
just release           # push tags + cut GitHub releases
```

Because we do not use conventional commits, the version bump from
`release-plz update` is best-effort — review it and edit `version` in
`Cargo.toml` by hand if you want a different level (e.g. a minor bump).

## Why GitHub-only?

The `tapir` name on crates.io belongs to an unrelated crate (`lcnr/tapir`, a
tapping library), and the project lives under the `tapir` brand (domain,
email) regardless. Publishing only to GitHub keeps the name and skips the
registry entirely; consumers depend on it via git:

```toml
tapir = { git = "https://github.com/tapir-dev/tapir" }
```

To publish to crates.io later, set `publish = true` (and likely
`semver_check = true`) in `release-plz.toml`, add a crates.io token, and adjust
the `just release` recipe.
