# Releasing

tapir is released from your machine — no CI yet. Releases are **GitHub-only**
and **single per version**: one `vX.Y.Z` tag and one GitHub release for the
whole workspace. The three crates share one version
(`[workspace.package] version` in `Cargo.toml`). tapir is consumed as a git
dependency, not from crates.io.

## Prerequisites

- `just` installed and logged in with `gh` — the `release` recipe reads the
  token via `gh auth token`. No crates.io token is needed.
- A clean working tree (the `release` recipe refuses to run otherwise).

## Steps

1. Bump `version` in `[workspace.package]` (`Cargo.toml`).
2. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md`.
3. Commit and push:

   ```sh
   git add Cargo.toml Cargo.lock CHANGELOG.md
   git commit -m "docs: changelog for vX.Y.Z"
   git push
   ```

4. Preview the notes, then cut the release:

   ```sh
   just release-notes   # sanity-check the extracted notes
   just release         # tag vX.Y.Z + create the GitHub release
   ```

`just release` reads the version from `Cargo.toml`, tags `vX.Y.Z`, pushes the
tag, and creates a GitHub release whose body is the matching `CHANGELOG.md`
section.

## Why GitHub-only?

The `tapir` name on crates.io belongs to an unrelated crate (`lcnr/tapir`, a
tapping library), and the project lives under the `tapir` brand (domain, email)
regardless. Publishing only to GitHub keeps the name and skips the registry;
consumers depend on it via git:

```toml
tapir = { git = "https://github.com/tapir-dev/tapir" }
```
