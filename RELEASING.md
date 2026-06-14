# Releasing

tapir is released from your machine through `just` — no CI yet. Each crate is
versioned and released **independently**, and releases are **GitHub-only**: a
`<crate>-vX.Y.Z` tag and a GitHub release per crate, with no `cargo publish`.
tapir is consumed as a git dependency, not from crates.io.

You decide which crates a change touches. A change to `tapir-ai` bumps
`tapir-ai`; because the `tapir` facade re-exports `tapir-ai`, it usually bumps
`tapir` too, but **not** `tapir-core` (whose own API is unchanged).

## Prerequisites

- `just` installed and logged in with `gh` — the recipes read the token via
  `gh auth token`. No crates.io token is needed.
- A clean working tree (the `release` recipe refuses to run otherwise).

## Steps

For each crate you're releasing (e.g. `tapir-ai`):

1. Bump `version` in `tapir-ai/Cargo.toml`.
2. If a dependent must require the new version, bump it in the root
   `[workspace.dependencies]` (e.g. `tapir-ai = { ..., version = "0.2.0" }`)
   and release that dependent too.
3. Add a `## [X.Y.Z] - YYYY-MM-DD` section to `tapir-ai/CHANGELOG.md`.
4. Commit and push:

   ```sh
   git add tapir-ai/Cargo.toml Cargo.lock tapir-ai/CHANGELOG.md
   git commit -m "feat(tapir-ai): ..."
   git push
   ```

5. Preview the notes, then cut the release:

   ```sh
   just release-notes tapir-ai   # sanity-check the extracted notes
   just release tapir-ai         # tag tapir-ai-vX.Y.Z + create the GitHub release
   ```

`just release <crate>` reads the version from the crate's `Cargo.toml`, tags
`<crate>-vX.Y.Z`, pushes the tag, and creates a GitHub release whose body is the
matching `CHANGELOG.md` section.

## Why GitHub-only?

The `tapir` name on crates.io belongs to an unrelated crate (`lcnr/tapir`, a
tapping library), and the project lives under the `tapir` brand (domain, email)
regardless. Publishing only to GitHub keeps the name and skips the registry;
consumers depend on it via git:

```toml
tapir = { git = "https://github.com/tapir-dev/tapir" }
```
