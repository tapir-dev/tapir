# tapir task runner. Run `just` to list recipes.
# Releases are GitHub-only and single per version: one `vX.Y.Z` tag and one
# GitHub release for the whole workspace. See RELEASING.md.

# List available recipes.
default:
    @just --list

# Type-check the whole workspace.
check:
    cargo check --workspace

# Run the test suite (some tests bind localhost / spawn processes).
test:
    cargo test --workspace

# Lint the dependency graph: advisories, licenses, bans, sources.
deny:
    cargo deny check

# Regenerate THIRD-PARTY-LICENSE from the dependency graph.
third-party:
    cargo about generate about.hbs | grep -vE '^  - tapir(-core|-ai)? ' > THIRD-PARTY-LICENSE

# Preview the release notes for the version currently in Cargo.toml.
release-notes:
    #!/usr/bin/env sh
    set -eu
    version=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
    awk -v v="$version" '$0 ~ "^## \\[" v "\\]" {p=1; next} p && /^## \[/ {exit} p' CHANGELOG.md

# Bump the version + update CHANGELOG.md and commit first, then run this:
# tag the Cargo.toml version and cut one GitHub release (notes from CHANGELOG).
release:
    #!/usr/bin/env sh
    set -eu
    if [ -n "$(git status --porcelain)" ]; then echo "working tree dirty; commit first" >&2; exit 1; fi
    version=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
    notes=$(awk -v v="$version" '$0 ~ "^## \\[" v "\\]" {p=1; next} p && /^## \[/ {exit} p' CHANGELOG.md)
    git tag -a "v$version" -m "tapir v$version"
    git push origin "v$version"
    GIT_TOKEN=$(gh auth token) gh release create "v$version" --verify-tag --title "tapir v$version" --notes "$notes"
