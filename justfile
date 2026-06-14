# tapir task runner. Run `just` to list recipes.
# Releases are GitHub-only and per crate: bump one crate, tag `<crate>-vX.Y.Z`,
# cut a GitHub release. Each crate is versioned independently. See RELEASING.md.

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

# Preview the release notes for a crate, e.g. `just release-notes tapir-ai`.
release-notes crate:
    #!/usr/bin/env sh
    set -eu
    version=$(grep '^version' "{{crate}}/Cargo.toml" | head -1 | cut -d'"' -f2)
    awk -v v="$version" '$0 ~ "^## \\[" v "\\]" {p=1; next} p && /^## \[/ {exit} p' "{{crate}}/CHANGELOG.md"

# Bump the crate's version + update its CHANGELOG.md and commit first, then run
# this to tag <crate>-vX.Y.Z and cut its GitHub release. E.g. `just release tapir-ai`.
release crate:
    #!/usr/bin/env sh
    set -eu
    test -f "{{crate}}/Cargo.toml" || { echo "no such crate: {{crate}}" >&2; exit 1; }
    if [ -n "$(git status --porcelain)" ]; then echo "working tree dirty; commit first" >&2; exit 1; fi
    version=$(grep '^version' "{{crate}}/Cargo.toml" | head -1 | cut -d'"' -f2)
    tag="{{crate}}-v$version"
    notes=$(awk -v v="$version" '$0 ~ "^## \\[" v "\\]" {p=1; next} p && /^## \[/ {exit} p' "{{crate}}/CHANGELOG.md")
    git tag -a "$tag" -m "{{crate}} v$version"
    git push origin "$tag"
    GIT_TOKEN=$(gh auth token) gh release create "$tag" --verify-tag --title "{{crate}} v$version" --notes "$notes"
