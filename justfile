# tapir task runner. Run `just` to list recipes.
# Releases are GitHub-only (tag + GitHub release, no crates.io). See RELEASING.md
# for the why and the prerequisites (GIT_TOKEN, clean working tree).

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

# Release step 1: bump versions and write changelogs for review, then commit.
release-prepare:
    release-plz update

# Preview the release without tagging or creating anything.
release-dry:
    GIT_TOKEN=$(gh auth token) release-plz release --dry-run

# Release step 2: push tags and cut GitHub releases (no crates.io).
release:
    GIT_TOKEN=$(gh auth token) release-plz release
