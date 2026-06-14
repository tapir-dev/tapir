# tapir task runner. Run `just` to list recipes.
# Releases use independent, per-crate versioning via release-plz, GitHub-only
# (tag + GitHub release per changed crate, no crates.io). See RELEASING.md.

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

# Release step 1: per-crate version bump + changelog from the commit history.
release-prepare:
    release-plz update

# Preview what would be released without tagging or creating anything.
release-dry:
    GIT_TOKEN=$(gh auth token) release-plz release --dry-run

# Release step 2: tag and cut a GitHub release for each changed crate.
release:
    GIT_TOKEN=$(gh auth token) release-plz release
