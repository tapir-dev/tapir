#!/bin/bash
# Blocks `cargo test` in favor of cargo-nextest, per CLAUDE.md.
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# Match `cargo test` / `cargo t` as a subcommand, allowing global flags in
# between (e.g. `cargo --release test`). Word boundaries avoid matching things
# like `cargo nextest` or `cargo test-foo`.
if echo "$COMMAND" | grep -Eq '\bcargo\b([[:space:]]+-[^[:space:]]+)*[[:space:]]+(test|t)([[:space:]]|$)'; then
    echo "Blocked: this project uses cargo-nextest, not 'cargo test'." >&2
    echo "Use one of instead:" >&2
    echo "  just test              # fast tests (default)" >&2
    echo "  cargo nextest run <name>   # a specific test" >&2
    echo "  just test-doc          # doctests (nextest doesn't run these)" >&2
    exit 2
fi

exit 0
