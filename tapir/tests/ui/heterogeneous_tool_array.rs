// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

// Two `#[tool]` fns erase to distinct zero-sized types, so a literal array of
// them does not typecheck. Authors register a heterogeneous set by chaining
// `.tool()` (or erasing each to `Arc<dyn ErasedTool>`).

use schemars::JsonSchema;
use serde::Deserialize;
use tapir::tool;

#[derive(Deserialize, JsonSchema)]
struct AlphaArgs {
    x: String,
}

#[derive(Deserialize, JsonSchema)]
struct BetaArgs {
    y: String,
}

#[tool]
/// Alpha.
async fn alpha(args: AlphaArgs) -> String {
    args.x
}

#[tool]
/// Beta.
async fn beta(args: BetaArgs) -> String {
    args.y
}

fn main() {
    let _tools = [alpha, beta];
}
