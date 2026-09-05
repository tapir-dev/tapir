// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Compile-fail cases for the tool system, checked with `trybuild`. The
//! canonical disallowed use is a heterogeneous array of `#[tool]` fns: each
//! erases to its own zero-sized type, so they cannot share an array — authors
//! must chain `.tool()` (or erase to `Arc<dyn ErasedTool>`) instead.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
