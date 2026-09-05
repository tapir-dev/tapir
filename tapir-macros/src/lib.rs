// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The `#[tool]` proc-macro for `tapir`, re-exported through the `tapir` crate.
//! A proc-macro crate cannot also export normal items, which is why this lives
//! apart from the library.
//!
//! Skeleton: the attribute is a passthrough for now; it maps an async fn to a
//! `Tool` impl in the tool ticket.

use proc_macro::TokenStream;

/// Turns an annotated async fn into a tool. Skeleton passthrough; the mapping to
/// a `Tool` impl and portable schema emission land in a later ticket.
#[proc_macro_attribute]
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
