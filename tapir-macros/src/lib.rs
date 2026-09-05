// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The `#[tool]` proc-macro for `tapir`, re-exported through the `tapir` crate.
//! A proc-macro crate cannot also export normal items, which is why this lives
//! apart from the library.
//!
//! `#[tool]` maps an annotated `async fn` to a typed `Tool` impl: it reads the
//! fn name as the tool name, the doc comment as the description, the single
//! argument parameter as `type Args`, and the return type as
//! `type Output`/`type Error`. A `ctx: &ToolCtx` parameter makes the tool
//! contextual. The fn identifier is shadowed by a generated unit struct, so
//! `.tool(the_fn)` registers a value.

use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    Attribute, Expr, ExprLit, FnArg, ItemFn, Lit, Meta, ReturnType, Token, Type,
};

/// Turn an annotated `async fn` into a tool.
///
/// The fn is replaced by a zero-sized struct of the same name implementing the
/// typed `Tool` trait. Accepts `#[tool(read_only)]` to mark the tool
/// parallel-safe, and `#[tool(name = "...")]` to override the tool name.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as ItemFn);
    let args = match Args::parse(attr.into()) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };
    match expand(func, args) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Parsed `#[tool(...)]` attribute arguments.
#[derive(Default)]
struct Args {
    read_only: bool,
    name: Option<String>,
}

impl Args {
    fn parse(tokens: proc_macro2::TokenStream) -> syn::Result<Self> {
        let mut out = Args::default();
        if tokens.is_empty() {
            return Ok(out);
        }
        let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
        let metas = syn::parse::Parser::parse2(parser, tokens)?;
        for meta in metas {
            match &meta {
                Meta::Path(path) if path.is_ident("read_only") => {
                    out.read_only = true;
                }
                Meta::NameValue(nv) if nv.path.is_ident("name") => {
                    out.name = Some(lit_str(&nv.value)?);
                }
                _ => {
                    return Err(syn::Error::new(
                        meta.span(),
                        "expected `read_only` or `name = \"...\"`",
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// Read a string literal out of a name-value expression.
fn lit_str(expr: &Expr) -> syn::Result<String> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Str(s), ..
    }) = expr
    {
        Ok(s.value())
    } else {
        Err(syn::Error::new(expr.span(), "expected a string literal"))
    }
}

fn expand(func: ItemFn, args: Args) -> syn::Result<proc_macro2::TokenStream> {
    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new(
            func.sig.fn_token.span(),
            "#[tool] requires an `async fn`",
        ));
    }

    let fn_ident = func.sig.ident.clone();
    let vis = func.vis.clone();
    let tool_name = args.name.unwrap_or_else(|| fn_ident.to_string());
    let description = doc_string(&func.attrs);

    let params: Vec<&FnArg> = func.sig.inputs.iter().collect();
    let (args_pat, args_ty) = match params.first() {
        Some(FnArg::Typed(pat)) => (&pat.pat, &pat.ty),
        _ => {
            return Err(syn::Error::new(
                func.sig.span(),
                "#[tool] fn needs an arguments parameter as its first parameter",
            ));
        }
    };

    // A second parameter is the contextual `ctx: &ToolCtx`. When absent, the
    // generated `execute` still takes a `ctx`, ignored.
    let ctx_pat = match params.get(1) {
        Some(FnArg::Typed(pat)) => {
            let pat = &pat.pat;
            quote!(#pat)
        }
        Some(FnArg::Receiver(recv)) => {
            return Err(syn::Error::new(
                recv.span(),
                "#[tool] fn cannot take `self`",
            ));
        }
        None => quote!(_ctx),
    };
    if params.len() > 2 {
        let msg = "#[tool] fn takes at most an arguments parameter and a `ctx: &ToolCtx` parameter; stream updates by implementing `Tool` directly";
        return Err(syn::Error::new(params[2].span(), msg));
    }

    let block = &func.block;
    let (output_ty, error_ty) = split_return(&func.sig.output);

    // Result-returning bodies pass through; a plain return type is wrapped in
    // `Ok`, with `Infallible` as the error so `?` and the trait bound still hold.
    let (error_ty, body) = match error_ty {
        Some(error_ty) => (quote!(#error_ty), quote!(#block)),
        None => (
            quote!(::core::convert::Infallible),
            quote! {{
                ::core::result::Result::Ok((async move #block).await)
            }},
        ),
    };

    let concurrency = if args.read_only {
        quote! {
            fn concurrency(&self) -> ::tapir::tool::Concurrency {
                ::tapir::tool::Concurrency::Safe
            }
        }
    } else {
        quote!()
    };

    let doc_attrs: Vec<&Attribute> = func
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .collect();

    Ok(quote! {
        #(#doc_attrs)*
        #[allow(non_camel_case_types)]
        #vis struct #fn_ident;

        #[::tapir::__private::async_trait]
        impl ::tapir::tool::Tool for #fn_ident {
            type Args = #args_ty;
            type Output = #output_ty;
            type Error = #error_ty;

            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            #concurrency

            async fn execute(
                &self,
                #args_pat: #args_ty,
                #ctx_pat: &::tapir::tool::ToolCtx,
                _on_update: &mut ::tapir::tool::UpdateSink<'_>,
            ) -> ::core::result::Result<Self::Output, Self::Error> {
                #body
            }
        }
    })
}

/// Split a return type into `(Output, Option<Error>)`. A `Result<O, E>` yields
/// `(O, Some(E))`; anything else yields `(that-type, None)`, with a missing
/// return type read as `()`.
fn split_return(
    ret: &ReturnType,
) -> (proc_macro2::TokenStream, Option<proc_macro2::TokenStream>) {
    let ty = match ret {
        ReturnType::Default => return (quote!(()), None),
        ReturnType::Type(_, ty) => ty,
    };
    if let Type::Path(path) = ty.as_ref()
        && let Some(seg) = path.path.segments.last()
        && seg.ident == "Result"
        && let syn::PathArguments::AngleBracketed(generics) = &seg.arguments
    {
        let types: Vec<&Type> = generics
            .args
            .iter()
            .filter_map(|a| match a {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect();
        if let [ok, err] = types.as_slice() {
            return (quote!(#ok), Some(quote!(#err)));
        }
    }
    (quote!(#ty), None)
}

/// Concatenate `#[doc = "..."]` lines into a single trimmed description.
fn doc_string(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
        {
            lines.push(s.value().trim().to_owned());
        }
    }
    lines.join("\n").trim().to_owned()
}
