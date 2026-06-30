// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attribute macros for concise components-v2 profile config structs.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, Attribute, Field, Fields, GenericArgument, ItemStruct, LitStr,
    PathArguments, Type,
};

/// Adds standard config derives, strict serde behavior, and a stable profile type.
#[proc_macro_attribute]
pub fn profile_config(args: TokenStream, input: TokenStream) -> TokenStream {
    let kind = parse_macro_input!(args as LitStr);
    let input = parse_macro_input!(input as ItemStruct);
    emit_profile_config_struct(input, kind).into()
}

// Re-emits the struct with Switchyard's standard profile-config surface.
fn emit_profile_config_struct(input: ItemStruct, kind: LitStr) -> proc_macro2::TokenStream {
    let ident = input.ident.clone();
    let raw_ident = format_ident!("__{}Raw", ident);
    let named_fields = match &input.fields {
        Fields::Named(fields) => fields,
        Fields::Unnamed(_) | Fields::Unit => {
            return syn::Error::new_spanned(
                input,
                "profile_config only supports structs with named fields",
            )
            .to_compile_error();
        }
    };

    let mut original_fields = Vec::new();
    let mut raw_fields = Vec::new();
    let mut build_fields = Vec::new();

    for field in &named_fields.named {
        let Some(field_ident) = field.ident.clone() else {
            return syn::Error::new_spanned(field, "profile_config requires named fields")
                .to_compile_error();
        };
        let has_profile_target = has_profile_target_attr(&field.attrs);
        let cleaned_field = field_without_profile_target_attr(field);
        let raw_ty = if has_profile_target {
            match profile_target_raw_ty(&field.ty) {
                Ok(raw_ty) => raw_ty,
                Err(error) => return error.to_compile_error(),
            }
        } else {
            field.ty.clone()
        };
        let mut raw_field = cleaned_field.clone();
        raw_field.ty = raw_ty;
        let value = if has_profile_target {
            match profile_target_resolver(&field_ident, &field.ty) {
                Ok(value) => value,
                Err(error) => return error.to_compile_error(),
            }
        } else {
            quote! { raw.#field_ident }
        };

        original_fields.push(cleaned_field);
        raw_fields.push(raw_field);
        build_fields.push(quote! { #field_ident: #value });
    }

    let mut original = input.clone();
    original.fields = Fields::Named(syn::FieldsNamed {
        brace_token: named_fields.brace_token,
        named: original_fields.into_iter().collect(),
    });

    quote! {
        #[derive(
            Clone,
            Debug,
            PartialEq,
            ::serde::Serialize,
            ::serde::Deserialize,
        )]
        #[serde(deny_unknown_fields)]
        #original

        #[derive(::serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct #raw_ident {
            #(#raw_fields,)*
        }

        impl #ident {
            /// Stable discriminator used when this profile appears in serialized config.
            pub const PROFILE_TYPE: &'static str = #kind;

            /// Returns the stable serialized discriminator for this profile config.
            pub fn profile_type(&self) -> &'static str {
                Self::PROFILE_TYPE
            }
        }

        impl ::switchyard_components_v2::__private::ProfileConfigDefinition for #ident {
            const PROFILE_TYPE: &'static str = #kind;

            fn parse_profile_config(
                value: ::serde_json::Value,
                env: &::switchyard_components_v2::__private::ProfileBuildEnv<'_>,
            ) -> ::switchyard_core::Result<Self> {
                let raw: #raw_ident = ::serde_json::from_value(value).map_err(|error| {
                    ::switchyard_core::SwitchyardError::InvalidConfig(format!(
                        "failed to parse {} profile config: {error}",
                        #kind
                    ))
                })?;
                Ok(Self {
                    #(#build_fields,)*
                })
            }
        }

        const _: () = {
            fn assert_profile_config<T: ::switchyard_components_v2::ProfileConfig>() {}
            let _ = assert_profile_config::<#ident>;
        };
    }
}

// Removes the helper attribute before the real struct reaches the compiler.
fn field_without_profile_target_attr(field: &Field) -> Field {
    let mut cleaned = field.clone();
    cleaned.attrs = field
        .attrs
        .iter()
        .filter(|attr| !is_profile_target_attr(attr))
        .cloned()
        .collect();
    cleaned
}

fn has_profile_target_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(is_profile_target_attr)
}

fn is_profile_target_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("profile_target")
}

fn profile_target_raw_ty(ty: &Type) -> syn::Result<Type> {
    if is_type_ident(ty, "LlmTarget") {
        Ok(syn::parse_quote! { ::switchyard_core::LlmTargetId })
    } else if let Some(inner) = single_generic_arg(ty, "Vec") {
        if is_type_ident(inner, "LlmTarget") {
            Ok(syn::parse_quote! { Vec<::switchyard_core::LlmTargetId> })
        } else {
            Err(syn::Error::new_spanned(
                ty,
                "#[profile_target] Vec fields must contain LlmTarget",
            ))
        }
    } else if let Some(inner) = single_generic_arg(ty, "Option") {
        if is_type_ident(inner, "LlmTarget") {
            Ok(syn::parse_quote! { Option<::switchyard_core::LlmTargetId> })
        } else {
            Err(syn::Error::new_spanned(
                ty,
                "#[profile_target] Option fields must contain LlmTarget",
            ))
        }
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "#[profile_target] supports LlmTarget, Vec<LlmTarget>, and Option<LlmTarget>",
        ))
    }
}

fn profile_target_resolver(
    field_ident: &syn::Ident,
    ty: &Type,
) -> syn::Result<proc_macro2::TokenStream> {
    if is_type_ident(ty, "LlmTarget") {
        Ok(quote! { env.target(&raw.#field_ident)?.clone() })
    } else if single_generic_arg(ty, "Vec").is_some() {
        Ok(quote! {
            raw.#field_ident
                .into_iter()
                .map(|target_id| env.target(&target_id).map(Clone::clone))
                .collect::<::switchyard_core::Result<Vec<_>>>()?
        })
    } else if single_generic_arg(ty, "Option").is_some() {
        Ok(quote! {
            match raw.#field_ident {
                Some(target_id) => Some(env.target(&target_id)?.clone()),
                None => None,
            }
        })
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "#[profile_target] supports LlmTarget, Vec<LlmTarget>, and Option<LlmTarget>",
        ))
    }
}

fn is_type_ident(ty: &Type, ident: &str) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path.path.segments.last().is_some_and(|segment| {
        segment.ident == ident && matches!(segment.arguments, PathArguments::None)
    })
}

fn single_generic_arg<'a>(ty: &'a Type, outer: &str) -> Option<&'a Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != outer {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut args = args.args.iter();
    let Some(GenericArgument::Type(inner)) = args.next() else {
        return None;
    };
    if args.next().is_some() {
        return None;
    }
    Some(inner)
}
