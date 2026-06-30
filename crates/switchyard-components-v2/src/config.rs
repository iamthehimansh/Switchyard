// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Profile config entry point split by lifecycle stage.
//!
//! Parsing produces a file-facing `ProfileConfigDocument`. Resolving turns that
//! document into an opaque `ProfileConfigPlan` with runtime-ready targets and
//! validated profile bodies.

mod loading;
mod parsing;
mod resolving;

pub use loading::{parse_profile_config_path, ProfileConfigFormat};
pub use parsing::{
    parse_profile_config_str, parse_profile_config_str_with_env_lookup, ProfileConfigDocument,
};
pub use resolving::{ProfileBuildEnv, ProfileConfigDefinition};
pub use resolving::{ProfileConfig, ProfileConfigPlan};
