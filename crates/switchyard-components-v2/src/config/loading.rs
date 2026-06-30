// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Filesystem loading and format detection for profile configs.

use std::fs;
use std::path::Path;

use switchyard_core::{Result, SwitchyardError};

use super::parsing::ProfileConfigDocument;

/// File format accepted by the v2 profile config loader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileConfigFormat {
    /// JSON configuration.
    Json,
    /// TOML configuration.
    Toml,
    /// YAML configuration.
    Yaml,
}

impl ProfileConfigFormat {
    /// Infers the config format from a path extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|extension| extension.to_str())
            .ok_or_else(|| {
                SwitchyardError::InvalidConfig(format!(
                    "profile config path {} has no file extension",
                    path.as_ref().display()
                ))
            })?;
        match extension {
            "json" => Ok(Self::Json),
            "toml" => Ok(Self::Toml),
            "yaml" | "yml" => Ok(Self::Yaml),
            other => Err(SwitchyardError::InvalidConfig(format!(
                "unsupported profile config extension `{other}`"
            ))),
        }
    }
}

impl ProfileConfigDocument {
    /// Reads a profile config path and infers the format from its extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let input = fs::read_to_string(path).map_err(|error| {
            SwitchyardError::InvalidConfig(format!(
                "failed to read profile config {}: {error}",
                path.display()
            ))
        })?;
        Self::from_str(&input, ProfileConfigFormat::from_path(path)?)
    }
}

/// Reads and parses a profile config path, inferring the format from extension.
pub fn parse_profile_config_path(path: impl AsRef<Path>) -> Result<ProfileConfigDocument> {
    ProfileConfigDocument::from_path(path)
}
