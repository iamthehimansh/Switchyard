// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public API tests for the components-v2 passthrough profile config.

use serde_json::json;
use switchyard_components_v2::{PassthroughProfileConfig, ProfileConfig};
use switchyard_core::{BackendFormat, LlmTarget, LlmTargetId, ModelId, Result, SwitchyardError};

fn target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::OpenAi;
    Ok(target)
}

fn anthropic_target(id: &str, model: &str) -> Result<LlmTarget> {
    let mut target = LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?);
    target.format = BackendFormat::Anthropic;
    Ok(target)
}

fn auto_target(id: &str, model: &str) -> Result<LlmTarget> {
    Ok(LlmTarget::new(LlmTargetId::new(id)?, ModelId::new(model)?))
}

fn config(target: LlmTarget) -> PassthroughProfileConfig {
    PassthroughProfileConfig { target }
}

#[test]
fn profile_config_build_uses_existing_native_backend_stack() -> Result<()> {
    let config = config(target("direct", "provider/model")?);

    let _profile = config.build()?;
    Ok(())
}

#[test]
fn profile_config_build_supports_anthropic_native_backend_stack() -> Result<()> {
    let config = config(anthropic_target("direct", "anthropic/model")?);

    let _profile = config.build()?;
    Ok(())
}

#[test]
fn profile_config_build_rejects_unresolved_backend_format() -> Result<()> {
    let config = config(auto_target("direct", "provider/model")?);

    match config.build() {
        Err(SwitchyardError::InvalidConfig(message)) => {
            assert!(message.contains("must have a resolved backend format"));
        }
        Ok(_) => {
            return Err(SwitchyardError::Other(
                "unresolved passthrough backend format should fail".into(),
            ));
        }
        Err(other) => {
            return Err(SwitchyardError::Other(format!(
                "expected InvalidConfig, got {other}"
            )));
        }
    }
    Ok(())
}

#[test]
fn profile_config_macro_adds_type_metadata_and_strict_serde() -> Result<()> {
    let config = config(target("direct", "provider/model")?);

    assert_eq!(PassthroughProfileConfig::PROFILE_TYPE, "passthrough");
    assert_eq!(config.profile_type(), "passthrough");

    let old_stats_toggle = json!({
        "target": config.target,
        "stats": false,
    });
    let error = serde_json::from_value::<PassthroughProfileConfig>(old_stats_toggle)
        .err()
        .ok_or_else(|| SwitchyardError::Other("unknown profile field should fail".into()))?;
    assert!(error.to_string().contains("unknown field"));
    Ok(())
}
