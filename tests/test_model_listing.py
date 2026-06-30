# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from switchyard.lib.model_listing import model_entry, model_list_payload


def test_model_entry_merges_metadata_without_overriding_wire_identity() -> None:
    entry = model_entry(
        "served-model",
        metadata={
            "id": "wrong",
            "object": "wrong",
            "type": "wrong",
            "display_name": "Served Model",
            "description": "Runtime route.",
            "owned_by": "nvidia",
            "capabilities": {
                "context_window": 128_000,
                "tool_calling": True,
            },
        },
    )

    assert entry["id"] == "served-model"
    assert entry["object"] == "model"
    assert entry["type"] == "model"
    assert entry["display_name"] == "Served Model"
    assert entry["owned_by"] == "nvidia"
    assert entry["description"] == "Runtime route."
    assert entry["capabilities"]["context_window"] == 128_000
    assert entry["capabilities"]["tool_calling"] is True
    assert entry["capabilities"]["streaming"] is True
    assert entry["capabilities"]["supported_inbound_formats"] == [
        "openai-chat-completions",
        "openai-responses",
        "anthropic-messages",
    ]


def test_model_list_payload_exposes_default_pool_and_warnings() -> None:
    payload = model_list_payload(
        [
            model_entry("first"),
            model_entry("second"),
        ],
        warnings=["catalog unavailable"],
    )

    assert payload["object"] == "list"
    assert payload["first_id"] == "first"
    assert payload["last_id"] == "second"
    assert payload["default_model"] == "first"
    assert payload["model_pool"] == ["first", "second"]
    assert payload["has_more"] is False
    assert payload["warnings"] == ["catalog unavailable"]


def test_model_list_payload_uses_advertised_default_when_present() -> None:
    payload = model_list_payload(
        [
            model_entry("strong/model"),
            model_entry("weak/model"),
            model_entry("switchyard-route"),
        ],
        default_model="switchyard-route",
    )

    assert payload["first_id"] == "strong/model"
    assert payload["default_model"] == "switchyard-route"


def test_model_list_payload_falls_back_when_advertised_default_is_missing() -> None:
    payload = model_list_payload(
        [model_entry("first")],
        default_model="missing",
    )

    assert payload["default_model"] == "first"


def test_model_entry_defaults_capability_fields_to_non_null_values() -> None:
    entry = model_entry("unknown/model")

    assert entry["capabilities"]["tool_calling"] is True
    assert entry["capabilities"]["context_window"] == 128_000
