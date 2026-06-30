# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inbound-format JSON shapes for :func:`context_exhausted_response`."""

from __future__ import annotations

import json

from switchyard.lib.endpoints.upstream_error import context_exhausted_response


def _body(exc_message: str, inbound: str) -> dict:
    exc = RuntimeError(exc_message)
    response = context_exhausted_response(exc, inbound=inbound)  # type: ignore[arg-type]
    assert response.status_code == 400
    return json.loads(response.body)


def test_anthropic_inbound_shape() -> None:
    body = _body("context pool exhausted", "anthropic")
    assert body == {
        "error": {
            "message": "context pool exhausted",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    }


def test_openai_inbound_shape() -> None:
    body = _body("context pool exhausted", "openai")
    assert body == {
        "error": {
            "message": "context pool exhausted",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    }


def test_openai_responses_inbound_shape() -> None:
    body = _body("context pool exhausted", "openai-responses")
    assert body == {
        "error": {
            "message": "context pool exhausted",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    }
