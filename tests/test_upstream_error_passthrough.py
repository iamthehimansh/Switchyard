# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate that upstream HTTP errors keep stable provider details.

Before this fix, a 401 from the upstream LLM (typically a bad API key
or expired credential) became a generic 500 at the client because the
compatibility executor wrapped the Python ``openai.APIStatusError`` in a
``SwitchyardError::Backend(error.to_string())``, which surfaced as a
plain Python ``RuntimeError`` and FastAPI defaulted it to 500.

Python backends stash upstream status/body on ``ProxyContext.metadata``.
Rust backends surface a typed upstream exception with ``status_code`` and
``body`` attributes. The endpoints recover either signal, preserve the
HTTP status plus stable provider fields, and return the normalized
Switchyard error envelope instead of FastAPI's default plain-text 500.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from switchyard.cli.route_bundle import build_route_bundle_table
from switchyard.lib.backends.health_poller import HealthPoller
from switchyard.lib.backends.latency_service_llm_backend import (
    LatencyServiceLLMBackend,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.endpoints.upstream_error import (
    internal_chain_error_response,
    upstream_response_from_ctx,
)
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.lib.proxy_context import (
    CTX_UPSTREAM_HTTP_BODY,
    CTX_UPSTREAM_HTTP_STATUS,
    ProxyContext,
)
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest
from tests._chain_test_helpers import _OpenAICompatStub

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _api_status_error(status: int, body: object = None) -> openai.APIStatusError:
    """Build an ``openai.APIStatusError`` with a realistic response object."""
    request = httpx.Request("POST", "http://upstream.test/v1/chat/completions")
    response = httpx.Response(
        status_code=status,
        json=body if body is not None else {"error": {"message": f"upstream {status}"}},
        request=request,
    )
    return openai.APIStatusError(
        message=f"upstream returned {status}",
        response=response,
        body=body,
    )


def _latency_service_switchyard(
    config: LatencyServiceBackendConfig,
) -> ProfileSwitchyard:
    """Build the latency-service profile-backed serving adapter."""
    return ProfileSwitchyard(
        LatencyServiceProfileConfig.from_config(config)
        .build()
        .with_runtime_components(enable_stats=config.enable_stats)
    )


def _make_backend() -> LatencyServiceLLMBackend:
    config = LatencyServiceBackendConfig(
        latency_service_url="http://latency-service.test:8080",
        endpoints=[
            LatencyServiceEndpoint(
                model="model-A",
                api_key="bad-key",
                base_url="http://llm.test/v1",
            ),
        ],
    )
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"):
            return LatencyServiceLLMBackend(config)


def _openai_request() -> ChatRequest:
    return ChatRequest.openai_chat({
        "model": "model-A",
        "messages": [{"role": "user", "content": "hi"}],
    })


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestUpstreamResponseFromCtx:
    def test_returns_none_when_no_status_recorded(self) -> None:
        ctx = ProxyContext()
        assert upstream_response_from_ctx(ctx) is None

    def test_wraps_string_body_in_error_envelope(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 401
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = "Unauthorized"

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 401

    def test_normalizes_dict_body_into_switchyard_envelope(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 429
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
            "error": {"message": "rate limited", "type": "rate_limit"},
        }

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 429
        body = json.loads(response.body)
        assert body == {
            "error": {
                "message": "rate limited",
                "type": "rate_limit",
                "code": "rate_limit",
            }
        }

    def test_preserves_provider_error_param_when_present(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 400
        ctx.metadata[CTX_UPSTREAM_HTTP_BODY] = {
            "error": {
                "message": "bad input",
                "type": "invalid_request_error",
                "code": "invalid_value",
                "param": "messages.0.role",
            },
        }

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 400
        assert json.loads(response.body) == {
            "error": {
                "message": "bad input",
                "type": "invalid_request_error",
                "code": "invalid_value",
                "param": "messages.0.role",
            }
        }

    def test_synthesizes_envelope_when_body_missing(self) -> None:
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = 503

        response = upstream_response_from_ctx(ctx)

        assert response is not None
        assert response.status_code == 503
        body = json.loads(response.body)
        assert body == {
            "error": {
                "message": "upstream returned HTTP 503",
                "type": "upstream_error",
                "code": "upstream_error",
            }
        }

    def test_ignores_non_int_status(self) -> None:
        """Defensive: a stray non-int value must not crash the error path."""
        ctx = ProxyContext()
        ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] = "401"  # wrong type — ignore

        assert upstream_response_from_ctx(ctx) is None

    def test_internal_chain_error_uses_openai_error_envelope(self) -> None:
        response = internal_chain_error_response(RuntimeError("connection refused"), "openai")

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = json.loads(response.body)
        assert "connection refused" in body["error"]["message"]
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"

    def test_internal_chain_error_uses_same_envelope_for_anthropic_inbound(self) -> None:
        response = internal_chain_error_response(RuntimeError("connection refused"), "anthropic")

        assert response.status_code == 500
        assert response.headers["content-type"].startswith("application/json")
        body = json.loads(response.body)
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["code"] == "internal_chain_error"
        assert "connection refused" in body["error"]["message"]

    def test_internal_chain_error_truncates_long_repr(self) -> None:
        long_msg = "x" * 500
        response = internal_chain_error_response(RuntimeError(long_msg), "openai")
        body = json.loads(response.body)
        assert len(body["error"]["message"]) <= 200


# ---------------------------------------------------------------------------
# Backend records status on ctx before raising
# ---------------------------------------------------------------------------


class TestLatencyServiceBackendStashesStatus:
    async def test_401_recorded_on_ctx_before_raise(self) -> None:
        backend = _make_backend()
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(401, {"error": {"message": "bad key"}}),
        )

        ctx = ProxyContext()
        with pytest.raises(openai.APIStatusError):
            await backend.call(ctx, _openai_request())

        assert ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] == 401
        body = ctx.metadata[CTX_UPSTREAM_HTTP_BODY]
        assert isinstance(body, dict)
        assert body["error"]["message"] == "bad key"

    async def test_429_recorded_on_ctx_before_raise(self) -> None:
        backend = _make_backend()
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=_api_status_error(429, {"error": {"message": "slow down"}}),
        )

        ctx = ProxyContext()
        with pytest.raises(openai.APIStatusError):
            await backend.call(ctx, _openai_request())

        assert ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] == 429

    async def test_non_http_error_leaves_ctx_clean(self) -> None:
        """Network errors etc shouldn't trip the upstream-status passthrough."""
        backend = _make_backend()
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=RuntimeError("connection refused"),
        )

        ctx = ProxyContext()
        with pytest.raises(RuntimeError, match="connection refused"):
            await backend.call(ctx, _openai_request())

        assert CTX_UPSTREAM_HTTP_STATUS not in ctx.metadata
        assert CTX_UPSTREAM_HTTP_BODY not in ctx.metadata


# ---------------------------------------------------------------------------
# End-to-end: HTTP request through the recipe → upstream 401 → client 401
# ---------------------------------------------------------------------------


@pytest.fixture
async def latency_service_app() -> AsyncIterator[tuple[FastAPI, LatencyServiceLLMBackend]]:
    """Build a latency-service-backed FastAPI app sharing one backend instance.

    Returns ``(app, backend)`` so each test can patch ``backend._clients[...].acompletion``
    to simulate a specific upstream response.
    """
    config = LatencyServiceBackendConfig(
        latency_service_url="http://latency-service.test:8080",
        endpoints=[
            LatencyServiceEndpoint(
                model="model-A",
                api_key="bad-key",
                base_url="http://llm.test/v1",
            ),
        ],
    )
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            switchyard = _latency_service_switchyard(config)
            app = build_switchyard_app(switchyard)
            # Reach into the chain to find the LatencyServiceLLMBackend
            # instance so tests can stub the upstream call.
            backend = _find_latency_backend(switchyard)
            yield app, backend


def _find_latency_backend(switchyard: object) -> LatencyServiceLLMBackend:
    iter_components = getattr(switchyard, "iter_components", None)
    assert iter_components is not None
    for component in iter_components():
        # The latency-service backend may be wrapped (it isn't today, but
        # be tolerant) — match by isinstance, with attribute-walk fallback.
        if isinstance(component, LatencyServiceLLMBackend):
            return component
        inner = getattr(component, "_inner", None) or getattr(component, "inner", None)
        if isinstance(inner, LatencyServiceLLMBackend):
            return inner
    raise AssertionError("LatencyServiceLLMBackend not found in chain")


def test_upstream_401_passes_through_as_401(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    """A 401 from upstream must become a 401 to the client, not a 500."""
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=_api_status_error(
            401, {"error": {"message": "bad key", "type": "invalid_api_key"}},
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-A", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 401, (
        f"expected upstream 401 to pass through; got {response.status_code}, "
        f"body={response.text!r}"
    )
    body = response.json()
    assert body["error"]["message"] == "bad key"
    assert body["error"]["type"] == "invalid_api_key"
    assert body["error"]["code"] == "invalid_api_key"


def test_upstream_429_passes_through_as_429(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=_api_status_error(
            429, {"error": {"message": "rate limit", "type": "rate_limit_exceeded"}},
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-A", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "message": "rate limit",
            "type": "rate_limit_exceeded",
            "code": "rate_limit_exceeded",
        }
    }


def test_non_http_error_returns_wrapped_500(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    """Errors without an upstream HTTP status still return a JSON error envelope."""
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=RuntimeError("connection refused"),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-A", "messages": [{"role": "user", "content": "ping"}]},
        )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["type"] == "internal_error"
    assert body["error"]["code"] == "internal_chain_error"
    assert "connection refused" in body["error"]["message"]


def test_post_dispatch_exception_returns_json_500(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    """Exception raised after dispatch (e.g. during result serialization) must not
    fall through to FastAPI's plain-text 500 handler."""
    app, _ = latency_service_app

    class _UnserializableResult:
        def model_dump(self) -> None:
            raise RuntimeError("serialization exploded")

    with patch(
        "switchyard.lib.endpoints.openai_chat_endpoint.dispatch_chat_request",
        new_callable=AsyncMock,
        return_value=_UnserializableResult(),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "model-A", "messages": [{"role": "user", "content": "ping"}]},
            )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["type"] == "internal_error"
    assert body["error"]["code"] == "internal_chain_error"
    assert "serialization exploded" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Anthropic and Responses endpoints share the same helper
# ---------------------------------------------------------------------------


def test_anthropic_endpoint_passes_through_upstream_401(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=_api_status_error(401, {"error": {"message": "bad key"}}),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "model-A",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "upstream_error",
            "code": "upstream_error",
        }
    }


def test_responses_endpoint_passes_through_upstream_401(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock(
        side_effect=_api_status_error(401, {"error": {"message": "bad key"}}),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "model-A", "input": "ping"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "upstream_error",
            "code": "upstream_error",
        }
    }


def test_rust_openai_route_upstream_401_returns_structured_openai_error() -> None:
    """Rust OpenAI-native backend errors must not fall through to FastAPI's 500."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "model",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "bad-key",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


def test_rust_openai_route_upstream_401_returns_same_error_shape_for_anthropic_inbound() -> None:
    """Anthropic inbound clients should receive the same HTTP error envelope."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "model",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "bad-key",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


def test_rust_openai_route_upstream_401_returns_same_error_shape_for_responses_inbound() -> None:
    """Responses inbound clients should receive the same HTTP error envelope."""
    with _OpenAICompatStub() as upstream:
        upstream.respond_json(
            {"error": {"message": "bad key", "type": "invalid_api_key"}},
            status=401,
        )
        table = build_route_bundle_table({
            "defaults": {
                "api_key": "bad-key",
                "base_url": upstream.base_url,
                "format": "openai",
            },
            "routes": {
                "bad-key": {
                    "type": "model",
                    "target": "nvidia/nvidia/nemotron-nano-9b-v2",
                }
            },
        })

        with TestClient(build_switchyard_app(table), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "bad-key",
                    "input": "ping",
                },
            )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        },
    }


# ---------------------------------------------------------------------------
# Invalid request payloads
# ---------------------------------------------------------------------------
# A transparent router must reject the same payloads the upstream provider
# would. An unsupported message role (e.g. "api") was previously coerced to
# "user" and returned 200; it now surfaces as a provider-compatible 400 via the
# same ctx-stash → endpoint-passthrough mechanism used for upstream HTTP errors.


class TestInvalidRoleRejection:
    async def test_invalid_role_stashes_400_without_upstream_call(self) -> None:
        """An unsupported role is rejected at translation, before any upstream call."""
        backend = _make_backend()
        backend._clients["model-A"].acompletion = AsyncMock()

        ctx = ProxyContext()
        request = ChatRequest.openai_responses({
            "model": "model-A",
            "input": [{"type": "message", "role": "api", "content": "hi"}],
        })
        with pytest.raises(ValueError):
            await backend.call(ctx, request)

        assert ctx.metadata[CTX_UPSTREAM_HTTP_STATUS] == 400
        body = ctx.metadata[CTX_UPSTREAM_HTTP_BODY]
        assert isinstance(body, dict)
        assert body["error"]["code"] == "invalid_value"
        # The upstream must never be called for an invalid payload.
        backend._clients["model-A"].acompletion.assert_not_awaited()

    async def test_valid_role_passes_translation_and_reaches_upstream(self) -> None:
        """A valid role must not be over-rejected — translation succeeds and the
        request proceeds to the upstream call (no invalid-value 400 stashed)."""
        backend = _make_backend()
        backend._clients["model-A"].acompletion = AsyncMock(
            side_effect=RuntimeError("reached upstream"),
        )

        ctx = ProxyContext()
        request = ChatRequest.openai_responses({
            "model": "model-A",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        })
        with pytest.raises(RuntimeError, match="reached upstream"):
            await backend.call(ctx, request)

        assert CTX_UPSTREAM_HTTP_STATUS not in ctx.metadata


def test_responses_invalid_role_returns_400(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    """role:"api" on the Responses endpoint must return 400, not a coerced 200/500."""
    app, backend = latency_service_app
    # No upstream stub: the request must be rejected before any upstream call.
    backend._clients["model-A"].acompletion = AsyncMock()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "model-A",
                "input": [{"type": "message", "role": "api", "content": "ping"}],
            },
        )

    assert response.status_code == 400, (
        f"expected invalid role to be rejected with 400; got {response.status_code}, "
        f"body={response.text!r}"
    )
    body = response.json()
    assert body["error"]["code"] == "invalid_value"
    backend._clients["model-A"].acompletion.assert_not_awaited()


def test_anthropic_invalid_role_returns_400(
    latency_service_app: tuple[FastAPI, LatencyServiceLLMBackend],
) -> None:
    """role:"api" on the Anthropic endpoint is rejected with a 400.

    The Anthropic inbound format must be translated to OpenAI Chat for the
    latency backend, so the request is decoded and the unsupported role is
    rejected before any upstream call. (A native OpenAI-Chat request is a
    passthrough for this backend, so its contract is enforced upstream.)
    """
    app, backend = latency_service_app
    backend._clients["model-A"].acompletion = AsyncMock()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "model-A",
                "max_tokens": 16,
                "messages": [{"role": "api", "content": "ping"}],
            },
        )

    assert response.status_code == 400, (
        f"expected invalid role to be rejected with 400; got {response.status_code}, "
        f"body={response.text!r}"
    )
    backend._clients["model-A"].acompletion.assert_not_awaited()
