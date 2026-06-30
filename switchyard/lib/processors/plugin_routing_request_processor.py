# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request processor that asks an external routing plugin which tier to use.

The processor owns the :class:`PluginClient` lifecycle through the
``startup`` / ``shutdown`` hooks on the component. On each
inbound request it builds a normalized
:class:`RoutingRequestSummary`, asks the plugin for a tier label, and
stamps the selected target/model on ``ProxyContext`` so the downstream
``MultiLlmBackend`` can dispatch.

When the plugin fails — timeout, crash, malformed response, or an
explicit error envelope — the processor either falls back to the
configured ``fallback_tier`` (if the operator opted in) or re-raises so
the chain returns a 5xx upstream. Failures are logged at ``warning``
level so operators can spot misbehaving plugins without grepping debug.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from switchyard.lib.plugin import (
    PluginClient,
    PluginCrashError,
    PluginRoutingError,
    PluginTimeoutError,
    RouteDecision,
    RouteError,
    RouteRequest,
    RoutingRequestSummary,
)
from switchyard_rust.components import set_stats_route_label

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

# Compatibility metadata key; Rust ``MultiLlmBackend`` dispatches from
# ``ProxyContext.selected_target``.
CTX_OSS_ROUTER_TIER = "_oss_router_tier"


class PluginRoutingRequestProcessor:
    """Calls an external plugin to choose a tier per request.

    Args:
        plugin_command: Argv (or shell string) for the plugin executable.
        tier_models: Mapping of tier label → model id. The labels are
            sent to the plugin via the handshake's ``available_tiers``;
            the model id is what gets written into ``request.body["model"]``
            after the plugin picks. Insertion order matters: the first
            entry is the defensive default.
        fallback_tier: Optional tier label to dispatch to when the plugin
            returns an error or fails to respond. ``None`` re-raises so
            the chain returns 5xx.
        request_timeout_s / handshake_timeout_s: Plumbed through to the
            :class:`PluginClient`.
        env: Optional extra env vars merged on top of the proxy's
            environment when spawning the plugin.
        expose_metadata_keys: Whitelist of ``ctx.metadata`` keys to forward
            to the plugin (off by default — plugins see only the request
            summary unless the operator opts specific keys in).
    """

    def __init__(
        self,
        *,
        plugin_command: str | Sequence[str],
        tier_models: Mapping[str, str],
        tier_target_ids: Mapping[str, str] | None = None,
        fallback_tier: str | None = None,
        request_timeout_s: float = 5.0,
        handshake_timeout_s: float = 10.0,
        env: dict[str, str] | None = None,
        expose_metadata_keys: Sequence[str] = (),
    ) -> None:
        if not tier_models:
            raise ValueError("PluginRoutingRequestProcessor requires at least one tier")
        if fallback_tier is not None and fallback_tier not in tier_models:
            raise ValueError(
                f"fallback_tier {fallback_tier!r} is not in tier_models "
                f"{sorted(tier_models)}",
            )

        self._plugin_command = plugin_command
        self._tier_models: dict[str, str] = dict(tier_models)
        self._tier_target_ids: dict[str, str] = dict(tier_target_ids or tier_models)
        self._tier_labels: tuple[str, ...] = tuple(self._tier_models)
        self._fallback_tier = fallback_tier
        self._request_timeout_s = request_timeout_s
        self._handshake_timeout_s = handshake_timeout_s
        self._env = env
        self._expose_metadata_keys = tuple(expose_metadata_keys)

        self._client: PluginClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        if self._client is not None:
            return
        self._client = await PluginClient.start(
            command=self._plugin_command,
            available_tiers=self._tier_labels,
            request_timeout_s=self._request_timeout_s,
            handshake_timeout_s=self._handshake_timeout_s,
            env=self._env,
        )

    async def shutdown(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.shutdown()
        finally:
            self._client = None

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    async def process(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
    ) -> ChatRequest:
        client = self._client
        if client is None:
            raise RuntimeError(
                "PluginRoutingRequestProcessor used without startup(); "
                "callers must await startup() before the first request",
            )

        request_id = ctx.metadata.get("request_id")
        if not isinstance(request_id, str):
            request_id = str(uuid.uuid4())

        route_request = RouteRequest(
            request_id=request_id,
            available_tiers=self._tier_labels,
            summary=_summarize_request(request),
            metadata=self._exposed_metadata(ctx),
        )

        try:
            outcome = await client.route(route_request)
        except (PluginTimeoutError, PluginCrashError, PluginRoutingError) as exc:
            return self._handle_failure(ctx, request, exc=exc)

        if isinstance(outcome, RouteError):
            return self._handle_route_error(ctx, request, outcome)
        return self._apply_decision(ctx, request, outcome)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_decision(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        decision: RouteDecision,
    ) -> ChatRequest:
        # _classify_route_envelope already validated the tier; this is just
        # a defensive guard in case a future client path bypasses that.
        if decision.tier not in self._tier_models:
            return self._handle_failure(
                ctx,
                request,
                exc=PluginRoutingError(
                    f"Plugin returned unknown tier {decision.tier!r}",
                ),
            )
        self._stamp(ctx, decision.tier, plugin_metadata=decision.metadata)
        request.set_model(self._tier_models[decision.tier])
        return request

    def _handle_route_error(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        outcome: RouteError,
    ) -> ChatRequest:
        # Prefer the plugin-suggested fallback when it advertised one and
        # we know the label; otherwise fall through to the operator-configured
        # fallback. Operator config is the source of truth — plugins can hint
        # but cannot override.
        candidate = outcome.fallback_tier if outcome.fallback_tier in self._tier_models else None
        chosen = candidate or self._fallback_tier
        log.warning(
            "PluginRoutingRequestProcessor: plugin returned error code=%d msg=%r "
            "fallback=%r resolved=%r",
            outcome.code,
            outcome.message,
            outcome.fallback_tier,
            chosen,
        )
        if chosen is None:
            raise PluginRoutingError(
                f"Plugin returned error: code={outcome.code} message={outcome.message!r}",
            )
        self._stamp(
            ctx,
            chosen,
            plugin_metadata={"fallback_reason": outcome.message, "fallback_code": outcome.code},
        )
        request.set_model(self._tier_models[chosen])
        return request

    def _handle_failure(
        self,
        ctx: ProxyContext,
        request: ChatRequest,
        *,
        exc: Exception,
    ) -> ChatRequest:
        if self._fallback_tier is None:
            raise exc
        log.warning(
            "PluginRoutingRequestProcessor: plugin failure %s: %s; "
            "dispatching to fallback tier %r",
            type(exc).__name__,
            exc,
            self._fallback_tier,
        )
        self._stamp(
            ctx,
            self._fallback_tier,
            plugin_metadata={"fallback_reason": str(exc), "fallback_exception": type(exc).__name__},
        )
        request.set_model(self._tier_models[self._fallback_tier])
        return request

    def _stamp(
        self,
        ctx: ProxyContext,
        tier: str,
        *,
        plugin_metadata: Mapping[str, Any],
    ) -> None:
        ctx.metadata[CTX_OSS_ROUTER_TIER] = tier
        ctx.metadata["_oss_router_model"] = self._tier_models[tier]
        set_stats_route_label(ctx, tier)
        ctx.selected_target = self._tier_target_ids[tier]
        ctx.selected_model = self._tier_models[tier]
        if plugin_metadata:
            ctx.metadata["_oss_router_plugin_metadata"] = dict(plugin_metadata)

    def _exposed_metadata(self, ctx: ProxyContext) -> dict[str, Any]:
        if not self._expose_metadata_keys:
            return {}
        return {
            key: ctx.metadata[key]
            for key in self._expose_metadata_keys
            if key in ctx.metadata
        }


# ---------------------------------------------------------------------------
# Request summarizer
# ---------------------------------------------------------------------------


def _summarize_request(request: ChatRequest) -> RoutingRequestSummary:
    """Build the minimal opt-out-by-default summary we send the plugin.

    Defensive against malformed bodies — plugins should never hard-fail
    Switchyard, so each field falls back to a safe default rather than
    raising.
    """
    body: dict[str, Any] = getattr(request, "body", {}) or {}
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    tools = body.get("tools") or []
    tool_names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                # OpenAI Chat: {"function": {"name": ...}}; Anthropic: {"name": ...}
                fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
                name = (fn or tool).get("name") if isinstance(fn or tool, dict) else None
                if isinstance(name, str):
                    tool_names.append(name)

    return RoutingRequestSummary(
        input_token_estimate=_rough_token_estimate(messages),
        message_count=len(messages),
        has_tool_use=bool(tool_names),
        tool_names=tuple(tool_names),
    )


def _rough_token_estimate(messages: list[Any]) -> int:
    """4-chars-per-token heuristic — fine for routing decisions, not billing."""
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
    return max(1, total_chars // 4)
