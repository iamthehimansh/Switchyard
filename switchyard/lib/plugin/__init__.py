# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External-process routing plugin contract.

See :mod:`switchyard.lib.plugin.plugin_protocol` for the wire schema and
:mod:`switchyard.lib.plugin.plugin_client` for the in-process client that
spawns the plugin and routes requests over its stdio channel.
"""

from switchyard.lib.plugin.plugin_client import (
    PluginClient,
    PluginCrashError,
    PluginHandshakeError,
    PluginRoutingError,
    PluginTimeoutError,
)
from switchyard.lib.plugin.plugin_protocol import (
    PROTOCOL_VERSION,
    HandshakeRequest,
    HandshakeResponse,
    RouteDecision,
    RouteError,
    RouteRequest,
    RouteResult,
    RoutingRequestSummary,
)

__all__ = [
    "PROTOCOL_VERSION",
    "HandshakeRequest",
    "HandshakeResponse",
    "PluginClient",
    "PluginCrashError",
    "PluginHandshakeError",
    "PluginRoutingError",
    "PluginTimeoutError",
    "RouteDecision",
    "RouteError",
    "RouteRequest",
    "RouteResult",
    "RoutingRequestSummary",
]
