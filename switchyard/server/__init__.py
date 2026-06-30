# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP serving primitives.

Exposes:

* ``server_util`` — CLI helpers.
* ``switchyard_app`` — FastAPI app factory for a ``Switchyard`` chain.
"""

from switchyard.server.server_util import (
    DEFAULT_SECRETS_FILE,
    REPO_ROOT,
    add_common_args,
    add_transport_args,
    build_and_serve,
    ensure_openai_api_key_env,
    load_secrets,
    resolve_config_with_secrets,
    resolve_credentials_from_env,
)

__all__ = [
    "DEFAULT_SECRETS_FILE",
    "REPO_ROOT",
    "add_common_args",
    "add_transport_args",
    "build_and_serve",
    "ensure_openai_api_key_env",
    "load_secrets",
    "resolve_config_with_secrets",
    "resolve_credentials_from_env",
]
