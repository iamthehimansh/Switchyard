# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust components-v2 profile server bindings."""

from switchyard_rust.core import _load_native


def run_profile_server(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 4000,
    backlog: int = 65_535,
    dry_run: bool = False,
) -> None:
    """Run the Rust components-v2 profile server from an installed package."""
    _load_native().run_profile_server(config_path, host, port, backlog, dry_run)


__all__ = ["run_profile_server"]
