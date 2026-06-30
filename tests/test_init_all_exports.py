# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every name in switchyard.__all__ must be accessible from the top-level package.

These tests fail when symbols are listed in __all__ but are not handled by the
module-level __getattr__ lazy-loader, causing AttributeError / ImportError.
"""

from __future__ import annotations

import switchyard


def test_all_symbols_accessible() -> None:
    """All __all__ exports must be gettable without AttributeError."""
    missing = []
    for name in switchyard.__all__:
        try:
            getattr(switchyard, name)
        except AttributeError:
            missing.append(name)
    assert missing == [], f"Symbols in __all__ not importable from switchyard: {missing}"


def test_intake_payload_builder_importable() -> None:
    from switchyard import IntakePayloadBuilder  # noqa: F401


def test_intake_request_processor_importable() -> None:
    from switchyard import IntakeRequestProcessor  # noqa: F401


def test_intake_response_processor_importable() -> None:
    from switchyard import IntakeResponseProcessor  # noqa: F401


def test_intake_sink_config_importable() -> None:
    from switchyard import IntakeSinkConfig  # noqa: F401


def test_random_routing_config_importable() -> None:
    from switchyard import RandomRoutingConfig  # noqa: F401
