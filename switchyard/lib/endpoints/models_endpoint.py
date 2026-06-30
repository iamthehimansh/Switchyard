# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint serving ``GET /v1/models`` for local model discovery."""

import logging

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from switchyard.lib.endpoints.base import Endpoint as NemoSwitchyardEndpoint
from switchyard.lib.endpoints.dispatch import (
    model_entries,
    model_listing_default,
    model_listing_warnings,
)
from switchyard.lib.model_listing import model_list_payload

log = logging.getLogger(__name__)


class ModelsEndpoint(NemoSwitchyardEndpoint):
    """Expose registered model ids for clients with model discovery."""

    def register(self, app: FastAPI) -> None:
        """Attach ``GET /v1/models`` onto *app*."""
        router = APIRouter()

        @router.get("/v1/models", response_model=None)
        async def models(request: Request) -> JSONResponse:
            obj = request.app.state.switchyard
            entries = model_entries(obj)
            log.debug("GET /v1/models returned %d model(s)", len(entries))
            return JSONResponse(
                content=model_list_payload(
                    entries,
                    default_model=model_listing_default(obj),
                    warnings=model_listing_warnings(obj),
                )
            )

        app.include_router(router, tags=["Model Discovery"])
