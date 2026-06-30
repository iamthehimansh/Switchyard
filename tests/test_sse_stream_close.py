# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming teardown: the upstream stream is closed on every exit path.

Regression coverage for the connection-pool leak where an interrupted SSE
response (client disconnect) never closed the upstream SDK ``AsyncStream``, so
its httpx connection was never returned to the pool. The fix closes the stream
from the SSE helpers' ``finally`` and gives ``ChatResponseStream`` an ``aclose``
that releases the original Python source.
"""

from __future__ import annotations

from typing import Any

from switchyard.lib.chat_response.openai_chat import ResponseStream
from switchyard.lib.endpoints.sse_helpers import (
    _aclose_stream,
    iter_anthropic_sse,
    iter_chat_completion_sse,
    iter_preframed_sse,
)


class _FakeAsyncStream:
    """Mimics the OpenAI/Anthropic SDK ``AsyncStream``.

    Async-iterable and closable via an async ``close()`` (the SDK shape — not
    the async-generator ``aclose``). Records whether ``close()`` ran so tests
    can assert the upstream connection would have been released.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.closed = False

    def __aiter__(self) -> _FakeAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# ChatResponseStream.aclose (Rust binding)
# ---------------------------------------------------------------------------


async def test_chat_response_stream_aclose_closes_source() -> None:
    """``aclose`` releases the original Python source even before iteration."""
    fake = _FakeAsyncStream([{"id": "1"}])
    stream = ResponseStream(fake)
    await stream.aclose()
    assert fake.closed


async def test_chat_response_stream_aclose_is_idempotent() -> None:
    fake = _FakeAsyncStream([{"id": "1"}])
    stream = ResponseStream(fake)
    await stream.aclose()
    await stream.aclose()
    assert fake.closed


# ---------------------------------------------------------------------------
# SSE helpers close their stream on early termination and normal completion
# ---------------------------------------------------------------------------


async def test_chat_sse_closes_stream_on_client_disconnect() -> None:
    """A client disconnect makes the server ``aclose()`` the SSE generator;
    the helper's ``finally`` must then close the upstream stream."""
    fake = _FakeAsyncStream([{"id": "1"}, {"id": "2"}, {"id": "3"}])
    generator = iter_chat_completion_sse(fake)
    first = await generator.__anext__()
    assert first.startswith("data:")
    await generator.aclose()  # simulate disconnect mid-stream
    assert fake.closed


async def test_chat_sse_closes_stream_on_normal_completion() -> None:
    fake = _FakeAsyncStream([{"id": "1"}])
    frames = [frame async for frame in iter_chat_completion_sse(fake)]
    assert frames[-1] == "data: [DONE]\n\n"
    assert fake.closed


async def test_chat_sse_closes_rust_stream_source_on_disconnect() -> None:
    """End-to-end: SSE over a ``ChatResponseStream`` → disconnect → the wrapped
    SDK ``AsyncStream`` (and its connection) is released. This is the exact leak
    scenario from the OOM incident."""
    fake = _FakeAsyncStream([{"id": "1"}, {"id": "2"}])
    generator = iter_chat_completion_sse(ResponseStream(fake))
    await generator.__anext__()
    await generator.aclose()
    assert fake.closed


async def test_anthropic_sse_closes_stream_on_disconnect() -> None:
    fake = _FakeAsyncStream([{"type": "message_start"}, {"type": "content_block_delta"}])
    generator = iter_anthropic_sse(fake)
    await generator.__anext__()
    await generator.aclose()
    assert fake.closed


async def test_preframed_sse_closes_stream_on_disconnect() -> None:
    fake = _FakeAsyncStream(["event: ping\ndata: {}\n\n", "event: ping\ndata: {}\n\n"])
    generator = iter_preframed_sse(fake)
    await generator.__anext__()
    await generator.aclose()
    assert fake.closed


async def test_preframed_sse_frames_responses_mapping_events() -> None:
    fake = _FakeAsyncStream([
        {
            "type": "response.created",
            "response": {"id": "resp-test"},
        }
    ])
    frames = [frame async for frame in iter_preframed_sse(fake)]
    assert frames == [
        'event: response.created\ndata: {"type": "response.created", "response": {"id": "resp-test"}}\n\n'
    ]
    assert fake.closed


# ---------------------------------------------------------------------------
# _aclose_stream contract
# ---------------------------------------------------------------------------


async def test_aclose_stream_prefers_aclose() -> None:
    class WithAclose:
        def __init__(self) -> None:
            self.aclosed = False

        async def aclose(self) -> None:
            self.aclosed = True

    obj = WithAclose()
    await _aclose_stream(obj)
    assert obj.aclosed


async def test_aclose_stream_tolerates_missing_closer() -> None:
    # An object with neither ``aclose`` nor ``close`` must not raise.
    await _aclose_stream(object())
