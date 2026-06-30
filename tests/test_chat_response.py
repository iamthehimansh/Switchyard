# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ChatResponse type hierarchy and all stream wrappers."""

from collections.abc import AsyncIterator

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import RawContentBlockDeltaEvent
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.text_delta import TextDelta
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseTextDeltaEvent

from switchyard.lib.chat_response import AnthropicResponseStream, ResponsesApiStream, ResponseStream
from switchyard_rust.core import (
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_completion(*, model: str = "gpt-4o", content: str = "hello") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_chunk(*, index: int = 0, content: str = "hi") -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-test",
        object="chat.completion.chunk",
        created=1700000000,
        model="gpt-4o",
        choices=[
            ChunkChoice(
                index=index,
                delta=ChoiceDelta(content=content),
                finish_reason=None,
            )
        ],
    )


async def fake_stream(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


# --- Anthropic fixtures ---


def make_anthropic_message(
    *, model: str = "claude-sonnet-4-20250514", text: str = "hello"
) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        usage=AnthropicUsage(input_tokens=10, output_tokens=5),
    )


def make_anthropic_content_delta(*, text: str = "hi") -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=TextDelta(type="text_delta", text=text),
    )


async def fake_anthropic_stream(
    events: list[RawContentBlockDeltaEvent],
) -> AsyncIterator[RawContentBlockDeltaEvent]:
    for event in events:
        yield event


# --- OpenAI Responses API fixtures ---


def make_responses_api_response(
    *, model: str = "gpt-4o", text: str = "hello"
) -> OpenAIResponse:
    return OpenAIResponse(
        id="resp_test",
        created_at=1700000000,
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": "msg_test",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        tools=[],
        status="completed",
        parallel_tool_calls=True,
        text={"format": {"type": "text"}},
    )


def make_responses_text_delta(*, delta: str = "hi") -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent(
        type="response.output_text.delta",
        item_id="item_test",
        output_index=0,
        content_index=0,
        delta=delta,
        logprobs=[],
        sequence_number=0,
    )


async def fake_responses_stream(
    events: list[ResponseTextDeltaEvent],
) -> AsyncIterator[ResponseTextDeltaEvent]:
    for event in events:
        yield event


# ---------------------------------------------------------------------------
# ChatResponseType
# ---------------------------------------------------------------------------


class TestChatResponseType:
    def test_enum_values(self):
        assert ChatResponseType.OPENAI_COMPLETION.value == "openai_completion"
        assert ChatResponseType.OPENAI_STREAM.value == "openai_stream"
        assert (
            ChatResponseType.OPENAI_RESPONSES_COMPLETION.value
            == "openai_responses_completion"
        )
        assert ChatResponseType.OPENAI_RESPONSES_STREAM.value == "openai_responses_stream"
        assert ChatResponseType.ANTHROPIC_COMPLETION.value == "anthropic_completion"
        assert ChatResponseType.ANTHROPIC_STREAM.value == "anthropic_stream"

    def test_enum_members(self):
        assert [
            item.value
            for item in (
                ChatResponseType.OPENAI_COMPLETION,
                ChatResponseType.OPENAI_STREAM,
                ChatResponseType.OPENAI_RESPONSES_COMPLETION,
                ChatResponseType.OPENAI_RESPONSES_STREAM,
                ChatResponseType.ANTHROPIC_COMPLETION,
                ChatResponseType.ANTHROPIC_STREAM,
            )
        ] == [
            "openai_completion",
            "openai_stream",
            "openai_responses_completion",
            "openai_responses_stream",
            "anthropic_completion",
            "anthropic_stream",
        ]


# ---------------------------------------------------------------------------
# ChatResponse ABC
# ---------------------------------------------------------------------------


class TestChatResponseABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ChatResponse()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# OpenAI completion ChatResponse
# ---------------------------------------------------------------------------


class TestOpenAICompletionResponse:
    def test_response_type(self):
        resp = ChatResponse.openai_completion(make_completion())
        assert resp.response_type == ChatResponseType.OPENAI_COMPLETION

    def test_body_access(self):
        completion = make_completion()
        resp = ChatResponse.openai_completion(completion)
        assert resp.body == completion.model_dump(mode="json", exclude_none=True)
        assert resp.body is not completion

    def test_body_sdk_fields(self):
        resp = ChatResponse.openai_completion(make_completion(model="gpt-4o", content="world"))
        assert resp.body["model"] == "gpt-4o"
        assert resp.body["choices"][0]["message"]["content"] == "world"
        assert resp.body["usage"]["total_tokens"] == 15

    def test_response_type_match(self):
        resp = ChatResponse.openai_completion(make_completion())
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_STREAM)


# ---------------------------------------------------------------------------
# OpenAI stream ChatResponse
# ---------------------------------------------------------------------------


class TestOpenAIStreamChatResponse:
    def test_response_type(self):
        stream = ResponseStream(fake_stream([]))
        resp = ChatResponse.openai_stream(stream)
        assert resp.response_type == ChatResponseType.OPENAI_STREAM

    async def test_stream_access(self):
        chunks = [make_chunk(content="streamed")]
        stream = ResponseStream(fake_stream(chunks))
        resp = ChatResponse.openai_stream(stream)
        assert [chunk async for chunk in resp.stream] == chunks

    def test_response_type_match(self):
        stream = ResponseStream(fake_stream([]))
        resp = ChatResponse.openai_stream(stream)
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.OPENAI_STREAM)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)


# ---------------------------------------------------------------------------
# response type dispatch
# ---------------------------------------------------------------------------


class TestIsinstanceDispatch:
    def _dispatch(self, response: ChatResponse) -> str:
        if response_type_matches(response, ChatResponseType.OPENAI_COMPLETION):
            return "openai_completion"
        if response_type_matches(response, ChatResponseType.OPENAI_STREAM):
            return "openai_stream"
        if response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_COMPLETION):
            return "responses_completion"
        if response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_STREAM):
            return "responses_stream"
        if response_type_matches(response, ChatResponseType.ANTHROPIC_COMPLETION):
            return "anthropic_completion"
        if response_type_matches(response, ChatResponseType.ANTHROPIC_STREAM):
            return "anthropic_stream"
        return "unknown"

    def test_dispatch_completion(self):
        assert self._dispatch(ChatResponse.openai_completion(make_completion())) == "openai_completion"

    def test_dispatch_stream(self):
        resp = ChatResponse.openai_stream(ResponseStream(fake_stream([])))
        assert self._dispatch(resp) == "openai_stream"

    def test_dispatch_responses_completion(self):
        resp = ChatResponse.openai_responses_completion(make_responses_api_response())
        assert self._dispatch(resp) == "responses_completion"

    def test_dispatch_responses_stream(self):
        stream = ResponsesApiStream(fake_responses_stream([]))
        resp = ChatResponse.openai_responses_stream(stream)
        assert self._dispatch(resp) == "responses_stream"

    def test_dispatch_anthropic_completion(self):
        assert self._dispatch(ChatResponse.anthropic_completion(make_anthropic_message())) == "anthropic_completion"

    def test_dispatch_anthropic_stream(self):
        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        resp = ChatResponse.anthropic_stream(stream)
        assert self._dispatch(resp) == "anthropic_stream"


# ---------------------------------------------------------------------------
# ResponseStream
# ---------------------------------------------------------------------------


class TestResponseStream:
    async def test_basic_iteration(self):
        chunks = [make_chunk(content="a"), make_chunk(content="b")]
        stream = ResponseStream(fake_stream(chunks))
        result = [chunk async for chunk in stream]
        assert len(result) == 2
        assert result[0].choices[0].delta.content == "a"
        assert result[1].choices[0].delta.content == "b"

    async def test_empty_stream(self):
        stream = ResponseStream(fake_stream([]))
        result = [chunk async for chunk in stream]
        assert result == []

    async def test_single_consumption(self):
        stream = ResponseStream(fake_stream([make_chunk()]))
        _ = [chunk async for chunk in stream]
        with pytest.raises(RuntimeError, match="already been consumed"):
            _ = [chunk async for chunk in stream]

    async def test_tap_observes_all_chunks(self):
        observed: list[str] = []

        async def log_tap(chunk: ChatCompletionChunk) -> None:
            observed.append(chunk.choices[0].delta.content or "")

        chunks = [make_chunk(content="x"), make_chunk(content="y")]
        stream = ResponseStream(fake_stream(chunks))
        stream.tap(log_tap)

        _ = [chunk async for chunk in stream]
        assert observed == ["x", "y"]

    async def test_map_transforms_chunks(self):
        async def upper_map(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = (chunk.choices[0].delta.content or "").upper()
            return chunk

        stream = ResponseStream(fake_stream([make_chunk(content="hello")]))
        stream.map(upper_map)

        result = [chunk async for chunk in stream]
        assert result[0].choices[0].delta.content == "HELLO"

    async def test_tap_sees_original_before_map(self):
        tap_saw: list[str] = []

        async def observe(chunk: ChatCompletionChunk) -> None:
            tap_saw.append(chunk.choices[0].delta.content or "")

        async def transform(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = "TRANSFORMED"
            return chunk

        stream = ResponseStream(fake_stream([make_chunk(content="original")]))
        stream.tap(observe)
        stream.map(transform)

        result = [chunk async for chunk in stream]
        assert tap_saw == ["original"]
        assert result[0].choices[0].delta.content == "TRANSFORMED"

    async def test_multiple_maps_compose(self):
        async def add_exclaim(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = (chunk.choices[0].delta.content or "") + "!"
            return chunk

        async def add_question(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
            chunk.choices[0].delta.content = (chunk.choices[0].delta.content or "") + "?"
            return chunk

        stream = ResponseStream(fake_stream([make_chunk(content="hi")]))
        stream.map(add_exclaim).map(add_question)

        result = [chunk async for chunk in stream]
        assert result[0].choices[0].delta.content == "hi!?"

    async def test_tap_exception_does_not_break_stream(self):
        async def bad_tap(chunk: ChatCompletionChunk) -> None:
            raise ValueError("tap exploded")

        stream = ResponseStream(fake_stream([make_chunk(content="safe")]))
        stream.tap(bad_tap)

        result = [chunk async for chunk in stream]
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "safe"

    async def test_failing_tap_is_quarantined(self):
        call_count = 0

        async def flaky_tap(chunk: ChatCompletionChunk) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        chunks = [make_chunk(content="a"), make_chunk(content="b"), make_chunk(content="c")]
        stream = ResponseStream(fake_stream(chunks))
        stream.tap(flaky_tap)

        result = [chunk async for chunk in stream]
        assert len(result) == 3
        assert call_count == 1  # called once, then quarantined

    async def test_tap_returns_self_for_chaining(self):
        async def noop(chunk: ChatCompletionChunk) -> None:
            pass

        stream = ResponseStream(fake_stream([]))
        assert stream.tap(noop) is stream

    async def test_map_returns_self_for_chaining(self):
        async def identity(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
            return chunk

        stream = ResponseStream(fake_stream([]))
        assert stream.map(identity) is stream

    async def test_multiple_taps(self):
        seen_a: list[str] = []
        seen_b: list[str] = []

        async def tap_a(chunk: ChatCompletionChunk) -> None:
            seen_a.append(chunk.choices[0].delta.content or "")

        async def tap_b(chunk: ChatCompletionChunk) -> None:
            seen_b.append(chunk.choices[0].delta.content or "")

        stream = ResponseStream(fake_stream([make_chunk(content="x")]))
        stream.tap(tap_a).tap(tap_b)

        _ = [chunk async for chunk in stream]
        assert seen_a == ["x"]
        assert seen_b == ["x"]


# ---------------------------------------------------------------------------
# OpenAI Responses completion ChatResponse
# ---------------------------------------------------------------------------


class TestOpenAIResponsesCompletionResponse:
    def test_response_type(self):
        resp = ChatResponse.openai_responses_completion(make_responses_api_response())
        assert resp.response_type == ChatResponseType.OPENAI_RESPONSES_COMPLETION

    def test_body_access(self):
        body = make_responses_api_response()
        resp = ChatResponse.openai_responses_completion(body)
        assert resp.body == body.model_dump(mode="json", exclude_none=True)
        assert resp.body is not body

    def test_body_sdk_fields(self):
        resp = ChatResponse.openai_responses_completion(
            make_responses_api_response(model="gpt-4o", text="world")
        )
        assert resp.body["model"] == "gpt-4o"
        assert resp.body["output"][0]["content"][0]["text"] == "world"
        assert resp.body["status"] == "completed"

    def test_response_type_match(self):
        resp = ChatResponse.openai_responses_completion(make_responses_api_response())
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.OPENAI_RESPONSES_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_STREAM)
        assert not response_type_matches(resp, ChatResponseType.ANTHROPIC_COMPLETION)


# ---------------------------------------------------------------------------
# OpenAI Responses stream ChatResponse
# ---------------------------------------------------------------------------


class TestOpenAIResponsesStreamChatResponse:
    def test_response_type(self):
        stream = ResponsesApiStream(fake_responses_stream([]))
        resp = ChatResponse.openai_responses_stream(stream)
        assert resp.response_type == ChatResponseType.OPENAI_RESPONSES_STREAM

    async def test_stream_access(self):
        events = [make_responses_text_delta(delta="streamed")]
        stream = ResponsesApiStream(fake_responses_stream(events))
        resp = ChatResponse.openai_responses_stream(stream)
        assert [event async for event in resp.stream] == events

    def test_response_type_match(self):
        stream = ResponsesApiStream(fake_responses_stream([]))
        resp = ChatResponse.openai_responses_stream(stream)
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.OPENAI_RESPONSES_STREAM)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_RESPONSES_COMPLETION)


# ---------------------------------------------------------------------------
# ResponsesApiStream
# ---------------------------------------------------------------------------


class TestResponsesApiStream:
    async def test_basic_iteration(self):
        events = [make_responses_text_delta(delta="a"), make_responses_text_delta(delta="b")]
        stream = ResponsesApiStream(fake_responses_stream(events))
        result = [event async for event in stream]
        assert len(result) == 2
        assert result[0].delta == "a"
        assert result[1].delta == "b"

    async def test_empty_stream(self):
        stream = ResponsesApiStream(fake_responses_stream([]))
        result = [event async for event in stream]
        assert result == []

    async def test_single_consumption(self):
        stream = ResponsesApiStream(fake_responses_stream([make_responses_text_delta()]))
        _ = [event async for event in stream]
        with pytest.raises(RuntimeError, match="already been consumed"):
            _ = [event async for event in stream]

    async def test_tap_observes_all_events(self):
        observed: list[str] = []

        async def log_tap(event: ResponseTextDeltaEvent) -> None:
            observed.append(event.delta)

        events = [make_responses_text_delta(delta="x"), make_responses_text_delta(delta="y")]
        stream = ResponsesApiStream(fake_responses_stream(events))
        stream.tap(log_tap)

        _ = [event async for event in stream]
        assert observed == ["x", "y"]

    async def test_tap_exception_does_not_break_stream(self):
        async def bad_tap(event: ResponseTextDeltaEvent) -> None:
            raise ValueError("tap exploded")

        stream = ResponsesApiStream(
            fake_responses_stream([make_responses_text_delta(delta="safe")])
        )
        stream.tap(bad_tap)

        result = [event async for event in stream]
        assert len(result) == 1
        assert result[0].delta == "safe"

    async def test_failing_tap_is_quarantined(self):
        call_count = 0

        async def flaky_tap(event: ResponseTextDeltaEvent) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        events = [
            make_responses_text_delta(delta="a"),
            make_responses_text_delta(delta="b"),
            make_responses_text_delta(delta="c"),
        ]
        stream = ResponsesApiStream(fake_responses_stream(events))
        stream.tap(flaky_tap)

        result = [event async for event in stream]
        assert len(result) == 3
        assert call_count == 1

    async def test_tap_returns_self_for_chaining(self):
        async def noop(event: ResponseTextDeltaEvent) -> None:
            pass

        stream = ResponsesApiStream(fake_responses_stream([]))
        assert stream.tap(noop) is stream

    async def test_map_returns_self_for_chaining(self):
        async def identity(event: ResponseTextDeltaEvent) -> ResponseTextDeltaEvent:
            return event

        stream = ResponsesApiStream(fake_responses_stream([]))
        assert stream.map(identity) is stream


# ---------------------------------------------------------------------------
# Anthropic completion ChatResponse
# ---------------------------------------------------------------------------


class TestAnthropicCompletionResponse:
    def test_response_type(self):
        resp = ChatResponse.anthropic_completion(make_anthropic_message())
        assert resp.response_type == ChatResponseType.ANTHROPIC_COMPLETION

    def test_body_access(self):
        msg = make_anthropic_message()
        resp = ChatResponse.anthropic_completion(msg)
        assert resp.body == msg.model_dump(mode="json", exclude_none=True)
        assert resp.body is not msg

    def test_body_sdk_fields(self):
        resp = ChatResponse.anthropic_completion(
            make_anthropic_message(model="claude-sonnet-4-20250514", text="world")
        )
        assert resp.body["model"] == "claude-sonnet-4-20250514"
        assert resp.body["content"][0]["text"] == "world"
        assert resp.body["usage"]["input_tokens"] == 10
        assert resp.body["usage"]["output_tokens"] == 5
        assert resp.body["stop_reason"] == "end_turn"

    def test_response_type_match(self):
        resp = ChatResponse.anthropic_completion(make_anthropic_message())
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.ANTHROPIC_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_STREAM)
        assert not response_type_matches(resp, ChatResponseType.ANTHROPIC_STREAM)


# ---------------------------------------------------------------------------
# Anthropic stream ChatResponse
# ---------------------------------------------------------------------------


class TestAnthropicStreamChatResponse:
    def test_response_type(self):
        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        resp = ChatResponse.anthropic_stream(stream)
        assert resp.response_type == ChatResponseType.ANTHROPIC_STREAM

    async def test_stream_access(self):
        events = [make_anthropic_content_delta(text="streamed")]
        stream = AnthropicResponseStream(fake_anthropic_stream(events))
        resp = ChatResponse.anthropic_stream(stream)
        assert [event async for event in resp.stream] == events

    def test_response_type_match(self):
        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        resp = ChatResponse.anthropic_stream(stream)
        assert isinstance(resp, ChatResponse)
        assert response_type_matches(resp, ChatResponseType.ANTHROPIC_STREAM)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert not response_type_matches(resp, ChatResponseType.OPENAI_STREAM)
        assert not response_type_matches(resp, ChatResponseType.ANTHROPIC_COMPLETION)


# ---------------------------------------------------------------------------
# AnthropicResponseStream
# ---------------------------------------------------------------------------


class TestAnthropicResponseStream:
    async def test_basic_iteration(self):
        events = [make_anthropic_content_delta(text="a"), make_anthropic_content_delta(text="b")]
        stream = AnthropicResponseStream(fake_anthropic_stream(events))
        result = [event async for event in stream]
        assert len(result) == 2
        assert result[0].delta.text == "a"
        assert result[1].delta.text == "b"

    async def test_empty_stream(self):
        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        result = [event async for event in stream]
        assert result == []

    async def test_single_consumption(self):
        stream = AnthropicResponseStream(fake_anthropic_stream([make_anthropic_content_delta()]))
        _ = [event async for event in stream]
        with pytest.raises(RuntimeError, match="already been consumed"):
            _ = [event async for event in stream]

    async def test_tap_observes_all_events(self):
        observed: list[str] = []

        async def log_tap(event: RawContentBlockDeltaEvent) -> None:
            observed.append(event.delta.text)

        events = [make_anthropic_content_delta(text="x"), make_anthropic_content_delta(text="y")]
        stream = AnthropicResponseStream(fake_anthropic_stream(events))
        stream.tap(log_tap)

        _ = [event async for event in stream]
        assert observed == ["x", "y"]

    async def test_map_transforms_events(self):
        async def upper_map(event: RawContentBlockDeltaEvent) -> RawContentBlockDeltaEvent:
            return RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=event.index,
                delta=TextDelta(type="text_delta", text=event.delta.text.upper()),
            )

        stream = AnthropicResponseStream(
            fake_anthropic_stream([make_anthropic_content_delta(text="hello")])
        )
        stream.map(upper_map)

        result = [event async for event in stream]
        assert result[0].delta.text == "HELLO"

    async def test_tap_sees_original_before_map(self):
        tap_saw: list[str] = []

        async def observe(event: RawContentBlockDeltaEvent) -> None:
            tap_saw.append(event.delta.text)

        async def transform(event: RawContentBlockDeltaEvent) -> RawContentBlockDeltaEvent:
            return RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=event.index,
                delta=TextDelta(type="text_delta", text="TRANSFORMED"),
            )

        stream = AnthropicResponseStream(
            fake_anthropic_stream([make_anthropic_content_delta(text="original")])
        )
        stream.tap(observe)
        stream.map(transform)

        result = [event async for event in stream]
        assert tap_saw == ["original"]
        assert result[0].delta.text == "TRANSFORMED"

    async def test_tap_exception_does_not_break_stream(self):
        async def bad_tap(event: RawContentBlockDeltaEvent) -> None:
            raise ValueError("tap exploded")

        stream = AnthropicResponseStream(
            fake_anthropic_stream([make_anthropic_content_delta(text="safe")])
        )
        stream.tap(bad_tap)

        result = [event async for event in stream]
        assert len(result) == 1
        assert result[0].delta.text == "safe"

    async def test_failing_tap_is_quarantined(self):
        call_count = 0

        async def flaky_tap(event: RawContentBlockDeltaEvent) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        events = [
            make_anthropic_content_delta(text="a"),
            make_anthropic_content_delta(text="b"),
            make_anthropic_content_delta(text="c"),
        ]
        stream = AnthropicResponseStream(fake_anthropic_stream(events))
        stream.tap(flaky_tap)

        result = [event async for event in stream]
        assert len(result) == 3
        assert call_count == 1

    async def test_tap_returns_self_for_chaining(self):
        async def noop(event: RawContentBlockDeltaEvent) -> None:
            pass

        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        assert stream.tap(noop) is stream

    async def test_map_returns_self_for_chaining(self):
        async def identity(event: RawContentBlockDeltaEvent) -> RawContentBlockDeltaEvent:
            return event

        stream = AnthropicResponseStream(fake_anthropic_stream([]))
        assert stream.map(identity) is stream
