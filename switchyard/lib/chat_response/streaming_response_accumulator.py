# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulate streaming responses into completed native responses."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from anthropic.types import (
    ContentBlock,
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    Usage as AnthropicUsage,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as ChatCompletionToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response as OpenAIResponse,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import TypeAdapter, ValidationError

from switchyard_rust.core import (
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)

_ANTHROPIC_EVENT_ADAPTER: TypeAdapter[RawMessageStreamEvent] = TypeAdapter(RawMessageStreamEvent)
_OpenAIFinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
]


class StreamingResponseAccumulator(Protocol):
    """Accumulates provider-native stream events into a completed response."""

    def consume(self, event: object) -> None: ...

    def as_response(self) -> ChatResponse: ...


CompletedResponseCallback = Callable[[ChatResponse], Awaitable[None]]


@dataclass
class _OpenAIToolCallState:
    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def to_openai(self, *, fallback_id: str) -> ChatCompletionMessageFunctionToolCall:
        return ChatCompletionMessageFunctionToolCall(
            id=self.id or fallback_id,
            type="function",
            function=ChatCompletionToolCallFunction(
                name=self.name or "",
                arguments=self.arguments,
            ),
        )


@dataclass
class _AnthropicContentBlockState:
    kind: Literal["text", "tool_use"]
    text: str = ""
    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def to_anthropic(self, *, fallback_id: str) -> TextBlock | ToolUseBlock:
        if self.kind == "text":
            return TextBlock(type="text", text=self.text)
        return ToolUseBlock(
            type="tool_use",
            id=self.id or fallback_id,
            name=self.name or "",
            input=_json_object_from_argument_string(self.arguments),
        )


@dataclass
class _ResponsesToolCallState:
    id: str | None = None
    call_id: str | None = None
    name: str = ""
    arguments: str = ""

    def to_response_output_item(self, *, fallback_id: str) -> ResponseFunctionToolCall | None:
        if not self.name and not self.arguments:
            return None
        return ResponseFunctionToolCall(
            type="function_call",
            id=self.id,
            call_id=self.call_id or self.id or fallback_id,
            name=self.name,
            arguments=self.arguments,
            status="completed",
        )


def attach_final_response_callback(
    response: ChatResponse,
    *,
    served_model: str,
    callback: CompletedResponseCallback,
) -> bool:
    """Attach a callback that receives the completed native response.

    Returns ``False`` when *response* is not a supported streaming response.
    The callback only runs when the stream drains normally; stream wrappers own
    that completion contract.
    """
    accumulator = create_streaming_response_accumulator(
        response,
        served_model=served_model,
    )
    if accumulator is None:
        return False

    async def _tap(event: object) -> None:
        accumulator.consume(event)

    async def _on_complete() -> None:
        await callback(accumulator.as_response())

    if response_type_matches(response, ChatResponseType.OPENAI_STREAM):
        response.stream.tap(_tap).on_complete(_on_complete)
    elif response_type_matches(response, ChatResponseType.ANTHROPIC_STREAM):
        response.stream.tap(_tap).on_complete(_on_complete)
    elif response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_STREAM):
        response.stream.tap(_tap).on_complete(_on_complete)
    else:
        return False
    return True


def create_streaming_response_accumulator(
    response: ChatResponse,
    *,
    served_model: str,
) -> StreamingResponseAccumulator | None:
    """Create the native accumulator for a streaming response."""
    if response_type_matches(response, ChatResponseType.OPENAI_STREAM):
        return _OpenAIChatStreamAccumulator(served_model=served_model)
    if response_type_matches(response, ChatResponseType.ANTHROPIC_STREAM):
        return _AnthropicStreamAccumulator(served_model=served_model)
    if response_type_matches(response, ChatResponseType.OPENAI_RESPONSES_STREAM):
        return _ResponsesStreamAccumulator(served_model=served_model)
    return None


class _OpenAIChatStreamAccumulator:
    """Accumulate OpenAI Chat Completion chunks into one ChatCompletion."""

    def __init__(self, *, served_model: str) -> None:
        self._served_model = served_model
        self._content = ""
        self._reasoning_content = ""
        self._tool_calls: list[_OpenAIToolCallState] = []
        self._usage: CompletionUsage | None = None
        self._finish_reason: str | None = None
        self._response_id: str | None = None
        self._created: int | None = None

    def consume(self, event: object) -> None:
        chunk = event
        if isinstance(chunk, Mapping):
            chunk = ChatCompletionChunk.model_validate(chunk)
        if not isinstance(chunk, ChatCompletionChunk):
            return

        self._response_id = chunk.id or self._response_id
        self._created = chunk.created or self._created
        self._served_model = chunk.model or self._served_model
        if chunk.usage is not None:
            self._usage = chunk.usage
        if not chunk.choices:
            return

        choice = chunk.choices[0]
        delta = choice.delta
        if delta is not None:
            if isinstance(delta.content, str):
                self._content += delta.content
            reasoning_content = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning_content, str):
                self._reasoning_content += reasoning_content
            for tool_call in delta.tool_calls or []:
                self._merge_tool_call(tool_call)

        if choice.finish_reason is not None:
            self._finish_reason = choice.finish_reason

    def as_response(self) -> ChatResponse:
        message = ChatCompletionMessage(
            role="assistant",
            content=self._content or None,
            tool_calls=[
                tool_call.to_openai(fallback_id=f"call_switchyard_{index}")
                for index, tool_call in enumerate(self._tool_calls)
            ] or None,
        )
        response = ChatCompletion(
            id=self._response_id or "chatcmpl-switchyard-stream",
            object="chat.completion",
            created=self._created or int(time.time()),
            model=self._served_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=_openai_finish_reason(
                        self._finish_reason,
                        has_tools=bool(self._tool_calls),
                    ),
                ),
            ],
            usage=self._usage,
        )
        if self._reasoning_content:
            # The OpenAI SDK does not type vendor-specific reasoning fields
            # on final messages. Preserve them at the serialization boundary.
            response_dict = response.model_dump(mode="json", exclude_none=True)
            response_dict["choices"][0]["message"]["reasoning_content"] = (
                self._reasoning_content
            )
            response = ChatCompletion.model_validate(response_dict)
        return ChatResponse.openai_completion(response)

    def _merge_tool_call(self, tool_call: object) -> None:
        index = getattr(tool_call, "index", None)
        if not isinstance(index, int):
            index = len(self._tool_calls)
        while len(self._tool_calls) <= index:
            self._tool_calls.append(_OpenAIToolCallState())
        existing = self._tool_calls[index]
        tool_call_id = getattr(tool_call, "id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            existing.id = tool_call_id
        func = getattr(tool_call, "function", None)
        if func is None:
            return
        name = getattr(func, "name", None)
        if isinstance(name, str) and name:
            existing.name = name
        arguments = getattr(func, "arguments", None)
        if isinstance(arguments, str) and arguments:
            existing.arguments += arguments


class _AnthropicStreamAccumulator:
    """Accumulate Anthropic Messages events into one Anthropic message."""

    def __init__(self, *, served_model: str) -> None:
        self._served_model = served_model
        self._response_id: str | None = None
        self._content_blocks: dict[int, _AnthropicContentBlockState] = {}
        self._usage: dict[str, int] = {}
        self._stop_reason: str | None = None

    def consume(self, event: object) -> None:
        typed_event = _coerce_anthropic_event(event)
        if isinstance(typed_event, Mapping):
            self._consume_mapping_event(typed_event)
            return

        if isinstance(typed_event, RawMessageStartEvent):
            self._response_id = typed_event.message.id or self._response_id
            self._served_model = typed_event.message.model or self._served_model
            self._merge_usage(typed_event.message.usage)
            return

        if isinstance(typed_event, RawContentBlockStartEvent):
            block = typed_event.content_block
            if isinstance(block, ToolUseBlock):
                self._content_blocks[typed_event.index] = _AnthropicContentBlockState(
                    kind="tool_use",
                    id=block.id,
                    name=block.name,
                    arguments=_json_argument_string(block.input),
                )
                return
            if isinstance(block, TextBlock):
                self._content_blocks[typed_event.index] = _AnthropicContentBlockState(
                    kind="text",
                    text=block.text,
                )
                return
            self._content_blocks[typed_event.index] = _AnthropicContentBlockState(
                kind="text",
            )
            return

        if isinstance(typed_event, RawContentBlockDeltaEvent):
            state = self._content_blocks.setdefault(
                typed_event.index,
                _AnthropicContentBlockState(kind="text"),
            )
            delta = typed_event.delta
            if isinstance(delta, InputJSONDelta):
                if state.kind != "tool_use":
                    state.kind = "tool_use"
                    state.text = ""
                    state.id = None
                    state.name = None
                    state.arguments = ""
                state.arguments += delta.partial_json
                return
            if isinstance(delta, TextDelta) and state.kind == "text":
                state.text += delta.text
                return
            if isinstance(delta, ThinkingDelta) and state.kind == "text":
                state.text += delta.thinking
                return

        if isinstance(typed_event, RawMessageDeltaEvent):
            self._stop_reason = typed_event.delta.stop_reason or self._stop_reason
            self._merge_usage(typed_event.usage)

    def _consume_mapping_event(self, event: Mapping[str, object]) -> None:
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message")
            if not isinstance(message, Mapping):
                return
            response_id = message.get("id")
            if isinstance(response_id, str):
                self._response_id = response_id
            model = message.get("model")
            if isinstance(model, str):
                self._served_model = model
            self._merge_usage(message.get("usage"))
            return

        if event_type == "content_block_start":
            index = _int_value(event.get("index"), 0)
            block = event.get("content_block")
            if not isinstance(block, Mapping):
                return
            if block.get("type") == "tool_use":
                self._content_blocks[index] = _AnthropicContentBlockState(
                    kind="tool_use",
                    id=_str_value(block.get("id")),
                    name=_str_value(block.get("name")),
                    arguments=_json_argument_string(block.get("input")),
                )
                return
            text = block.get("text")
            self._content_blocks[index] = _AnthropicContentBlockState(
                kind="text",
                text=text if isinstance(text, str) else "",
            )
            return

        if event_type == "content_block_delta":
            index = _int_value(event.get("index"), 0)
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                return
            block = self._content_blocks.setdefault(
                index,
                _AnthropicContentBlockState(kind="text"),
            )
            if delta.get("type") == "input_json_delta":
                if block.kind != "tool_use":
                    block.kind = "tool_use"
                    block.text = ""
                    block.id = None
                    block.name = None
                    block.arguments = ""
                partial_json = delta.get("partial_json")
                if isinstance(partial_json, str):
                    block.arguments += partial_json
                return
            if block.kind != "text":
                return
            text = delta.get("text")
            if isinstance(text, str):
                block.text += text
                return
            thinking = delta.get("thinking")
            if isinstance(thinking, str):
                block.text += thinking
            return

        if event_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping):
                stop_reason = delta.get("stop_reason")
                if isinstance(stop_reason, str):
                    self._stop_reason = stop_reason
            self._merge_usage(event.get("usage"))

    def as_response(self) -> ChatResponse:
        content: list[ContentBlock] = []
        has_tools = False
        for _, block in sorted(self._content_blocks.items()):
            if block.kind == "tool_use":
                has_tools = True
            content.append(
                block.to_anthropic(fallback_id=f"toolu_switchyard_{len(content)}"),
            )

        response = AnthropicMessage(
            id=self._response_id or "msg_switchyard_stream",
            type="message",
            role="assistant",
            content=content,
            model=self._served_model,
            stop_reason=cast(StopReason, self._stop_reason or ("tool_use" if has_tools else "end_turn")),
            stop_sequence=None,
            usage=AnthropicUsage(
                input_tokens=self._usage.get("input_tokens", 0),
                output_tokens=self._usage.get("output_tokens", 0),
                cache_creation_input_tokens=self._usage.get(
                    "cache_creation_input_tokens",
                    0,
                ),
                cache_read_input_tokens=self._usage.get("cache_read_input_tokens", 0),
            ),
        )
        return ChatResponse.anthropic_completion(response)

    def _merge_usage(self, usage: object) -> None:
        if usage is None:
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = usage.get(key) if isinstance(usage, Mapping) else getattr(usage, key, None)
            if isinstance(value, int):
                self._usage[key] = value


class _ResponsesStreamAccumulator:
    """Accumulate Responses API stream events into one Response."""

    def __init__(self, *, served_model: str) -> None:
        self._served_model = served_model
        self._response_id: str | None = None
        self._created_at: float | None = None
        self._content = ""
        self._tool_calls: dict[int, _ResponsesToolCallState] = {}
        self._usage: ResponseUsage | None = None
        self._final_response: OpenAIResponse | Mapping[str, object] | None = None

    def consume(self, event: object) -> None:
        if isinstance(event, ResponseTextDeltaEvent):
            self._content += event.delta
            return

        if isinstance(event, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent)):
            self._merge_output_item(event.item, event.output_index)
            return

        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            tool_call = self._tool_calls.setdefault(
                event.output_index,
                _ResponsesToolCallState(),
            )
            tool_call.arguments += event.delta
            return

        if isinstance(event, ResponseCompletedEvent):
            self._capture_response_metadata(event.response)
            self._final_response = event.response
            return

        if isinstance(event, Mapping):
            self._consume_mapping_event(event)
            return

        response = getattr(event, "response", None)
        if isinstance(response, OpenAIResponse):
            self._capture_response_metadata(response)

    def as_response(self) -> ChatResponse:
        if isinstance(self._final_response, OpenAIResponse):
            return ChatResponse.openai_responses_completion(self._final_response)
        if isinstance(self._final_response, Mapping):
            return ChatResponse.openai_responses_completion(
                OpenAIResponse.model_validate(dict(self._final_response)),
            )

        response = OpenAIResponse.model_validate({
            "id": self._response_id or "resp_switchyard_stream",
            "object": "response",
            "created_at": self._created_at or time.time(),
            "status": "completed",
            "model": self._served_model,
            "output": self._output_items(),
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": self._usage.model_dump(mode="json") if self._usage else None,
        })
        return ChatResponse.openai_responses_completion(response)

    def _consume_mapping_event(self, event: Mapping[str, object]) -> None:
        event_type = event.get("type")
        response = event.get("response")
        if isinstance(response, Mapping):
            self._capture_response_mapping(response)

        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                self._content += delta
            return
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._merge_output_item_mapping(
                event.get("item"),
                _int_value(event.get("output_index"), 0),
            )
            return
        if event_type == "response.function_call_arguments.delta":
            tool_call = self._tool_calls.setdefault(
                _int_value(event.get("output_index"), 0),
                _ResponsesToolCallState(),
            )
            delta = event.get("delta")
            if isinstance(delta, str):
                tool_call.arguments += delta
            return
        if event_type == "response.completed" and isinstance(response, Mapping):
            self._final_response = response

    def _capture_response_metadata(self, response: OpenAIResponse) -> None:
        self._response_id = response.id or self._response_id
        self._served_model = response.model or self._served_model
        self._created_at = response.created_at or self._created_at
        if response.usage is not None:
            self._usage = response.usage

    def _capture_response_mapping(self, response: Mapping[str, object]) -> None:
        response_id = response.get("id")
        if isinstance(response_id, str):
            self._response_id = response_id
        model = response.get("model")
        if isinstance(model, str):
            self._served_model = model
        created = response.get("created_at") or response.get("created")
        if isinstance(created, (int, float)):
            self._created_at = float(created)
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            self._usage = _responses_usage_from_mapping(usage)

    def _merge_output_item(self, item: object, output_index: int) -> None:
        if isinstance(item, ResponseOutputMessage):
            content = _extract_responses_message_text(item)
            if content and not self._content:
                self._content = content
            return
        if not isinstance(item, ResponseFunctionToolCall):
            return
        tool_call = self._tool_calls.setdefault(output_index, _ResponsesToolCallState())
        tool_call.id = item.id
        tool_call.call_id = item.call_id
        tool_call.name = item.name
        tool_call.arguments = item.arguments

    def _merge_output_item_mapping(self, item: object, output_index: int) -> None:
        if not isinstance(item, Mapping):
            return
        item_type = item.get("type")
        if item_type == "message":
            content = _extract_responses_message_text_mapping(item)
            if content and not self._content:
                self._content = content
            return
        if item_type != "function_call":
            return
        tool_call = self._tool_calls.setdefault(output_index, _ResponsesToolCallState())
        call_id = item.get("call_id") or item.get("id")
        if isinstance(call_id, str):
            tool_call.call_id = call_id
        name = item.get("name")
        if isinstance(name, str):
            tool_call.name = name
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            tool_call.arguments = arguments

    def _output_items(self) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        if self._content:
            output.append({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": self._content}],
            })
        for _, call in sorted(self._tool_calls.items()):
            item = call.to_response_output_item(
                fallback_id=f"fc_switchyard_{len(output)}",
            )
            if item is not None:
                output.append(item.model_dump(mode="json", exclude_none=True))
        return output


def _coerce_anthropic_event(
    event: object,
) -> RawMessageStreamEvent | Mapping[str, object]:
    if isinstance(event, Mapping):
        try:
            return _ANTHROPIC_EVENT_ADAPTER.validate_python(event)
        except ValidationError:
            return event
    return cast(RawMessageStreamEvent, event)


def _json_argument_string(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value))
    return ""


def _json_object_from_argument_string(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _openai_finish_reason(
    value: str | None,
    *,
    has_tools: bool,
) -> _OpenAIFinishReason:
    if value in {"stop", "length", "tool_calls", "content_filter", "function_call"}:
        return cast(_OpenAIFinishReason, value)
    return "tool_calls" if has_tools else "stop"


def _responses_usage_from_mapping(usage: Mapping[str, object]) -> ResponseUsage:
    input_tokens = _int_value(usage.get("input_tokens"), 0)
    output_tokens = _int_value(usage.get("output_tokens"), 0)
    total_tokens = _int_value(
        usage.get("total_tokens"),
        input_tokens + output_tokens,
    )
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=(
                _int_value(input_details.get("cached_tokens"), 0)
                if isinstance(input_details, Mapping)
                else 0
            ),
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=(
                _int_value(output_details.get("reasoning_tokens"), 0)
                if isinstance(output_details, Mapping)
                else 0
            ),
        ),
    )


def _extract_responses_message_text(item: ResponseOutputMessage) -> str:
    parts: list[str] = []
    for part in item.content:
        if isinstance(part, ResponseOutputText):
            parts.append(part.text)
    return "".join(parts)


def _extract_responses_message_text_mapping(item: Mapping[str, object]) -> str:
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") not in {"output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _int_value(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _str_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
