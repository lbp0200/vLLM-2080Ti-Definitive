# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    RequestResponseMetadata,
)
from vllm.outputs import CompletionOutput, RequestOutput


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
NAMED_CHOICE = {"type": "function", "function": {"name": "get_weather"}}
ARGUMENTS = '{"city": "Shanghai"}'


def _request(stream: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "tool_choice": NAMED_CHOICE,
            "stream": stream,
        }
    )


def _result(text: str, finish_reason: str | None) -> RequestOutput:
    return RequestOutput(
        request_id="named-tool",
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[3],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=finish_reason is not None,
        num_cached_tokens=0,
    )


async def _results(texts: list[tuple[str, str | None]]):
    for text, finish_reason in texts:
        yield _result(text, finish_reason)


class _PlainContentParser:
    """Parser stub representing a model parser without named-choice support."""

    def __init__(self, *_args, **_kwargs):
        self.tool_parser_cls = None

    def parse(
        self,
        model_output,
        _request,
        enable_auto_tools=False,
        model_output_token_ids=(),
    ):
        return None, model_output, []

    def parse_delta(self, delta_text, **_kwargs):
        return DeltaMessage(content=delta_text)


def _serving(parser_cls=None):
    instance = OpenAIServingChat.__new__(OpenAIServingChat)
    instance.parser_cls = parser_cls
    instance.enable_auto_tools = False
    instance.enable_force_include_usage = False
    instance.response_role = "assistant"
    instance.enable_log_outputs = False
    instance.request_logger = None
    instance.enable_prompt_tokens_details = False
    instance.enable_per_request_metrics = False
    instance.system_fingerprint = None
    instance.model_config = None
    return instance


async def _full(text: str):
    request = _request(False)
    return await _serving().chat_completion_full_generator(
        request,
        _results([(text, "stop")]),
        "named-full",
        "test-model",
        request.messages,
        object(),
        RequestResponseMetadata(request_id="named-full"),
        _PlainContentParser(),
    )


async def _full_with_finish(text: str, finish_reason: str):
    request = _request(False)
    return await _serving().chat_completion_full_generator(
        request,
        _results([(text, finish_reason)]),
        "named-full",
        "test-model",
        request.messages,
        object(),
        RequestResponseMetadata(request_id="named-full"),
        _PlainContentParser(),
    )


async def _stream(texts: list[tuple[str, str | None]]):
    request = _request(True)
    chunks = []
    async for item in _serving(_PlainContentParser).chat_completion_stream_generator(
        request,
        _results(texts),
        "named-stream",
        "test-model",
        request.messages,
        object(),
        RequestResponseMetadata(request_id="named-stream"),
    ):
        if item.startswith("data: ") and item.strip() != "data: [DONE]":
            chunks.append(json.loads(item[len("data: ") :].strip()))
    return [choice for chunk in chunks for choice in chunk.get("choices", [])]


def test_named_tool_choice_full_wraps_plain_content():
    response = asyncio.run(_full(ARGUMENTS))
    choice = response.choices[0]

    assert choice.message.content is None
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls
    assert choice.message.tool_calls[0].function.name == "get_weather"
    assert choice.message.tool_calls[0].function.arguments == ARGUMENTS


def test_named_tool_choice_stream_wraps_plain_content():
    choices = asyncio.run(_stream([("{", None), ('"city": "Shanghai"}', "stop")]))
    assert choices[0]["delta"].get("role") == "assistant"
    assert "content" not in choices[0]["delta"]
    tool_deltas = [
        choice["delta"]["tool_calls"][0]
        for choice in choices
        if choice["delta"].get("tool_calls")
    ]

    assert tool_deltas[0]["id"]
    assert tool_deltas[0]["type"] == "function"
    assert tool_deltas[0]["function"] == {
        "name": "get_weather",
        "arguments": "{",
    }
    assert tool_deltas[1]["function"]["arguments"] == '"city": "Shanghai"}'
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_named_tool_choice_full_preserves_truncated_finish_reason():
    response = asyncio.run(_full_with_finish('{"city":', "length"))
    choice = response.choices[0]
    assert choice.message.tool_calls
    assert choice.finish_reason == "length"


def test_named_tool_choice_stream_preserves_truncated_finish_reason():
    choices = asyncio.run(_stream([('{"city":', "length")]))
    assert choices[-1]["finish_reason"] == "length"


def test_named_tool_choice_empty_output_does_not_create_tool_call():
    response = asyncio.run(_full(""))
    choice = response.choices[0]
    assert choice.message.content is None
    assert not choice.message.tool_calls
    assert choice.finish_reason == "stop"
