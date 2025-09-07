from __future__ import annotations as _annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
import datetime
from typing import Any, Literal, Mapping, Union, cast
from unittest.mock import patch

import httpx
import pytest
from ollama import ResponseError
from dirty_equals import IsListOrTuple
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    BinaryContent,
    FinalResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import (
    IsDatetime,
    IsInstance,
    IsNow,
    IsStr,
    raise_if_exception,
    try_import,
)
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from ollama import AsyncClient, ChatResponse, Message
    from pydanticai_ollama.models.ollama import OllamaModel
    from pydanticai_ollama.settings.ollama import OllamaModelSettings
    from pydanticai_ollama.providers.ollama import OllamaProvider

    # note: we use Union here so that casting works with Python 3.9
    MockChatCompletion = Union[ChatResponse, Exception]
    MockChatCompletionChunk = Union[ChatResponse, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason="ollama not installed"),
    pytest.mark.anyio,
]


def test_init():
    m = OllamaModel(
        "qwen3:4b-intsruct", provider=OllamaProvider(base_url="http://localhost:11434")
    )
    assert m.model_name == "qwen3:4b-intsruct"
    assert m.system == "ollama"
    assert m.base_url == "http://localhost:11434"


@dataclass
class MockOllama:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: (
        Sequence[MockChatCompletionChunk]
        | Sequence[Sequence[MockChatCompletionChunk]]
        | None
    ) = None
    index: int = 0
    _client: httpx.Client = httpx.Client(base_url="http://localhost:11434")

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        return await self.chat_completions_create(*args, **kwargs)

    @classmethod
    def create_mock(
        cls, completions: MockChatCompletion | Sequence[MockChatCompletion]
    ) -> AsyncClient:
        return cast(AsyncClient, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls,
        stream: (
            Sequence[MockChatCompletionChunk]
            | Sequence[Sequence[MockChatCompletionChunk]]
        ),
    ) -> AsyncClient:
        return cast(AsyncClient, cls(stream=stream))

    async def chat_completions_create(
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> ChatResponse | MockAsyncStream[MockChatCompletionChunk]:
        if stream:
            assert (
                self.stream is not None
            ), "you can only used `stream=True` if `stream` is provided"
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(  # pragma: no cover
                    iter(cast(list[MockChatCompletionChunk], self.stream[self.index]))
                )
            else:
                response = MockAsyncStream(
                    iter(cast(list[MockChatCompletionChunk], self.stream))
                )
        else:
            assert (
                self.completions is not None
            ), "you can only used `stream=False` if `completions` are provided"
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(ChatResponse, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(ChatResponse, self.completions)
        self.index += 1
        return response


def completion_message(
    message: Message, *, usage: RequestUsage | None = None
) -> ChatResponse:
    return ChatResponse(
        model="qwen3:4b-instruct",
        message=message,
        created_at=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        done=True,
        done_reason="stop",
        eval_count=usage.output_tokens if usage else 0,
        eval_duration=276276837,
        load_duration=55491704,
        prompt_eval_count=usage.input_tokens if usage else 0,
        prompt_eval_duration=11110509,
        total_duration=343518254,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(Message(content="world", role="assistant"))
    mock_client = MockOllama.create_mock(c)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)

    result = await agent.run("hello")
    assert result.output == "world"
    assert result.usage() == snapshot(RunUsage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run("hello", message_history=result.new_messages())
    assert result.output == "world"
    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="hello", timestamp=IsNow(tz=datetime.timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="world")],
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="hello", timestamp=IsNow(tz=datetime.timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="world")],
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
        ]
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        Message(content="world", role="assistant"),
        usage=RequestUsage(input_tokens=2, output_tokens=1),
    )
    mock_client = MockOllama.create_mock(c)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)

    result = await agent.run("Hello")
    assert result.output == "world"


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        Message(
            content=None,
            role="assistant",
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        arguments={"response": [1, 2, 123]}, name="final_result"
                    ),
                )
            ],
        )
    )
    mock_client = MockOllama.create_mock(c)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m, output_type=list[int])

    result = await agent.run("Hello")
    assert result.output == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="Hello", timestamp=IsNow(tz=datetime.timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={"response": [1, 2, 123]},
                        tool_call_id=IsStr(),
                    )
                ],
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="final_result",
                        content="Final result processed.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                    )
                ]
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            Message(
                content=None,
                role="assistant",
                tool_calls=[
                    Message.ToolCall(
                        function=Message.ToolCall.Function(
                            arguments={"loc_name": "San Fransisco"}, name="get_location"
                        ),
                    )
                ],
            ),
            usage=RequestUsage(
                input_tokens=2,
                output_tokens=1,
            ),
        ),
        completion_message(
            Message(
                content=None,
                role="assistant",
                tool_calls=[
                    Message.ToolCall(
                        function=Message.ToolCall.Function(
                            arguments={"loc_name": "London"}, name="get_location"
                        ),
                    )
                ],
            ),
            usage=RequestUsage(
                input_tokens=3,
                output_tokens=2,
            ),
        ),
        completion_message(Message(content="final response", role="assistant")),
    ]
    mock_client = MockOllama.create_mock(responses)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m, system_prompt="this is the system prompt")

    @agent.tool_plain
    async def get_location(loc_name: str) -> dict[str, Any]:
        if loc_name == "London":
            return {"lat": 51, "lng": 0}
        else:
            raise ModelRetry("Wrong location, please try again")

    result = await agent.run("Hello")
    assert result.output == "final response"
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content="this is the system prompt",
                        timestamp=IsNow(tz=datetime.timezone.utc),
                    ),
                    UserPromptPart(
                        content="Hello", timestamp=IsNow(tz=datetime.timezone.utc)
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_location",
                        args={"loc_name": "San Fransisco"},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name="get_location",
                        content="Wrong location, please try again",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_location",
                        args={"loc_name": "London"},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=2),
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_location",
                        content={"lat": 51, "lng": 0},
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="final response")],
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
            ),
        ]
    )


FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


def chunk(
    delta: Message | None, finish_reason: FinishReason | None = None
) -> ChatResponse:
    return ChatResponse(
        model="qwen3:4b-instruct",
        message=delta or Message(role="assistant"),
        created_at=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        done=True,
        done_reason=finish_reason,
        eval_count=1,
        eval_duration=276276837,
        load_duration=55491704,
        prompt_eval_count=2,
        prompt_eval_duration=11110509,
        total_duration=343518254,
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> ChatResponse:
    return chunk(Message(content=text, role="assistant"), finish_reason=finish_reason)


async def test_stream_text(allow_model_requests: None):
    stream = [text_chunk("hello "), text_chunk("world"), chunk(None)]
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)

    async with agent.run_stream("") as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ["hello ", "hello world", "hello world"]
        )
        assert result.is_complete


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = (
        text_chunk("hello "),
        text_chunk("world"),
        text_chunk(".", finish_reason="stop"),
    )
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)

    async with agent.run_stream("") as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ["hello ", "hello world", "hello world.", "hello world."]
        )
        assert result.is_complete


def struc_chunk(
    tool_name: str | None,
    tool_arguments: Mapping[str, Any] | None,
    finish_reason: FinishReason | None = None,
) -> ChatResponse:
    tool_calls = [
        Message.ToolCall(
            function=Message.ToolCall.Function(
                name=tool_name or "", arguments=tool_arguments or {}
            )
        )
    ]
    return chunk(
        Message(tool_calls=tool_calls, role="assistant"),
        finish_reason=finish_reason,
    )


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_stream_structured(allow_model_requests: None):
    stream = (
        chunk(Message(role="assistant", content="{")),
        chunk(Message(role="assistant", content='"')),
        chunk(Message(role="assistant", content="first")),
        chunk(Message(role="assistant", content='":')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="One")),
        chunk(Message(role="assistant", content='",')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="second")),
        chunk(Message(role="assistant", content='":')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="Two")),
        chunk(Message(role="assistant", content='"')),
        chunk(Message(role="assistant", content="}")),
        chunk(None),
    )
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream("") as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One", "second": ""},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
            ]
        )
        assert result.is_complete

    assert result.usage() == snapshot(
        RunUsage(input_tokens=30, output_tokens=15, requests=1)
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="", timestamp=IsNow(tz=datetime.timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={"first": "One", "second": "Two"},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=30, output_tokens=15),
                model_name="qwen3:4b-instruct",
                timestamp=IsNow(tz=datetime.timezone.utc),
                provider_name="ollama",
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="final_result",
                        content="Output tool not used - a final result was already processed.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                    )
                ]
            ),
        ]
    )


async def test_stream_structured_finish_reason(allow_model_requests: None):
    stream = (
        chunk(Message(role="assistant", content="{")),
        chunk(Message(role="assistant", content='"')),
        chunk(Message(role="assistant", content="first")),
        chunk(Message(role="assistant", content='":')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="One")),
        chunk(Message(role="assistant", content='",')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="second")),
        chunk(Message(role="assistant", content='":')),
        chunk(Message(role="assistant", content=' "')),
        chunk(Message(role="assistant", content="Two")),
        chunk(Message(role="assistant", content='"')),
        chunk(Message(role="assistant", content="}")),
        chunk(None, finish_reason="stop"),
    )
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream("") as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One"},
                {"first": "One", "second": ""},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
                {"first": "One", "second": "Two"},
            ]
        )
        assert result.is_complete


async def test_no_content(allow_model_requests: None):
    stream = chunk(Message(role="assistant")), chunk(Message(role="assistant"))
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m, output_type=MyTypedDict)

    with pytest.raises(UnexpectedModelBehavior, match="Received empty model response"):
        async with agent.run_stream(""):
            pass


async def test_no_delta(allow_model_requests: None):
    stream = chunk(None), text_chunk("hello "), text_chunk("world")
    mock_client = MockOllama.create_mock_stream(stream)
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)

    async with agent.run_stream("") as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ["hello ", "hello world", "hello world"]
        )
        assert result.is_complete


async def test_extra_headers(allow_model_requests: None, ollama_base_url: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = OllamaModel(
        "qwen3:4b-instruct",
        provider=OllamaProvider(
            base_url=ollama_base_url, headers={"Extra-Header-Key": "Extra-Header-Value"}
        ),
    )
    agent = Agent(m)
    await agent.run("hello")


async def test_image_url_input(allow_model_requests: None, ollama_base_url: str):
    m = OllamaModel(
        "gemma3:latest",
        provider=OllamaProvider(base_url=ollama_base_url),
        settings=OllamaModelSettings(temperature=0.0),
    )
    agent = Agent(m)

    result = await agent.run(
        [
            "What is the name of this fruit?",
            ImageUrl(
                url="https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg"
            ),
        ]
    )
    assert result.output == snapshot(
        """\
That’s a potato! \n\

Specifically, it looks like a Russet potato – a very common and popular variety. \n\

Potatoes are vegetables, not fruits.\
"""
    )


@pytest.mark.skip("TODO: won't work")
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, ollama_base_url: str, image_content: BinaryContent
):
    m = OllamaModel(
        "granite3.2-vision:latest", provider=OllamaProvider(base_url=ollama_base_url)
    )
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(
        [
            "What fruit is in the image you can get from the get_image tool (without any arguments)?"
        ]
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What fruit is in the image you can get from the get_image tool (without any arguments)?"
                        ],
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_image", args="{}", tool_call_id="call_wkpd"
                    )
                ],
                usage=RequestUsage(input_tokens=192, output_tokens=8),
                model_name="granite3.2-vision",
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_image",
                        content="See file 1c8566",
                        tool_call_id="call_wkpd",
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            "This is file 1c8566:",
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="The fruit in the image is a kiwi.")],
                usage=RequestUsage(input_tokens=2552, output_tokens=11),
                model_name="granite3.2-vision",
                timestamp=IsDatetime(),
            ),
        ]
    )


@pytest.mark.parametrize("media_type", ["audio/wav", "audio/mpeg"])
async def test_audio_as_binary_content_input(
    allow_model_requests: None, media_type: str
):
    c = completion_message(Message(content="world", role="assistant"))
    mock_client = MockOllama.create_mock(c)
    m = OllamaModel("gemma3:latest", provider=OllamaProvider(ollama_client=mock_client))
    agent = Agent(m)

    base64_content = b"//uQZ"

    with pytest.raises(
        RuntimeError, match="Only images are supported for binary content in Ollama."
    ):
        await agent.run(
            ["hello", BinaryContent(data=base64_content, media_type=media_type)]
        )


async def test_image_as_binary_content_input(
    allow_model_requests: None, ollama_base_url: str, image_content: BinaryContent
) -> None:
    m = OllamaModel(
        "gemma3:latest",
        provider=OllamaProvider(base_url=ollama_base_url),
        settings=OllamaModelSettings(temperature=0.0),
    )
    agent = Agent(m)

    result = await agent.run(["What is the name of this fruit?", image_content])
    assert result.output == snapshot(
        """\
Based on the image, this is a **kiwi**. \n\

You can see the characteristic brown skin with the dark green flesh and tiny black seeds inside.\
"""
    )


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockOllama.create_mock(
        ResponseError(
            error="test error",
            status_code=500,
        )
    )
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(ollama_client=mock_client)
    )
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync("hello")
    assert str(exc_info.value) == snapshot(
        "status_code: 500, model_name: qwen3:4b-instruct, body: test error"
    )


async def test_init_with_provider():
    provider = OllamaProvider(base_url="http://localhost:11434")
    model = OllamaModel("qwen3:4b-instruct", provider=provider)
    assert model.model_name == "qwen3:4b-instruct"
    assert model.client == provider.client


async def test_init_with_provider_string():
    with patch.dict(
        os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}, clear=False
    ):
        model = OllamaModel("qwen3:4b-instruct", provider=OllamaProvider())
        assert model.model_name == "qwen3:4b-instruct"
        assert model.client is not None


async def test_ollama_model_instructions(
    allow_model_requests: None, ollama_base_url: str
):
    m = OllamaModel(
        "qwen3:4b-instruct",
        provider=OllamaProvider(base_url=ollama_base_url),
        settings=OllamaModelSettings(temperature=0.0),
    )
    agent = Agent(m, instructions="You are a helpful assistant.")

    result = await agent.run("What is the capital of France?")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="What is the capital of France?", timestamp=IsDatetime()
                    )
                ],
                instructions="You are a helpful assistant.",
            ),
            ModelResponse(
                parts=[TextPart(content="The capital of France is Paris.")],
                usage=RequestUsage(input_tokens=24, output_tokens=8),
                model_name="qwen3:4b-instruct",
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_ollama_model_builtin_tool(
    allow_model_requests: None, ollama_base_url: str
):
    m = OllamaModel(
        "qwen3:4b-instruct", provider=OllamaProvider(base_url=ollama_base_url)
    )
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    with pytest.raises(UserError, match="Builtin tools are not supported by Ollama."):
        await agent.run("What day is today?")


async def test_ollama_model_thinking_part(
    allow_model_requests: None, ollama_base_url: str
):
    m = OllamaModel("qwen3:latest", provider=OllamaProvider(base_url=ollama_base_url))
    settings = OllamaModelSettings(temperature=0.0, think=None)
    agent = Agent(m, instructions="You are a chef.", model_settings=settings)

    result = await agent.run("I want a recipe to cook Uruguayan alfajores.")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="I want a recipe to cook Uruguayan alfajores.",
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a chef.",
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=31, output_tokens=1272),
                model_name="qwen3:latest",
                timestamp=IsDatetime(),
            ),
        ]
    )

    result = await agent.run(
        "Considering the Uruguayan recipe, how can I cook the Argentinian one?",
        message_history=result.all_messages(),
        model_settings=OllamaModelSettings(temperature=0.0, think=None),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="I want a recipe to cook Uruguayan alfajores.",
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a chef.",
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=31, output_tokens=1272),
                model_name="qwen3:latest",
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="Considering the Uruguayan recipe, how can I cook the Argentinian one?",
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="You are a chef.",
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=702, output_tokens=1523),
                model_name="qwen3:latest",
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_ollama_model_thinking_part_iter(
    allow_model_requests: None, ollama_base_url: str
):
    m = OllamaModel("qwen3:latest", provider=OllamaProvider(base_url=ollama_base_url))
    settings = OllamaModelSettings(temperature=0.0, think=None)
    agent = Agent(m, instructions="You're good at geography.", model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt="What is the capital of France?") as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        IsListOrTuple(
            positions={
                0: PartStartEvent(index=0, part=ThinkingPart(content="")),
                1: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="\n")),
                2: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta="Okay")
                ),
                3: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=",")),
                4: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" the")
                ),
                5: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" user")
                ),
                6: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" is")
                ),
                7: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" asking")
                ),
                8: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" for")
                ),
                9: PartDeltaEvent(
                    index=0, delta=ThinkingPartDelta(content_delta=" the")
                ),
                177: PartStartEvent(index=1, part=TextPart(content="\n\n")),
                178: FinalResultEvent(tool_name=None, tool_call_id=None),
                179: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="The")),
                180: PartDeltaEvent(
                    index=1, delta=TextPartDelta(content_delta=" capital")
                ),
                181: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" of")),
                182: PartDeltaEvent(
                    index=1, delta=TextPartDelta(content_delta=" France")
                ),
                183: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" is")),
            }
        )
    )
