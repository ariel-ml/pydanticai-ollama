"""Ollama model integration for pydantic-ai.

This module provides implementations for interacting with Ollama's language models through the
pydantic-ai framework. It includes the `OllamaModel` class that handles API communication
and response streaming, and `OllamaStreamedResponse` for handling streaming responses from
the Ollama service.

The module integrates with pydantic-ai's model interface to provide a consistent API for
generating text and handling tool calls with Ollama's models.
"""

import os
import json
import urllib.parse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Any, Iterable, Optional, assert_never, cast
from contextlib import asynccontextmanager
import httpx
from ollama import AsyncClient, ChatResponse, ResponseError
from pydantic_core import from_json
from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai._utils import (
    guard_tool_call_id as _guard_tool_call_id,
    PeekableAsyncStream,
    Unset,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ModelMessage,
    ThinkingPart,
    ToolCallPart,
    SystemPromptPart,
    UserPromptPart,
    ToolReturnPart,
    ModelResponseStreamEvent,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage
from pydantic_ai._thinking_part import split_content_into_text_and_thinking
from pydanticai_ollama.settings.ollama import OllamaModelSettings
from pydanticai_ollama.providers.ollama import OllamaProvider


class OllamaModel(Model):
    """Ollama model implementation for pydantic-ai.

    This class provides the concrete implementation for interacting with Ollama's
    language models, handling message formatting, requests, and responses.
    """

    client: AsyncClient
    _model_name: str
    _provider: OllamaProvider

    def __init__(
        self,
        model_name: str,
        provider: OllamaProvider | None = None,
        settings: OllamaModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ):
        """Initializes the OllamaModel.

        Args:
            model_name (str): The name of the Ollama model to use.
            provider (OllamaProvider | None): The Ollama provider instance. Defaults to a new instance.
            settings (OllamaModelSettings | None): Model-specific settings. Defaults to new settings.
            profile (ModelProfileSpec | None): Model profile specifications.
        """
        if provider is None:
            provider = OllamaProvider(base_url="http://localhost:11434")
        if settings is None:
            settings = OllamaModelSettings()
        super().__init__(settings=settings, profile=profile)
        self._model_name = model_name
        self._provider = provider
        self.client = self._provider.client

    @property
    def system(self) -> str:
        """Returns the system identifier for the Ollama model."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Returns the name of the Ollama model."""
        return self._model_name

    @property
    def base_url(self) -> str:
        """Returns the base URL of the Ollama provider."""
        return self._provider.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Sends a request to the Ollama model and returns a ModelResponse.

        Args:
            messages (list[ModelMessage]): The list of messages to send to the model.
            model_settings (ModelSettings | None): Model-specific settings.
            model_request_parameters (ModelRequestParameters): Parameters for the model request.

        Returns:
            ModelResponse: The response from the Ollama model.
        """
        response = cast(
            ChatResponse,
            await self._completions_create(
                messages, model_settings, model_request_parameters
            ),
        )

        model_response = self._process_response(response, model_request_parameters)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Sends a streaming request to the Ollama model and yields a StreamedResponse.

        Args:
            messages (list[ModelMessage]): The list of messages to send to the model.
            model_settings (ModelSettings | None): Model-specific settings.
            model_request_parameters (ModelRequestParameters): Parameters for the model request.
            run_context (RunContext[Any] | None): The run context for the request.

        Yields:
            AsyncIterator[StreamedResponse]: An async iterator that yields a StreamedResponse.
        """
        response = cast(
            AsyncIterator[ChatResponse],
            await self._completions_create(
                messages, model_settings, model_request_parameters, stream=True
            ),
        )

        yield await self._process_streamed_response(response, model_request_parameters)

        # yield OllamaStreamedResponse(
        #     model_request_parameters=model_request_parameters,
        #     _model_name=self._model_name,
        #     _model_profile=self.profile,
        #     _response=cast(AsyncIterator[ChatResponse], response),
        #     _timestamp=datetime.now(timezone.utc),
        # )

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        stream: bool = False,
    ) -> AsyncIterator[ChatResponse] | ChatResponse:
        ollama_messages = self._map_messages(messages)

        options: OllamaModelSettings = (
            cast(OllamaModelSettings, model_settings) if model_settings else {}
        )

        includes_tool_return = self._includes_tool_return(messages[-1])
        if includes_tool_return:
            ollama_tools = []
        else:
            ollama_tools = self._get_tools(model_request_parameters)
            ollama_tools += self._get_builtin_tools(model_request_parameters)

        # Determine the format based on output_tools
        output_format = None
        if (
            model_request_parameters.output_tools
            and model_request_parameters.output_mode == "tool"
            and len(ollama_tools) == 0
        ):
            # Assuming the first output tool defines the structured output format
            output_format = model_request_parameters.output_tools[
                0
            ].parameters_json_schema

        try:
            return await self.client.chat(
                model=self._model_name,
                messages=ollama_messages,
                options=options,
                think=options.get("think", None),
                tools=ollama_tools if ollama_tools else [],
                stream=stream,
                format=output_format,  # Pass the JSON schema as format
                keep_alive=options.get("keep_alive", None),
            )
        except ResponseError as e:
            raise ModelHTTPError(
                status_code=e.status_code, model_name=self._model_name, body=e.error
            ) from e

    def _process_response(
        self, response: ChatResponse, model_request_parameters: ModelRequestParameters
    ) -> ModelResponse:
        """Processes the Ollama response message and extracts parts (ToolCallPart or TextPart)."""
        parts = []
        response_message = response.message
        timestamp = iso_string_to_datetime(response.created_at)
        # If tool_calls are present, prioritize them
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                parts.append(
                    ToolCallPart(
                        tool_name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                    )
                )
        # Otherwise, if content exists, treat it as TextPart
        elif response_message.content:
            # If output_tools are present, attempt to parse content as JSON for a tool call
            if model_request_parameters.output_tools:
                try:
                    content_json = json.loads(response_message.content)
                    tool_name = model_request_parameters.output_tools[0].name
                    parts.append(
                        ToolCallPart(
                            tool_name=tool_name,
                            args=content_json,
                        )
                    )
                except json.JSONDecodeError:
                    # If not valid JSON, fall back to TextPart
                    parts.append(TextPart(content=response_message.content))
            else:
                parts.extend(
                    split_content_into_text_and_thinking(
                        response_message.content, self.profile.thinking_tags
                    )
                )

        return ModelResponse(
            parts=parts,
            model_name=self._model_name,
            usage=_map_usage(response),
            timestamp=timestamp,
        )

    async def _process_streamed_response(
        self,
        response: AsyncIterator[ChatResponse],
        model_request_parameters: ModelRequestParameters,
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                "Streamed response ended without content or tool calls"
            )

        # return GroqStreamedResponse(
        #     model_request_parameters=model_request_parameters,
        #     _response=peekable_response,
        #     _model_name=self._model_name,
        #     _model_profile=self.profile,
        #     _timestamp=number_to_datetime(first_chunk.created),
        # )
        return OllamaStreamedResponse(
            model_request_parameters=model_request_parameters,
            _response=peekable_response,
            _model_name=self._model_name,
            _model_profile=self.profile,
            _timestamp=iso_string_to_datetime(first_chunk.created_at),
        )

    def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Formats a list of pydantic-ai ModelMessage into a list of dictionaries suitable for the Ollama API.

        Args:
            messages (list[ModelMessage]): The messages to format.

        Returns:
            list[dict[str, Any]]: A list of ollama message dictionaries.
        """
        ollama_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                ollama_messages.extend(self._map_user_message(message))
            elif isinstance(message, ModelResponse):
                role = "assistant"
                content = ""
                tool_calls = []
                for part in message.parts:
                    if isinstance(part, TextPart):
                        content += part.content
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append(self._map_tool_call(part))
                    elif isinstance(part, ThinkingPart):
                        # Skip thinking parts when mapping to Ollama messages
                        continue
                    elif isinstance(
                        part, (BuiltinToolCallPart, BuiltinToolReturnPart)
                    ):  # pragma: no cover
                        # This is currently never returned from Olama
                        pass
                    else:
                        assert_never(part)
                if tool_calls:
                    ollama_messages.append(
                        {"role": role, "content": content, "tool_calls": tool_calls}
                    )
                else:
                    ollama_messages.append({"role": role, "content": content})
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages):
            ollama_messages.insert(0, {"role": "system", "content": instructions})
        return ollama_messages

    def _get_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[dict[str, Any]]:
        return [
            self._map_tool_definition(r)
            for r in model_request_parameters.function_tools
        ]

    def _get_builtin_tools(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[dict[str, Any]]:
        if model_request_parameters.builtin_tools:
            raise UserError("Builtin tools are not supported by Ollama.")
        return []

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": f.name,
                "description": f.description or "",
                "parameters": f.parameters_json_schema,
            },
        }

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> dict[str, Any]:
        return {
            "id": _guard_tool_call_id(t=t),
            "type": "function",
            "function": {
                "name": t.tool_name,
                "arguments": t.args_as_dict(),
            },
        }

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[dict[str, Any]]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield {"role": "system", "content": part.content}
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield {
                    "role": "tool",
                    "content": str(part.content),
                    "tool_call_id": _guard_tool_call_id(t=part),
                }
            elif isinstance(part, RetryPromptPart):  # pragma: no branch
                if part.tool_name is None:
                    yield {"role": "user", "content": part.model_response()}
                else:
                    yield {
                        "role": "tool",
                        "content": part.model_response(),
                        "tool_call_id": _guard_tool_call_id(t=part),
                    }

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> dict[str, Any]:
        content: str | list[str]
        images: list[str] = []
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(item)
                elif isinstance(item, ImageUrl):
                    base64_encoded = _get_image_base64(item)
                    if base64_encoded:
                        images.append(base64_encoded)
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        base64_encoded = base64.b64encode(item.data).decode("utf-8")
                        images.append(base64_encoded)
                    else:
                        raise RuntimeError(
                            "Only images are supported for binary content in Ollama."
                        )
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    raise RuntimeError("DocumentUrl is not supported in Ollama.")
                else:  # pragma: no cover
                    raise RuntimeError(f"Unsupported content type: {type(item)}")

        user_prompt: dict[str, Any] = {"role": "user", "content": "".join(content)}
        if images:
            user_prompt["images"] = images
        return user_prompt

    @staticmethod
    def _includes_tool_return(message: ModelMessage) -> bool:
        """Checks if a given ModelMessage represents a tool return response.

        Args:
            message (ModelMessage): The message to check.

        Returns:
            bool: True if the message is a tool return, False otherwise.
        """
        return (
            isinstance(message, ModelRequest)
            and len(message.parts) > 0
            and isinstance(message.parts[0], ToolReturnPart)
        )


@dataclass
class OllamaStreamedResponse(StreamedResponse):
    """Handles streaming responses from the Ollama service, adapting them to pydantic-ai's StreamedResponse.

    Attributes:
        _model_name (str): The name of the Ollama model.
        _ollama_stream (AsyncIterator[ChatResponse]): The raw async iterator from Ollama.
    """

    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterator[ChatResponse]
    _timestamp: datetime

    @property
    def model_name(self) -> str:
        """Returns the name of the model."""
        return self._model_name

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "ollama"

    @property
    def timestamp(self) -> datetime:
        """Returns the timestamp of the response."""
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Asynchronously iterates over Ollama stream chunks and yields ModelResponseStreamEvent.

        This method handles both structured (tool call) and unstructured (text) outputs
        from the Ollama stream, converting them into pydantic-ai's stream events.

        Yields:
            AsyncIterator[ModelResponseStreamEvent]: A stream of model response events.
        """
        struct_content = ""
        async for chunk in self._response:
            self._usage += _map_usage(chunk)
            response_message = chunk.message

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=None,
                        tool_name=tool_call.function.name,
                        args=dict(tool_call.function.arguments),
                        tool_call_id=None,
                    )
                    if maybe_event:
                        yield maybe_event
                continue

            content_delta = response_message.content
            if not content_delta:
                continue

            if self.model_request_parameters.output_tools:
                # If we expect a structured output, buffer the text deltas
                struct_content += content_delta

                # Try to parse the buffered text as partial JSON
                output_json = from_json(
                    struct_content, allow_partial="trailing-strings"
                )

                if output_json:
                    output_tool = self.model_request_parameters.output_tools[0]
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id="structured_output",
                        tool_name=output_tool.name,
                        args=output_json,
                    )
            else:
                # Fallback to normal text streaming if no structured output is expected
                text_event = self._parts_manager.handle_text_delta(
                    vendor_part_id="content",
                    content=content_delta,
                    thinking_tags=self._model_profile.thinking_tags,
                    ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                )
                if text_event:
                    yield text_event


def _map_usage(chunk: ChatResponse) -> RequestUsage:
    """Maps an Ollama chat response chunk to a pydantic-ai Usage object.

    Args:
        chunk (ChatResponse): The raw response chunk from Ollama.

    Returns:
        Usage: The mapped usage information.
    """
    request_tokens = chunk.get("prompt_eval_count") or 0
    response_tokens = chunk.get("eval_count") or 0
    # Only increment requests if there are actual tokens reported in this chunk
    if request_tokens == 0 and response_tokens == 0:
        return RequestUsage(
            input_tokens=0, output_tokens=0
        )  # Return an empty Usage object

    return RequestUsage(
        input_tokens=request_tokens,
        output_tokens=response_tokens,
    )


def _get_image_base64(image_url: ImageUrl) -> Optional[str]:
    """
    Returns a base64 encoded string of an image from a web URL, file URL, or local path.

    Args:
        url_or_path (str): Web URL (http/https), file URL (file://), or direct file path

    Returns:
        str or None: base64 encoded image string, or None on error
    """
    parsed = urllib.parse.urlparse(image_url.url)
    scheme = parsed.scheme

    try:
        if scheme in ["file", ""]:
            # Local file URL or path
            local_path = parsed.path if scheme == "file" else image_url.url
            if os.path.isfile(local_path):
                with open(local_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            else:
                return None
        elif scheme in ["http", "https"]:
            # Web URL; add User-Agent to avoid 403 errors
            headers = {"User-Agent": "Mozilla/5.0 (compatible; OllamaBot/1.0)"}
            response = httpx.get(image_url.url, headers=headers, timeout=10)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode("utf-8")
            else:
                return None
        else:
            return None
    except (httpx.RequestError, OSError):
        return None


def iso_string_to_datetime(iso_string: str | None) -> datetime:
    """Convert an ISO 8601 formatted string to a datetime object.

    Args:
        iso_string: An ISO 8601 formatted string, or None. If None, returns current UTC time.

    Returns:
        datetime: A timezone-aware datetime object. If the input is None, returns current UTC time.
        If the input string has a 'Z' timezone, it's converted to '+00:00' for compatibility.
    """
    if iso_string is None:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
