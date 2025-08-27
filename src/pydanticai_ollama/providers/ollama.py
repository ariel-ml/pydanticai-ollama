"""Ollama model provider implementation for pydantic-ai.

This module provides the OllamaProvider class which enables integration with Ollama's
language models through a simple interface. It handles API communication and provides
configuration options for the Ollama service endpoint.
"""

import os
from typing import Any, Mapping, Optional
from pydantic_ai.providers import Provider
from pydantic_ai.exceptions import UserError
from ollama import AsyncClient


class OllamaProvider(Provider[AsyncClient]):
    """Provider for local or remote Ollama API."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def base_url(self) -> str:
        return str(self._base_url)

    @property
    def client(self) -> AsyncClient:
        return self._client

    def __init__(
        self,
        base_url: str | None = None,
        follow_redirects: bool = True,
        timeout: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        async_client_kwargs: Optional[Mapping[str, Any]] = None,
        ollama_client: AsyncClient | None = None,
    ) -> None:
        """Create a new Ollama provider.

        Args:
            base_url: The base url for the Ollama requests. If not provided, the `OLLAMA_BASE_URL` environment variable
                will be used if available.
            follow_redirects: Whether to follow redirects for the Ollama API. Defaults to True.
            timeout: Timeout for the Ollama API. Defaults to None.
            headers: Headers to be sent with the Ollama API requests. Defaults to None.
            async_client_kwargs: Additional arguments to pass to the Ollama httpx client. Defaults to None.
            ollama_client: An existing [`AsyncClient`] client to use. If provided, `base_url` must be `None`.
        """
        if ollama_client is not None:
            assert (
                base_url is None
            ), "Cannot provide both `ollama_client` and `base_url`"
            assert (
                timeout is None
            ), "Cannot provide both `ollama_client` and `timeout`"
            assert (
                headers is None
            ), "Cannot provide both `ollama_client` and `headers`"
            assert (
                async_client_kwargs is None
            ), "Cannot provide both `ollama_client` and `async_client_kwargs`"
            self._base_url = ollama_client._client.base_url
            self._client = ollama_client
        else:
            base_url = base_url or os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                raise UserError(
                    "Set the `OLLAMA_BASE_URL` environment variable or pass it via `OllamaProvider(base_url=...)`"
                    "to use the Ollama provider."
                )
            self._base_url = base_url
            self._client = AsyncClient(
                host=base_url,
                follow_redirects=follow_redirects,
                timeout=timeout,
                headers=headers,
                **(async_client_kwargs or {}),
            )
