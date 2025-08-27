from __future__ import annotations as _annotations

import re

import pytest
from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from ollama import AsyncClient

    from pydanticai_ollama.providers.ollama import OllamaProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='Ollama not installed')


def test_ollama_provider():
    provider = OllamaProvider(base_url='http://localhost:11434')
    assert provider.name == 'ollama'
    assert provider.base_url == 'http://localhost:11434'
    assert isinstance(provider.client, AsyncClient)


def test_ollama_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OLLAMA_BASE_URL')
    with pytest.raises(
        UserError,
        match=re.escape(
            "Set the `OLLAMA_BASE_URL` environment variable or pass it via `OllamaProvider(base_url=...)`"
            "to use the Ollama provider."
        ),
    ):
        OllamaProvider()


def test_ollama_provider_pass_ollama_client() -> None:
    ollama_client = AsyncClient(host='http://localhost:11434')
    provider = OllamaProvider(ollama_client=ollama_client)
    assert provider.client == ollama_client


def test_ollama_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    provider = OllamaProvider()
    assert provider.base_url == 'http://localhost:11434'
