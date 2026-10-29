import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def test_embed_documents_uses_openai_embeddings_api():
    provider = OpenAIEmbeddingProvider(
        "example/model", api_key="test-key", vector_size=3
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )
    )
    provider.client.embeddings.create = create

    embeddings = asyncio.run(provider.embed_documents(["first", "second"]))

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    create.assert_awaited_once_with(model="example/model", input=["first", "second"])


def test_embed_query_uses_openai_embeddings_api():
    provider = OpenAIEmbeddingProvider(
        "example/model", api_key="test-key", vector_size=3
    )
    create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
    )
    provider.client.embeddings.create = create

    embedding = asyncio.run(provider.embed_query("query"))

    assert embedding == [0.1, 0.2, 0.3]
    create.assert_awaited_once_with(model="example/model", input="query")


def test_vector_configuration():
    provider = OpenAIEmbeddingProvider(
        "example/model", api_key="test-key", vector_size=3
    )

    assert provider.get_vector_name() == "openai-example-model"
    assert provider.get_vector_size() == 3


def test_vector_size_is_required():
    provider = OpenAIEmbeddingProvider("example/model", api_key="test-key")

    with pytest.raises(ValueError, match="EMBEDDING_VECTOR_SIZE"):
        provider.get_vector_size()


def test_factory_creates_openai_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "example/model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "3")
    settings = EmbeddingProviderSettings()

    provider = create_embedding_provider(settings)

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.get_vector_size() == 3
