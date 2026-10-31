import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp_server_qdrant_embeddings.embeddings.cohere import CohereEmbeddingProvider
from mcp_server_qdrant_embeddings.embeddings.factory import create_embedding_provider
from mcp_server_qdrant_embeddings.embeddings.vertexai import VertexAIEmbeddingProvider
from mcp_server_qdrant_embeddings.settings import EmbeddingProviderSettings


def test_factory_requires_provider(monkeypatch):
    settings = EmbeddingProviderSettings()

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER is required"):
        create_embedding_provider(settings)


def test_factory_reports_missing_optional_provider_extra(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    settings = EmbeddingProviderSettings()

    def missing_openai_sdk(module_name):
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr(
        "mcp_server_qdrant_embeddings.embeddings.factory.import_module",
        missing_openai_sdk,
    )

    with pytest.raises(RuntimeError, match=r"\[openai\]"):
        create_embedding_provider(settings)


def test_cohere_documents_are_batched_and_marked_for_search():
    provider = CohereEmbeddingProvider("embed-v4.0", "test-key", vector_size=2)
    provider.client.embed = AsyncMock(
        side_effect=[
            SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1, 0.2]] * 96)),
            SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.3, 0.4]])),
        ]
    )
    documents = [f"document {index}" for index in range(97)]

    embeddings = asyncio.run(provider.embed_documents(documents))

    assert len(embeddings) == len(documents)
    assert provider.client.embed.await_count == 2
    first_call = provider.client.embed.await_args_list[0].kwargs
    second_call = provider.client.embed.await_args_list[1].kwargs
    assert len(first_call["texts"]) == 96
    assert second_call["texts"] == documents[96:]
    assert first_call["input_type"] == "search_document"
    assert first_call["embedding_types"] == ["float"]


def test_cohere_query_is_marked_as_search_query():
    provider = CohereEmbeddingProvider("embed-v4.0", "test-key", vector_size=2)
    provider.client.embed = AsyncMock(
        return_value=SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1, 0.2]]))
    )

    embedding = asyncio.run(provider.embed_query("query"))

    assert embedding == [0.1, 0.2]
    provider.client.embed.assert_awaited_once_with(
        model="embed-v4.0",
        texts=["query"],
        input_type="search_query",
        embedding_types=["float"],
    )


def test_vertexai_uses_retrieval_task_types(monkeypatch):
    embed_content = AsyncMock(
        return_value=SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    )
    monkeypatch.setattr(
        "mcp_server_qdrant_embeddings.embeddings.vertexai.genai.Client",
        lambda **kwargs: client,
    )
    provider = VertexAIEmbeddingProvider(
        "gemini-embedding-001",
        vector_size=2,
        project="test-project",
        location="us-central1",
    )

    document_embedding = asyncio.run(provider.embed_documents(["document"]))
    document_config = embed_content.await_args.kwargs["config"]
    query_embedding = asyncio.run(provider.embed_query("query"))
    query_config = embed_content.await_args.kwargs["config"]

    assert document_embedding == [[0.1, 0.2]]
    assert document_config.task_type == "RETRIEVAL_DOCUMENT"
    assert query_embedding == [0.1, 0.2]
    assert query_config.task_type == "RETRIEVAL_QUERY"
    assert provider.get_vector_name() == "vertexai-gemini-embedding-001"
    assert provider.get_vector_size() == 2


def test_factory_selects_cohere_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-v4.0")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "2")
    settings = EmbeddingProviderSettings()

    provider = create_embedding_provider(settings)

    assert isinstance(provider, CohereEmbeddingProvider)
    assert provider.get_vector_size() == 2


def test_factory_selects_vertexai_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "vertexai")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "2")
    settings = EmbeddingProviderSettings()

    provider = create_embedding_provider(settings)

    assert isinstance(provider, VertexAIEmbeddingProvider)
    assert provider.get_vector_size() == 2
