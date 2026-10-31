from collections.abc import Callable
from importlib import import_module

from mcp_server_qdrant_embeddings.embeddings.base import EmbeddingProvider
from mcp_server_qdrant_embeddings.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant_embeddings.settings import EmbeddingProviderSettings


def _load_provider(
    provider_type: EmbeddingProviderType,
    module_name: str,
    class_name: str,
) -> Callable[..., EmbeddingProvider]:
    try:
        provider_module = import_module(module_name)
    except ImportError as exc:
        if exc.name and exc.name.startswith("mcp_server_qdrant_embeddings"):
            raise
        raise RuntimeError(
            f"The '{provider_type.value}' provider requires optional dependencies. "
            f"Install mcp-server-qdrant-embeddings[{provider_type.value}]."
        ) from exc
    return getattr(provider_module, class_name)


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type is None:
        raise ValueError(
            "EMBEDDING_PROVIDER is required. Choose fastembed, openai, vertexai, "
            "or cohere, and install the matching optional extra."
        )

    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        FastEmbedProvider = _load_provider(
            settings.provider_type,
            "mcp_server_qdrant_embeddings.embeddings.fastembed",
            "FastEmbedProvider",
        )
        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OPENAI:
        OpenAIEmbeddingProvider = _load_provider(
            settings.provider_type,
            "mcp_server_qdrant_embeddings.embeddings.openai",
            "OpenAIEmbeddingProvider",
        )
        return OpenAIEmbeddingProvider(
            model_name=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.openai_base_url,
            vector_size=settings.vector_size,
        )
    elif settings.provider_type == EmbeddingProviderType.VERTEXAI:
        VertexAIEmbeddingProvider = _load_provider(
            settings.provider_type,
            "mcp_server_qdrant_embeddings.embeddings.vertexai",
            "VertexAIEmbeddingProvider",
        )
        return VertexAIEmbeddingProvider(
            model_name=settings.model_name,
            vector_size=settings.vector_size,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    elif settings.provider_type == EmbeddingProviderType.COHERE:
        CohereEmbeddingProvider = _load_provider(
            settings.provider_type,
            "mcp_server_qdrant_embeddings.embeddings.cohere",
            "CohereEmbeddingProvider",
        )
        return CohereEmbeddingProvider(
            model_name=settings.model_name,
            api_key=settings.api_key,
            vector_size=settings.vector_size,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
