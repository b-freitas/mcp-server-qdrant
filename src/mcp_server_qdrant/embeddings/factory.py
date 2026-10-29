from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OPENAI:
        from mcp_server_qdrant.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            model_name=settings.model_name,
            api_key=settings.api_key,
            vector_size=settings.vector_size,
        )
    elif settings.provider_type == EmbeddingProviderType.VERTEXAI:
        from mcp_server_qdrant.embeddings.vertexai import VertexAIEmbeddingProvider

        return VertexAIEmbeddingProvider(
            model_name=settings.model_name,
            vector_size=settings.vector_size,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    elif settings.provider_type == EmbeddingProviderType.COHERE:
        from mcp_server_qdrant.embeddings.cohere import CohereEmbeddingProvider

        return CohereEmbeddingProvider(
            model_name=settings.model_name,
            api_key=settings.api_key,
            vector_size=settings.vector_size,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
