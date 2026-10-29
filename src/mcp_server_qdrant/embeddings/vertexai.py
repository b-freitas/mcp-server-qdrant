from google import genai
from google.genai import types

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class VertexAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Vertex AI's text embedding models."""

    def __init__(
        self,
        model_name: str,
        vector_size: int | None = None,
        project: str | None = None,
        location: str = "us-central1",
    ):
        self.model_name = model_name
        self.vector_size = vector_size
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=documents,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [embedding.values for embedding in response.embeddings]

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return response.embeddings[0].values

    def get_vector_name(self) -> str:
        model_name = self.model_name.lower().replace("/", "-")
        return f"vertexai-{model_name}"

    def get_vector_size(self) -> int:
        if self.vector_size is None:
            raise ValueError(
                "EMBEDDING_VECTOR_SIZE must be set for the vertexai provider."
            )
        return self.vector_size
