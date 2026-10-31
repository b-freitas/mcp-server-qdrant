from openai import AsyncOpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for OpenAI's embeddings API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        vector_size: int | None = None,
    ):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY must be set for the openai provider.")
        self.model_name = model_name
        self.vector_size = vector_size
        self.client = AsyncOpenAI(api_key=api_key)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=documents,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=query,
        )
        return response.data[0].embedding

    def get_vector_name(self) -> str:
        model_name = self.model_name.lower().replace("/", "-")
        return f"openai-{model_name}"

    def get_vector_size(self) -> int:
        if self.vector_size is None:
            raise ValueError(
                "EMBEDDING_VECTOR_SIZE must be set for the openai provider."
            )
        return self.vector_size
