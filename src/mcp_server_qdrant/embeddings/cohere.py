from cohere import AsyncClientV2

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class CohereEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Cohere's Embed API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        vector_size: int | None = None,
    ):
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY must be set for the cohere provider.")
        self.model_name = model_name
        self.vector_size = vector_size
        self.client = AsyncClientV2(api_key=api_key)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(documents), 96):
            response = await self.client.embed(
                model=self.model_name,
                texts=documents[start : start + 96],
                input_type="search_document",
                embedding_types=["float"],
            )
            embeddings.extend(response.embeddings.float_)
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        response = await self.client.embed(
            model=self.model_name,
            texts=[query],
            input_type="search_query",
            embedding_types=["float"],
        )
        return response.embeddings.float_[0]

    def get_vector_name(self) -> str:
        model_name = self.model_name.lower().replace("/", "-")
        return f"cohere-{model_name}"

    def get_vector_size(self) -> int:
        if self.vector_size is None:
            raise ValueError(
                "EMBEDDING_VECTOR_SIZE must be set for the cohere provider."
            )
        return self.vector_size
