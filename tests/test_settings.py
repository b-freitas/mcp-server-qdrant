import pytest

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        QdrantSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local/qdrant"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if Qdrant URL is provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        with pytest.raises(ValueError):
            QdrantSettings()

        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            QdrantSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Require callers to choose a provider explicitly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type is None
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type is None
        assert settings.model_name == "custom_model"

    def test_openai_config(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "1536")

        settings = EmbeddingProviderSettings()

        assert settings.provider_type == EmbeddingProviderType.OPENAI
        assert settings.model_name == "text-embedding-3-small"
        assert settings.api_key == "test-key"
        assert settings.vector_size == 1536

    def test_cohere_config(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere")
        monkeypatch.setenv("EMBEDDING_MODEL", "embed-v4.0")
        monkeypatch.setenv("EMBEDDING_API_KEY", "cohere-test-key")
        monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "1536")

        settings = EmbeddingProviderSettings()

        assert settings.provider_type == EmbeddingProviderType.COHERE
        assert settings.api_key == "cohere-test-key"
        assert settings.vector_size == 1536

    def test_vertexai_config(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "vertexai")
        monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
        monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "3072")

        settings = EmbeddingProviderSettings()

        assert settings.provider_type == EmbeddingProviderType.VERTEXAI
        assert settings.google_cloud_project == "test-project"
        assert settings.google_cloud_location == "europe-west1"
        assert settings.vector_size == 3072

    def test_vector_size_must_be_positive(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "0")

        with pytest.raises(ValueError):
            EmbeddingProviderSettings()


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_store_description(self, monkeypatch):
        """Test loading custom store description from environment variable."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_find_description(self, monkeypatch):
        """Test loading custom find description from environment variable."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    def test_all_custom_values(self, monkeypatch):
        """Test loading all custom values from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"
