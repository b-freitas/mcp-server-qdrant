from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    COHERE = "cohere"
