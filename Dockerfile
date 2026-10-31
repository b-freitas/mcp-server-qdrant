FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the package with the FastEmbed provider extra
RUN uv pip install --system --no-cache-dir 'mcp-server-qdrant-embeddings[fastembed]'

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_PROVIDER="fastembed"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Run the server with SSE transport
CMD ["mcp-server-qdrant-embeddings", "--transport", "sse"]
