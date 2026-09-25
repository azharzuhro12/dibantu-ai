"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load variables from .env (if present) into the environment.
load_dotenv()

DEFAULT_GLM_BASE_URL = "https://api.z.ai/api/anthropic"
DEFAULT_GLM_MODEL = "glm-5.3"


#: Default local embedding model for the RAG knowledge base (Step 12);
#: 384-dim, small, fully on-device after a one-time download.
DEFAULT_RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Settings(BaseModel):
    """Runtime settings for DibantuAI."""

    glm_api_key: str = Field(default="", description="API key for the GLM API.")
    glm_base_url: str = Field(
        default=DEFAULT_GLM_BASE_URL, description="Base URL of the GLM API."
    )
    glm_model: str = Field(default=DEFAULT_GLM_MODEL, description="GLM model name to use.")

    knowledge_dir: str = Field(
        default="data/knowledge",
        description="Directory of Markdown knowledge documents to ingest.",
    )
    rag_store_dir: str = Field(
        default="data/rag/chroma",
        description="Directory of the persistent ChromaDB vector store.",
    )
    rag_embeddings: str = Field(
        default="local",
        description="Embedding provider: 'local' (sentence-transformers) "
        "or 'hashing' (deterministic, offline — tests/eval).",
    )
    rag_embedding_model: str = Field(
        default=DEFAULT_RAG_EMBEDDING_MODEL,
        description="sentence-transformers model for local embeddings.",
    )
    rag_top_k: int = Field(
        default=3, description="Default number of passages retrieved per query."
    )


def get_settings() -> Settings:
    """Build a Settings instance from environment variables."""
    return Settings(
        glm_api_key=os.getenv("GLM_API_KEY", ""),
        glm_base_url=os.getenv("GLM_BASE_URL", DEFAULT_GLM_BASE_URL),
        glm_model=os.getenv("GLM_MODEL", DEFAULT_GLM_MODEL),
        knowledge_dir=os.getenv("KNOWLEDGE_DIR", "data/knowledge"),
        rag_store_dir=os.getenv("RAG_STORE_DIR", "data/rag/chroma"),
        rag_embeddings=os.getenv("RAG_EMBEDDINGS", "local"),
        rag_embedding_model=os.getenv(
            "RAG_EMBEDDING_MODEL", DEFAULT_RAG_EMBEDDING_MODEL
        ),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
    )
