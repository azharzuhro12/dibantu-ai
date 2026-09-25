"""Local embedding providers for the RAG knowledge base (Step 12).

Two interchangeable providers behind one small interface
(``embed_documents`` / ``embed_query``):

- ``LocalEmbeddings`` — sentence-transformers with the configured model
  (default all-MiniLM-L6-v2). Runs fully on-device after a one-time
  model download; no API calls, no per-request cost.
- ``HashingEmbeddings`` — deterministic token-hashing bag-of-words
  vectors: no dependencies, no network, no model download. Semantic
  quality is limited to shared vocabulary (queries and passages that
  share words rank close), which is exactly enough for deterministic
  offline tests and evaluations — it is never the production default.

Both emit L2-normalised vectors. all-MiniLM-L6-v2 happens to be
384-dimensional, so ``HashingEmbeddings`` uses the same width, but the
vector store records the provider (and its dimension) per collection:
switching providers always requires re-ingesting into a fresh store.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

from .errors import RagError

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbeddings",
    "LocalEmbeddings",
    "build_embeddings",
]

#: Output width of all-MiniLM-L6-v2; also used by HashingEmbeddings.
EMBEDDING_DIMENSION = 384

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Tokens for HashingEmbeddings: lowercase alphanumeric runs. Handles
#: Indonesian and English alike ("sudah-dibayar" -> ["sudah", "dibayar"]).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingError(RagError):
    """Raised when embeddings cannot be produced (deps/model missing)."""


class EmbeddingProvider(Protocol):
    """Anything that can turn text into fixed-size float vectors."""

    #: Stable identifier recorded in the vector store ("local"/"hashing").
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages (chunk contents)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""
        ...


class HashingEmbeddings:
    """Deterministic hashing-trick bag-of-words vectors (offline).

    Each token is hashed (blake2b, stable across runs and platforms)
    into one of ``dimension`` buckets; bucket counts are L2-normalised.
    Vectors are immutable per text, so tests and evaluations replay
    identically without any model download.
    """

    name = "hashing"
    dimension = EMBEDDING_DIMENSION

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


class LocalEmbeddings:
    """sentence-transformers embeddings computed on the local machine.

    The model loads lazily on first use (constructor), so importing this
    module never pulls in torch. Loading downloads the model once into
    the local Hugging Face cache (honour ``HF_HOME``); afterwards it is
    fully offline.
    """

    name = "local"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env-specific
            raise EmbeddingError(
                "sentence-transformers is not installed; install the "
                "RAG dependencies (see requirements.txt) or set "
                "RAG_EMBEDDINGS=hashing for offline runs."
            ) from exc
        try:
            self._model: Any = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - download/load failures vary
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}' "
                "(first run downloads it from Hugging Face — check "
                f"network access and the HF_HOME cache): {type(exc).__name__}"
            ) from exc
        # sentence-transformers >= 6 renamed this method; support both.
        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dimension = int(get_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in encoded]

    def embed_query(self, text: str) -> list[float]:
        encoded = self._model.encode([text], normalize_embeddings=True)
        return encoded[0].tolist()


def build_embeddings(
    provider: str = "local",
    model_name: str | None = None,
) -> EmbeddingProvider:
    """Create the configured embedding provider.

    ``provider`` is ``"local"`` (sentence-transformers, default) or
    ``"hashing"`` (deterministic offline vectors). Unknown names raise
    ``EmbeddingError``.
    """
    if provider == HashingEmbeddings.name:
        return HashingEmbeddings()
    if provider == LocalEmbeddings.name:
        return LocalEmbeddings(model_name or DEFAULT_EMBEDDING_MODEL)
    raise EmbeddingError(
        f"Unknown embeddings provider '{provider}' (expected 'local' or 'hashing')."
    )
