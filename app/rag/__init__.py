"""RAG knowledge base for DibantuAI (Step 12).

Local, free retrieval-augmented generation over the business's own
policy documents — no LangChain/LangGraph, no paid APIs:

- ``chunker``      deterministic Markdown-aware chunking
- ``embeddings``   sentence-transformers (default) or offline hashing
- ``vector_store`` persistent ChromaDB collection (cosine space)
- ``retriever``    query embedding -> top-k chunks with citations
- ``service``      ingestion (idempotent) + search, settings-driven
- ``knowledge_tool`` the ``search_knowledge_base`` agent tool
- ``ingest``       CLI: ``python -m app.rag.ingest``

The knowledge base is strictly separate from the PostgreSQL business
data: policies and documentation live here, live stock/orders/customers
stay in the database tools.
"""

from .errors import KnowledgeDirNotFoundError, RagError

__all__ = ["KnowledgeDirNotFoundError", "RagError"]
