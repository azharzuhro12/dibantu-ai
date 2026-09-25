"""CLI: ``python -m app.rag.ingest`` (Step 12).

(Re)ingests the configured knowledge directory into the ChromaDB
vector store and prints what happened: documents found / created /
updated / skipped, chunks created, total chunks, and the store
location. Idempotent — run it as often as the documents change.
"""

from __future__ import annotations

from .service import get_rag_service


def main() -> None:
    """Ingest knowledge documents and print the report."""
    service = get_rag_service()
    report = service.ingest()
    print(f"Knowledge directory: {service.knowledge_dir}")
    print(f"Vector store: {report.store_dir}")
    print(f"Documents found: {report.documents_found}")
    print(
        f"Documents created: {report.documents_created} | "
        f"updated: {report.documents_updated} | "
        f"unchanged (skipped): {report.documents_skipped}"
    )
    print(f"Chunks created: {report.chunks_created}")
    print(f"Total chunks in store: {report.total_chunks}")
    if report.sources:
        print("Sources: " + ", ".join(report.sources))


if __name__ == "__main__":
    main()
