"""Tests for app.rag.chunker (Step 12): deterministic Markdown chunking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.chunker import (
    CHUNK_MAX_CHARS,
    KnowledgeChunk,
    chunk_document,
)

DOC = """# Judul Dokumen

Paragraf pembuka sebelum heading pertama.

## Bagian Satu

""" + "\n\n".join(
    f"Kalimat untuk bagian satu paragraf {i} dengan isi yang cukup "
    f"panjang supaya packing chunk benar-benar teruji di sini."
    for i in range(12)
) + """

## Bagian Dua

Isi bagian dua lebih pendek.
"""


def test_chunker_is_deterministic() -> None:
    first = chunk_document(DOC, "demo.md")
    second = chunk_document(DOC, "demo.md")
    assert first == second


def test_chunks_never_exceed_max_chars() -> None:
    chunks = chunk_document(DOC, "demo.md")
    assert chunks
    for chunk in chunks:
        assert len(chunk.content) <= CHUNK_MAX_CHARS


def test_chunks_carry_section_metadata() -> None:
    chunks = chunk_document(DOC, "demo.md")
    by_section = {chunk.section for chunk in chunks}
    assert "Judul Dokumen" in by_section
    assert "Bagian Satu" in by_section
    assert "Bagian Dua" in by_section


def test_text_before_any_heading_gets_empty_section() -> None:
    doc = "Teks pembuka tanpa heading.\n\n## Baru\n\nIsi bagian."
    by_section = {chunk.section for chunk in chunk_document(doc, "x.md")}
    assert by_section == {"", "Baru"}


def test_chunk_ids_are_unique_and_stable() -> None:
    first = chunk_document(DOC, "demo.md")
    second = chunk_document(DOC, "demo.md")
    ids = [chunk.chunk_id for chunk in first]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert ids == [chunk.chunk_id for chunk in second]
    # ids depend on the source: same text, different file -> other ids.
    other = chunk_document(DOC, "other.md")
    assert set(ids).isdisjoint({chunk.chunk_id for chunk in other})


def test_chunk_indices_are_sequential() -> None:
    chunks = chunk_document(DOC, "demo.md")
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    """A long section carries trailing context into its next chunk."""
    long_section = (
        "# Dokumen\n\n## Bagian\n\n"
        + "\n\n".join(
            f"Paragraf nomor {i} berisi kalimat unik penanda-{i} untuk "
            f"mengukur overlap antar chunk."
            for i in range(30)
        )
    )
    chunks = chunk_document(long_section, "overlap.md")
    assert len(chunks) >= 2
    shared = 0
    for previous, following in zip(chunks, chunks[1:]):
        tail_words = set(previous.content.split()[-12:])
        if tail_words & set(following.content.split()):
            shared += 1
    assert shared > 0, "neighbouring chunks should share trailing words"


def test_single_long_paragraph_is_split() -> None:
    paragraph = "Satu kalimat panjang. " * 200  # ~4600 chars, no paragraphs
    doc = f"# Bagian\n\n{paragraph}"
    chunks = chunk_document(doc, "long.md")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= CHUNK_MAX_CHARS


def test_source_is_recorded_on_every_chunk() -> None:
    for chunk in chunk_document(DOC, "demo.md"):
        assert isinstance(chunk, KnowledgeChunk)
        assert chunk.source == "demo.md"
