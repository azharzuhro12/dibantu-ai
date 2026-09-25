"""Deterministic Markdown-aware chunker for knowledge documents (Step 12).

Documents are split by ``#``-headings into sections; each section's
paragraphs are packed greedily into chunks of at most ``max_chars``
characters, with a trailing ``overlap_chars``-character tail carried
into the next chunk so passages that straddle a boundary stay
retrievable. The whole pipeline is pure text processing: identical
input always yields identical chunks, ids included, on every run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "KnowledgeChunk",
    "chunk_document",
]

#: Target maximum chunk length in characters (~200 tokens, comfortably
#: inside all-MiniLM-L6-v2's 256-token window).
CHUNK_MAX_CHARS = 800

#: Characters of trailing context carried into the next chunk.
CHUNK_OVERLAP_CHARS = 150

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class KnowledgeChunk:
    """One retrievable passage cut from a knowledge document."""

    #: Passage text (may include heading-context sentences).
    content: str
    #: Document file name, e.g. ``refund_policy.md``.
    source: str
    #: Title of the ``#``/``##`` section the chunk was cut from, "" when
    #: the document opens with plain text before any heading.
    section: str
    #: Position of this chunk within its document (0-based).
    chunk_index: int
    #: Stable id: sha256 of source + section + index, hex, first 16 chars.
    chunk_id: str


def chunk_document(
    text: str,
    source: str,
    *,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[KnowledgeChunk]:
    """Split one Markdown document into retrievable chunks.

    ``source`` is the document's file name and is embedded in every
    chunk (and its id) so retrieval results can cite where a passage
    came from. Deterministic: the same document text and the same
    ``source`` always produce byte-identical chunks.
    """
    chunks: list[KnowledgeChunk] = []
    for section, paragraphs in _split_sections(text):
        pieces: list[str] = []
        for paragraph in paragraphs:
            pieces.extend(_split_long_text(paragraph, max_chars))
        buffer = ""
        for piece in pieces:
            candidate = f"{buffer}\n\n{piece}" if buffer else piece
            if not buffer or len(candidate) <= max_chars:
                buffer = candidate
                continue
            chunks.append(_make_chunk(buffer, source, section, len(chunks)))
            overlap = _tail_overlap(buffer, overlap_chars)
            if overlap and len(overlap) + len(piece) <= max_chars:
                buffer = f"{overlap}\n\n{piece}"
            else:
                buffer = piece
        if buffer.strip():
            chunks.append(_make_chunk(buffer, source, section, len(chunks)))
    return chunks


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a Markdown document into (section title, paragraphs) pairs.

    Headings open a new section; blank lines separate paragraphs inside
    it. Repeated section titles are kept as separate entries, in order.
    """
    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    title = ""
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        joined = " ".join(line.strip() for line in paragraph).strip()
        if joined:
            current.append(joined)
        paragraph.clear()

    def flush_section() -> None:
        flush_paragraph()
        if current:
            # Copy: `current` is reused for the next section.
            sections.append((title, list(current)))
        current.clear()

    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush_section()
            title = heading.group(2).strip()
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)
    flush_section()
    return sections


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split one paragraph into <= ``max_chars`` pieces on sentence
    boundaries; a single over-long sentence is hard-split on characters."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_RE.split(text):
        while len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > max_chars:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    return pieces


def _tail_overlap(buffer: str, overlap_chars: int) -> str:
    """Trailing ``overlap_chars`` of a closed chunk, cut at a word
    boundary so the next chunk starts mid-context but never mid-word."""
    if overlap_chars <= 0 or len(buffer) <= overlap_chars:
        return ""
    tail = buffer[-overlap_chars:]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def _make_chunk(
    content: str, source: str, section: str, index: int
) -> KnowledgeChunk:
    digest = hashlib.sha256(f"{source}|{section}|{index}".encode("utf-8"))
    return KnowledgeChunk(
        content=content.strip(),
        source=source,
        section=section,
        chunk_index=index,
        chunk_id=digest.hexdigest()[:16],
    )
