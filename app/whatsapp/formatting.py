"""Deterministic reply formatting for WhatsApp.

The agent replies in Markdown (GFM tables, ``**bold**``, bullets) —
the dashboard and chat render that natively, WhatsApp does not. These
pure functions convert a reply into plain WhatsApp-friendly text and
split anything longer than Meta's 4096-character text limit into
deterministic chunks, without changing the content's meaning.
"""

from __future__ import annotations

import re

from app.whatsapp.config import MAX_TEXT_LENGTH

__all__ = ["chunk_text", "format_reply_for_whatsapp"]

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(
        _SEPARATOR_CELL_RE.match(cell) for cell in cells if cell != ""
    )


def _format_table_row(cells: list[str], headers: list[str] | None) -> str:
    """Render one table row as 'Header: value' pairs (or a bare list)."""
    if headers:
        pairs = [
            f"{header}: {cell}"
            for header, cell in zip(headers, cells)
        ]
        return " · ".join(pairs)
    return " · ".join(cells)


def format_reply_for_whatsapp(text: str) -> str:
    """Convert an agent reply into WhatsApp-friendly plain text.

    Deterministic and content-preserving: GFM tables become
    ``Header: value · Header: value`` lines, ``**bold**`` becomes
    WhatsApp's ``*bold*``, headings lose their ``#`` markers, bullets
    become ``•``, and Markdown links open to ``text (url)``. Nothing
    else is rewritten.
    """
    formatted: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        if not table_lines:
            return
        rows = [_split_table_row(line) for line in table_lines]
        headers: list[str] | None = None
        if len(rows) >= 2 and _is_separator_row(rows[1]):
            headers = rows[0]
            rows = rows[2:]
        for cells in rows:
            formatted.append(_format_table_row(cells, headers))
        table_lines.clear()

    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line):
            table_lines.append(line)
            continue
        flush_table()
        heading = _HEADING_RE.match(line)
        if heading:
            line = heading.group(1)
        bullet = _BULLET_RE.match(line)
        if bullet:
            line = f"• {bullet.group(1)}"
        formatted.append(line)
    flush_table()

    joined = "\n".join(formatted)
    joined = _LINK_RE.sub(r"\1 (\2)", joined)
    joined = _BOLD_RE.sub(r"*\1*", joined)
    joined = _ITALIC_UNDERSCORE_RE.sub(r"_\1_", joined)
    return joined


def chunk_text(text: str, limit: int = MAX_TEXT_LENGTH) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Deterministic and lossless: prefer paragraph boundaries, then line
    boundaries, then hard slices. The concatenation of the chunks is
    always exactly the original text.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        take = _cut_at_boundary(remaining, limit)
        chunks.append(take)
        remaining = remaining[len(take):]
    return chunks


def _cut_at_boundary(text: str, limit: int) -> str:
    """Return the next chunk: the latest boundary at/before ``limit``."""
    window = text[:limit]
    for separator in ("\n\n", "\n"):
        cut = window.rfind(separator)
        if cut != -1:
            return text[: cut + len(separator)]
    return window
