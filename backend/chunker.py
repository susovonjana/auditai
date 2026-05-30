"""
Text chunking.

  - `chunk_text(text)`     — legacy: splits a flat string into overlapping
                             ~1000-char chunks. Kept for backwards compat.

  - `chunk_blocks(blocks)` — new entry point used by the upload pipeline.
                             Respects ParsedBlock boundaries:
                                 * "table" blocks are NEVER split
                                 * "heading" blocks are merged into the
                                    section_heading of subsequent chunks
                                 * "text" blocks are split with overlap,
                                    metadata (page, section) propagated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP
from parser import ParsedBlock


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------
@dataclass
class ParsedChunk:
    content: str
    chunk_index: int
    chunk_type: str = "text"                     # "text" | "table"
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    char_count: int = 0


# ---------------------------------------------------------------------------
# Legacy text splitter (still used internally by chunk_blocks)
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[\.\!\?])\s+")


def _find_break_point(text: str, target: int) -> int:
    if target >= len(text):
        return len(text)
    window_start = max(0, target - 200)
    window = text[window_start : target + 1]

    nl_pos = window.rfind("\n\n")
    if nl_pos != -1:
        return window_start + nl_pos + 2
    nl_pos = window.rfind("\n")
    if nl_pos != -1:
        return window_start + nl_pos + 1
    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        return window_start + matches[-1].end()
    sp_pos = window.rfind(" ")
    if sp_pos != -1:
        return window_start + sp_pos + 1
    return target


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a single string into overlapping chunks of roughly chunk_size."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end_target = min(start + chunk_size, n)
        if end_target >= n:
            piece = text[start:n].strip()
            if piece:
                chunks.append(piece)
            break
        end = _find_break_point(text, end_target)
        if end <= start + 1:
            end = end_target
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


# ---------------------------------------------------------------------------
# Structured chunking — the new entry point
# ---------------------------------------------------------------------------
def chunk_blocks(blocks: List[ParsedBlock]) -> List[ParsedChunk]:
    """
    Convert a list of ParsedBlocks into ParsedChunks ready for embedding.

    Rules:
      - Table blocks are emitted as ONE chunk each (never split, no matter
        how large), so columns/rows stay together.
      - Heading blocks update the current section_heading and are NOT
        emitted as their own searchable chunks (avoids polluting retrieval
        with one-line "Chapter 4" hits).
      - Text blocks are split with overlap by chunk_text(); each resulting
        chunk inherits the page_number and section_heading of its parent
        block.
    """
    chunks: List[ParsedChunk] = []
    current_section: Optional[str] = None
    idx = 0

    for block in blocks:
        if block.block_type == "heading":
            current_section = block.section_heading or block.content
            continue

        section = block.section_heading or current_section

        if block.block_type == "table":
            content = (block.content or "").strip()
            if not content:
                continue
            chunks.append(
                ParsedChunk(
                    content=content,
                    chunk_index=idx,
                    chunk_type="table",
                    page_number=block.page_number,
                    section_heading=section,
                    char_count=len(content),
                )
            )
            idx += 1

        else:  # text
            for piece in chunk_text(block.content):
                chunks.append(
                    ParsedChunk(
                        content=piece,
                        chunk_index=idx,
                        chunk_type="text",
                        page_number=block.page_number,
                        section_heading=section,
                        char_count=len(piece),
                    )
                )
                idx += 1

    return chunks
