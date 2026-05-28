"""
Text chunking with character-based overlap.

Default: ~1000 characters per chunk, ~150 characters of overlap,
splitting preferably at natural boundaries (newlines, sentences).
"""
from __future__ import annotations

import re
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP


# Boundary regex: prefer paragraph break > sentence end > space
_SENTENCE_END = re.compile(r"(?<=[\.\!\?])\s+")


def _find_break_point(text: str, target: int) -> int:
    """
    Look near the target position for a natural break (newline, sentence end,
    or whitespace). Falls back to the target itself if nothing nicer is found.
    """
    if target >= len(text):
        return len(text)

    # Look backwards up to 200 chars for a paragraph break
    window_start = max(0, target - 200)
    window = text[window_start : target + 1]

    # 1. Paragraph break
    nl_pos = window.rfind("\n\n")
    if nl_pos != -1:
        return window_start + nl_pos + 2

    # 2. Single newline
    nl_pos = window.rfind("\n")
    if nl_pos != -1:
        return window_start + nl_pos + 1

    # 3. Sentence end
    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        return window_start + matches[-1].end()

    # 4. Plain whitespace
    sp_pos = window.rfind(" ")
    if sp_pos != -1:
        return window_start + sp_pos + 1

    # Fallback: hard cut
    return target


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split `text` into overlapping chunks of approximately `chunk_size`
    characters, with `chunk_overlap` characters of overlap between
    consecutive chunks.
    """
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

        # On the last chunk, take everything to EOF
        if end_target >= n:
            chunk = text[start:n].strip()
            if chunk:
                chunks.append(chunk)
            break

        end = _find_break_point(text, end_target)
        # Guard: never produce a tiny / zero-length chunk
        if end <= start + 1:
            end = end_target

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start, leaving `chunk_overlap` of trailing context
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end  # prevent infinite loop
        start = next_start

    return chunks
