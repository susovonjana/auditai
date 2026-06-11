"""
Multi-query expansion for retrieval recall.

Why this module exists:
  Users often phrase questions in plain English ("who signs audits?") while the
  documents use precise vocabulary ("engagement partner sign-off under ISA
  700"). Vector + FTS still miss the right chunks because the query tokens
  don't overlap with the document tokens. We ask Gemini Flash to produce N
  vocabulary-diverse rephrasings, retrieve with all of them in parallel, fuse
  with RRF, and rerank against the ORIGINAL question. Reranking against the
  original is critical — expansions are only there to find candidates the user
  couldn't surface themselves, not to change what "relevant" means.

Design:
  - Returns [original, *rewrites]; on any failure returns [original] only,
    so expansion never blocks retrieval.
  - In-memory LRU cache keyed by (question, language) — expansions are
    deterministic enough that paraphrases of the same query hit cache.
  - Skips expansion entirely for very long questions (already specific) and
    when USE_QUERY_EXPANSION is false (kill-switch).
  - Reuses qa._get_gemini() so we don't reinstantiate the model.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from typing import List

from config import QUERY_EXPANSION_N, USE_QUERY_EXPANSION

logger = logging.getLogger(__name__)

# Skip expansion if the user already typed a long, specific question.
_MAX_QUESTION_CHARS = 300

# Hard timeout — never let expansion delay retrieval by more than this.
# Gemini Flash typically responds in 0.5-2 s on a host near a Google region;
# from a laptop / overseas Docker the round-trip can hit 3-5 s. 5 s is the
# default — bump via env if your deployment sees higher tail latency.
_EXPANSION_TIMEOUT_SEC = float(os.getenv("QUERY_EXPANSION_TIMEOUT_SEC", "5.0"))

# LRU cache (bounded) for expansions. 256 entries × ~4 small strings each
# is negligible memory, and saves a Gemini call per repeated/paraphrased ask.
_CACHE_MAX = 256
_cache: "OrderedDict[str, List[str]]" = OrderedDict()


def _cache_key(question: str, language: str) -> str:
    h = hashlib.sha256()
    h.update(question.strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update((language or "en").encode("utf-8"))
    return h.hexdigest()


def _cache_get(key: str) -> List[str] | None:
    if key in _cache:
        _cache.move_to_end(key)
        return list(_cache[key])
    return None


def _cache_put(key: str, value: List[str]) -> None:
    _cache[key] = list(value)
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _build_prompt(question: str, n: int, language: str) -> str:
    if language == "ar":
        return (
            f"أنت مساعد محترف في تدقيق الحسابات. أعد صياغة السؤال التالي بـ {n} "
            "صياغات بديلة تستخدم مفردات وصياغات مختلفة قد تظهر في وثائق التدقيق "
            "(IFRS, IAS, ISA, GAAP). أعد سطراً واحداً لكل صياغة، بدون ترقيم أو "
            "علامات اقتباس.\n\n"
            f"السؤال: {question}"
        )
    return (
        f"You are helping an audit professional search a knowledge base. "
        f"Generate exactly {n} alternative phrasings of the question below, "
        f"each using a DIFFERENT angle so retrieval matches more chunks:\n"
        f"  1. Lexical rewrite — same intent, different everyday vocabulary.\n"
        f"  2. Audit-terminology rewrite — use precise audit/accounting domain "
        f"terms (e.g. 'engagement partner', 'sign-off', 'working papers').\n"
        f"  3. Standards rewrite — phrase in terms of the relevant standards "
        f"(ISA, IFRS, GAAP, PCAOB) when they apply; otherwise generic.\n"
        f"  4. Abbreviation rewrite — expand acronyms or use them where the "
        f"original spelled them out.\n"
        f"Output rules: one rephrasing per line, no numbering, no quotes, no "
        f"prefixes. Skip any angle that doesn't apply to this question — "
        f"don't force it. Never repeat the original.\n\n"
        f"Question: {question}"
    )


def _parse_rewrites(text: str, n: int) -> List[str]:
    """Extract clean rewrites from the model's freeform output."""
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("\"'`").strip()
        # Strip leading numbering like "1.", "2)", "- "
        for prefix_len in (3, 2, 1):
            if line[:prefix_len] and line[0] in "0123456789-•*":
                stripped = line.lstrip("0123456789.)- •*").strip()
                if stripped:
                    line = stripped
                    break
        if not line or len(line) < 4:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= n:
            break
    return out


def _call_gemini_sync(prompt: str) -> str:
    """Sync call — imported lazily to avoid circular imports with qa.py."""
    from qa import _get_gemini  # local import: qa imports this module too

    model = _get_gemini()
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 256,
        },
    )
    return getattr(resp, "text", "") or ""


async def expand_query(
    question: str,
    language: str = "en",
    n: int = QUERY_EXPANSION_N,
) -> List[str]:
    """
    Return [original, *rewrites]. Falls back to [original] on any failure.
    """
    original = (question or "").strip()
    if not original:
        return []

    if not USE_QUERY_EXPANSION or n <= 0:
        return [original]

    if len(original) > _MAX_QUESTION_CHARS:
        logger.debug("Skipping expansion — question is already %d chars.", len(original))
        return [original]

    cache_key = _cache_key(original, language)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("Expansion cache hit for %r", original[:60])
        return [original, *cached]

    prompt = _build_prompt(original, n, language)
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_call_gemini_sync, prompt),
            timeout=_EXPANSION_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning("Query expansion timed out — falling back to original query only.")
        return [original]
    except Exception as exc:
        logger.warning("Query expansion failed (%s) — falling back to original query only.", exc)
        return [original]

    rewrites = _parse_rewrites(raw, n)
    if not rewrites:
        logger.debug("Expansion returned no usable rewrites; using original only.")
        return [original]

    _cache_put(cache_key, rewrites)
    logger.info(
        "Query expansion: %r → %d rewrites: %s",
        original[:80],
        len(rewrites),
        " | ".join(r[:80] for r in rewrites),
    )
    return [original, *rewrites]
