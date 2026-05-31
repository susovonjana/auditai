"""
Typo correction for retrieval queries.

Why this module exists:
  Users mistype questions all the time ("waht is tick mark?"). The bge
  embedding model is somewhat typo-tolerant, but vector + FTS retrieval
  still misses obvious matches because "waht" is a different token than
  "what". This module quietly corrects common typos BEFORE we embed and
  search — without touching what we send to Gemini (Gemini handles typos
  on its own and would otherwise see a "fixed" question that lost the
  user's original intent).

Design:
  - English only (Arabic skipped — pyspellchecker's Arabic dictionary
    is weak and would over-correct domain terms).
  - Domain whitelist: never "correct" audit acronyms or product nouns.
  - Conservative: only swap a word when there is exactly one obvious
    candidate at edit distance ≤ 2.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Words we MUST NOT "correct". Audit / accounting acronyms, product names,
# common abbreviations, currency codes, etc.
_VOCAB_WHITELIST: set[str] = {
    # Standards / regulators
    "isa", "isas", "gaap", "ifrs", "ias", "pcaob", "iaasb", "asb", "sec", "sox",
    "coso", "aicpa", "icaew", "acca",
    # Audit concepts as acronyms
    "coa", "tb", "bs", "is", "gl", "wp", "ppe", "nrv", "fv", "wacc", "ebitda",
    "ebit", "ebt", "roa", "roe", "dso", "dpo", "dio", "fy", "ytd", "ytm", "ar",
    "ap", "cogs", "cof", "kpi", "kpis", "cf",
    # Product / project
    "1audit", "auditai", "tickmark", "ticktmark",
    # Currencies & SI
    "usd", "eur", "gbp", "sar", "aed", "inr", "jpy", "cny",
    # Common ops words
    "ok", "ai", "ml", "url", "id", "ids", "pdf", "xlsx", "docx", "png", "jpg",
    "csv", "xml", "json", "api", "ui", "ux",
}


_spell = None  # lazy SpellChecker instance


def _get_speller():
    """Lazy-init the SpellChecker so import is cheap."""
    global _spell
    if _spell is not None:
        return _spell
    try:
        from spellchecker import SpellChecker
    except ImportError:
        logger.info(
            "pyspellchecker not installed — typo correction disabled. "
            "Run: pip install pyspellchecker"
        )
        return None
    sc = SpellChecker(distance=2)
    # Teach it our whitelist so they're never flagged as unknown
    sc.word_frequency.load_words(_VOCAB_WHITELIST)
    _spell = sc
    return sc


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # words, allow contractions


def correct_typos(text: str, language: str = "en") -> str:
    """
    Return a typo-corrected version of `text` suitable for retrieval.

    Conservative rules:
      - Only operates on English (language == "en"); other languages
        pass through unchanged.
      - Skips ALL-CAPS tokens (probably acronyms).
      - Skips tokens shorter than 3 chars.
      - Skips tokens that contain digits or punctuation.
      - Skips tokens in the whitelist.
      - Only substitutes when the spell checker has high confidence
        (single candidate, edit distance 1-2).
    """
    if not text or language != "en":
        return text or ""

    sc = _get_speller()
    if sc is None:
        return text

    # Find all word tokens with their offsets so we can replace in-place.
    out_parts: list[str] = []
    cursor = 0
    changed = False
    for match in _TOKEN_RE.finditer(text):
        word = match.group(0)
        start, end = match.span()
        out_parts.append(text[cursor:start])
        cursor = end

        replacement = word

        # Skip rules
        lower = word.lower()
        if (
            len(word) >= 3
            and word.isalpha()
            and not word.isupper()                 # acronyms
            and lower not in _VOCAB_WHITELIST
            and lower in sc.unknown([lower])       # unknown to dictionary
        ):
            candidate = sc.correction(lower)
            if (
                candidate
                and candidate != lower
                and candidate.isalpha()
                and abs(len(candidate) - len(lower)) <= 2
            ):
                # Preserve original casing pattern
                if word[0].isupper():
                    replacement = candidate.capitalize()
                else:
                    replacement = candidate
                changed = True

        out_parts.append(replacement)

    out_parts.append(text[cursor:])
    result = "".join(out_parts)
    if changed and result != text:
        logger.debug("Typo-corrected for retrieval: %r → %r", text, result)
    return result
