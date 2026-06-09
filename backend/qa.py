"""
Q&A Engine — FREE STACK (Google Gemini) + hybrid retrieval + reranker
+ conversation memory + streaming.

Pipeline per request:
  1. Embed the question locally (bge-small-en-v1.5) with the query prefix.
  2. Vector search (pgvector cosine)             → top N candidates
  3. Full-text search (PostgreSQL FTS, ts_rank)  → top N candidates
  4. Reciprocal Rank Fusion to combine the two lists
  5. Cross-encoder rerank the top 20 → keep top K
  6. Pull last 5 Q&A from this session for conversation memory
  7. Build a two-section structured prompt
  8. Call Gemini (stream or single-shot) → return Markdown answer
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional
from uuid import UUID

import re

from sqlalchemy import select, desc, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    INITIAL_CANDIDATES,
    TOP_K_CHUNKS,
    SIMILARITY_THRESHOLD,
    CONVERSATION_MEMORY_TURNS,
    USE_RERANKER,
)
from cache import answer_cache
from embeddings import embed_query
from models import DocumentChunk, Document, SearchHistory
from query_preprocessor import correct_typos
from reranker import rerank
import smalltalk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini client (lazy, configured once)
# ---------------------------------------------------------------------------
_gemini_ready: bool = False
_gemini_model = None


def _get_gemini():
    """Configure and return the Gemini GenerativeModel."""
    global _gemini_ready, _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )

    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if not _gemini_ready:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_ready = True

    _gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    return _gemini_model


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    document_filename: str
    content: str
    similarity: float        # vector similarity (0..1)
    fts_rank: float = 0.0    # FTS rank (raw)
    rerank_score: float = 0.0  # cross-encoder relevance score
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    chunk_type: str = "text"


@dataclass
class QAResult:
    answer: str
    was_answered: bool
    chunks_used: List[str] = field(default_factory=list)
    documents_referenced: List[str] = field(default_factory=list)
    document_filenames: List[str] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    question_embedding: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt — two-section answers with grounded vs supplemental content
# ---------------------------------------------------------------------------
# Phrases that strongly indicate "Gemini couldn't / didn't answer".
# We check ALL of these — Gemini phrases the same idea many ways and we
# need to catch them all so we can replace with the canonical message.
NO_ANSWER_MARKERS = {
    "en": [
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "i apologise",
        "unable to help",
        "unable to answer",
        "unable to provide",
        "cannot help",
        "can't help",
        "cannot answer",
        "can't answer",
        "do not have information",
        "don't have information",
        "do not have enough information",
        "don't have enough information",
        "no information about",
        "no specific answer",
        "knowledge base does not contain",
        "knowledge base doesn't contain",
        "provided knowledge base does not contain",
        "not contain information to answer",
        "not contain a specific answer",
    ],
    "ar": [
        "عذراً",
        "اعتذر",
        "أعتذر",
        "آسف",
        "اسف",
        "غير قادر",
        "لا تحتوي",
        "ليس لدي",
        "لا توجد معلومات",
        "لا أملك معلومات",
        "لا املك معلومات",
    ],
}


def looks_like_no_answer(text: str, language: str = "en") -> bool:
    """True if the response phrasing indicates no answer was produced."""
    if not text:
        return True
    lowered = text.lower()
    markers = NO_ANSWER_MARKERS.get(language) or NO_ANSWER_MARKERS["en"]
    if isinstance(markers, str):
        markers = [markers]
    return any(m.lower() in lowered for m in markers)


# Contact-us footer shown when no answer can be produced
SUPPORT_EMAIL = "info@1audit.com"
SUPPORT_PHONE = "+966 920 035 129"


def _language_instruction(language: str) -> str:
    """Instruction block appended to the user prompt to fix the response language."""
    if language == "ar":
        return (
            "IMPORTANT — RESPONSE LANGUAGE:\n"
            "Write the entire response in Arabic (العربية). "
            "Keep the two section markers exactly as '## From your knowledge base' and "
            "'## Follow-up Questions' in English (the UI needs them in English to parse), "
            "but ALL content inside both sections must be written in natural, fluent Arabic. "
            "If the source documents are in English, translate the relevant facts into Arabic "
            "for the answer. Numbers and dates remain in their original form. "
            "Follow-up questions must be in Arabic and must end with '؟' (the Arabic question mark). "
            "If the knowledge base does NOT contain an answer, write EXACTLY this Arabic message "
            "inside the 'From your knowledge base' section (replace ALL other content) and DO NOT "
            "produce a Follow-up Questions section at all:\n\n"
            f"عذراً، أنا غير قادر على مساعدتك في استفسارك.\n\n"
            f"يمكنك التواصل معنا مباشرة عبر البريد الإلكتروني **{SUPPORT_EMAIL}**\n"
            f"أو الهاتف: **{SUPPORT_PHONE}**\n\n"
        )
    return (
        "RESPONSE LANGUAGE: English.\n\n"
    )


SYSTEM_PROMPT = """You are AuditAI, a senior audit knowledge assistant for professional auditors. Your job is to give accurate, well-structured answers grounded strictly in the provided context.

Your response MUST use exactly these two sections in this order:

## From your knowledge base
- Use ONLY the provided KNOWLEDGE BASE CONTEXT to write this section.
- Start with a direct one-line answer to the question.
- Use `##` sub-headings to organize multi-part answers when helpful.
- Use **bold** for key terms, standard names (e.g., **ISA 315**), thresholds, and important figures.
- Reference source documents naturally: "According to ISA 315..." or "As stated in the uploaded compliance guidelines...".
- Keep paragraphs short — 2-3 sentences each.
- If the provided context does NOT contain a useful answer, write EXACTLY this (replace all other content in this section, do NOT add a Follow-up Questions section):

    I'm sorry. I'm unable to help you with your query.

    Feel free to email us directly at **info@1audit.com**

    Or call us at **+966 920 035 129**

## Follow-up Questions
Generate EXACTLY three short follow-up questions the user is likely to ask next, given the topic and what is available in the knowledge base context. Output them as a plain bullet list with `-`, one question per line, no extra commentary. Each question must:
- Be a natural next step or deeper dive on the same topic
- Be answerable using the same or related parts of the knowledge base (don't suggest topics that aren't there)
- Be 4-12 words long, phrased as a real question ending with `?`
- NOT repeat the user's current question
- NOT include numbering, prefixes like "Q:", or bold formatting

FORMATTING RULES (apply throughout):

1. **Markdown tables**: When presenting tabular data — financial figures, comparisons, ratios, classifications — you MUST use proper Markdown table syntax with pipes and a separator row. Right-align numeric columns using `---:` in the separator row. Never use tab-separated text. Never paste raw rows without pipes.

   Correct example:
   ```
   | Component | Basis | Amount (USD) |
   |---|---|---:|
   | Profit Before Tax | Audited PY adjusted | 3,000,000 |
   | Overall Materiality | 5% of PBT | 150,000 |
   ```

   Wrong (do NOT do this):
   ```
   Component    Basis    Amount
   Profit Before Tax  Audited  3,000,000
   ```

2. **Bullet lists** (`-`) for collections of items that have no inherent order:
   - Risks identified
   - Documents required
   - Account categories

3. **Numbered lists** (`1.` `2.` `3.`) for steps, sequences, or anything where order matters:
   1. Plan the audit
   2. Execute fieldwork
   3. Issue the opinion

4. **Task lists / checklists** (`- [ ]` for incomplete, `- [x]` for complete) when the answer represents items to verify, check, or complete:
   - [ ] Confirm bank balances
   - [x] Reviewed prior-year working papers
   - [ ] Test journal entries

5. **Bold** for thresholds, standard names, key figures (e.g., **$150,000**, **ISA 315**, **5% of PBT**).

6. **Inline code** (backticks) for account codes, formulas, or technical identifiers (e.g., `1000 Cash`, `=AVG(B2:B12)`).

Hard rules:
- Do NOT include an "Additional context" section. Do NOT add background, general knowledge, or supplementary explanations beyond what is in the provided context.
- NEVER fabricate facts. Everything in "From your knowledge base" must trace back to the provided context.
- NEVER include inline source citations such as [Source 1], [Source 6, Page 135; Source 7, Page 141], (Source 2), [Page 5], [Sources: 1, 2, 3], or any bracketed/parenthesised reference markers.
- If a previous Q&A in the conversation provides context for a follow-up ("what about...", "and the threshold?"), treat it as continuation of the same topic.
- Output must be valid Markdown. Be concise — aim for the shortest answer that is complete and accurate.
"""


# ---------------------------------------------------------------------------
# Post-processing: remove any inline source/page citations the model
# emits despite the instructions. Catches:
#   [Source 6, Page 135]              [Source 1]
#   [Source 6, Page 135; Source 7]   [Sources: 1, 2, 3]
#   [Page 135]                        [Pages 5-7]
#   (Source 2)                        (Page 4)
#   (Sources 1, 2)
# Plus trailing space/punctuation cleanup.
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(
    r"\s*[\[\(]\s*(?:Source|Sources|Page|Pages|Ref|References?|Excerpt|Excerpts)\b[^\]\)]*[\]\)]",
    flags=re.IGNORECASE,
)
_DANGLING_PUNCT_RE = re.compile(r"\s+([\.\,\;\:\!\?])")


def strip_inline_citations(text: str) -> str:
    """Remove inline source markers and tidy up leftover whitespace."""
    if not text:
        return text
    cleaned = _CITATION_RE.sub("", text)
    # Collapse "word , next" → "word, next"
    cleaned = _DANGLING_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
async def _vector_search(
    db: AsyncSession,
    question_embedding: List[float],
    limit: int,
) -> List[RetrievedChunk]:
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.content,
            DocumentChunk.page_number,
            DocumentChunk.section_heading,
            DocumentChunk.chunk_type,
            Document.filename,
            DocumentChunk.embedding.cosine_distance(question_embedding).label("dist"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == "active")
        .order_by("dist")
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        RetrievedChunk(
            chunk_id=r.id,
            document_id=r.document_id,
            document_filename=r.filename,
            content=r.content,
            similarity=float(1.0 - float(r.dist)),
            page_number=r.page_number,
            section_heading=r.section_heading,
            chunk_type=r.chunk_type or "text",
        )
        for r in rows
    ]

# ---------------------------------------------------------------------------
# Full-text search (BM25-ish via PostgreSQL FTS)
# ---------------------------------------------------------------------------
async def _fts_search(
    db: AsyncSession,
    question: str,
    limit: int,
) -> List[RetrievedChunk]:
    # plainto_tsquery is forgiving — it parses raw user text safely.
    sql = sql_text(
        """
        SELECT
            dc.id              AS id,
            dc.document_id     AS document_id,
            dc.content         AS content,
            dc.page_number     AS page_number,
            dc.section_heading AS section_heading,
            dc.chunk_type      AS chunk_type,
            d.filename         AS filename,
            ts_rank(to_tsvector('english', dc.content),
                    plainto_tsquery('english', :q)) AS rank
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        WHERE d.status = 'active'
          AND to_tsvector('english', dc.content) @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :limit
        """
    )
    rows = (await db.execute(sql, {"q": question, "limit": limit})).all()
    return [
        RetrievedChunk(
            chunk_id=r.id,
            document_id=r.document_id,
            document_filename=r.filename,
            content=r.content,
            similarity=0.0,
            fts_rank=float(r.rank),
            page_number=r.page_number,
            section_heading=r.section_heading,
            chunk_type=r.chunk_type or "text",
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(
    lists: List[List[RetrievedChunk]], k: int = 60
) -> List[RetrievedChunk]:
    """Combine multiple ranked lists. Higher RRF score = more relevant overall."""
    scores: dict[UUID, float] = {}
    seen: dict[UUID, RetrievedChunk] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            # keep the richest version of the chunk for downstream display
            if item.chunk_id not in seen:
                seen[item.chunk_id] = item
            else:
                existing = seen[item.chunk_id]
                if item.similarity > existing.similarity:
                    existing.similarity = item.similarity
                if item.fts_rank > existing.fts_rank:
                    existing.fts_rank = item.fts_rank
    fused = list(seen.values())
    fused.sort(key=lambda c: scores[c.chunk_id], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Reranker pass
# ---------------------------------------------------------------------------
async def _rerank_chunks(
    question: str, chunks: List[RetrievedChunk]
) -> List[RetrievedChunk]:
    if not chunks or not USE_RERANKER:
        return chunks
    try:
        scores = await rerank(question, [c.content for c in chunks])
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    except Exception as exc:
        logger.warning("Reranker failed (%s) — falling back to fused order.", exc)
    return chunks


# ---------------------------------------------------------------------------
# Hybrid retrieval entry point
# ---------------------------------------------------------------------------
async def _safe_fts_search(
    db: AsyncSession, question: str, limit: int
) -> List[RetrievedChunk]:
    """Wrapper that swallows errors so asyncio.gather can run alongside vector."""
    try:
        return await _fts_search(db, question, limit)
    except Exception as exc:
        logger.warning("FTS search failed (%s) — using vector results only.", exc)
        return []


async def retrieve_chunks(
    db: AsyncSession,
    question: str,
    question_embedding: List[float],
    top_k: int = TOP_K_CHUNKS,
) -> List[RetrievedChunk]:
    # Run vector + FTS in parallel — they're independent and similar speed,
    # so this saves ~80-100 ms per question.
    vector_hits, fts_hits = await asyncio.gather(
        _vector_search(db, question_embedding, INITIAL_CANDIDATES),
        _safe_fts_search(db, question, INITIAL_CANDIDATES),
    )

    fused = _reciprocal_rank_fusion([vector_hits, fts_hits])
    if not fused:
        return []

    fused = fused[:INITIAL_CANDIDATES]
    reranked = await _rerank_chunks(question, fused)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
async def load_recent_history(
    db: AsyncSession,
    session_id: UUID,
    turns: int = CONVERSATION_MEMORY_TURNS,
) -> List[SearchHistory]:
    """Return the last N Q&A turns for this session, oldest first."""
    rows = (
        await db.execute(
            select(SearchHistory)
            .where(SearchHistory.session_id == session_id)
            .order_by(desc(SearchHistory.asked_at))
            .limit(turns)
        )
    ).scalars().all()
    return list(reversed(rows))


def _format_history(history: List[SearchHistory]) -> str:
    if not history:
        return ""
    blocks = []
    for h in history:
        # Trim to keep prompt tight
        a = (h.ai_answer or "").strip()
        if len(a) > 800:
            a = a[:800] + " …"
        blocks.append(f"Q: {h.question}\nA: {a}")
    return "PREVIOUS CONVERSATION (most recent last):\n" + "\n\n".join(blocks) + "\n\n"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_user_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    history: List[SearchHistory],
    language: str = "en",
) -> str:
    # Use plain separators (no brackets) so the model is not tempted to
    # echo "[Source N]" style references back into its answer.
    context_blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        loc_parts = []
        if c.page_number:
            loc_parts.append(f"page {c.page_number}")
        if c.section_heading:
            loc_parts.append(f"section: {c.section_heading}")
        if c.chunk_type == "table":
            loc_parts.append("table")
        loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
        context_blocks.append(
            f"--- Excerpt {i} from {c.document_filename}{loc} ---\n{c.content}"
        )
    context_text = (
        "\n\n".join(context_blocks) if context_blocks else "(no context available)"
    )

    return (
        f"{_language_instruction(language)}"
        f"{_format_history(history)}"
        "Answer the question below using ONLY the KNOWLEDGE BASE CONTEXT. "
        "Output exactly two sections: '## From your knowledge base' and "
        "'## Follow-up Questions' (3 bullets). Do not add any other sections.\n\n"
        "=== KNOWLEDGE BASE CONTEXT ===\n"
        f"{context_text}\n"
        "=== END CONTEXT ===\n\n"
        f"Current question: {question}"
    )


# ---------------------------------------------------------------------------
# Constants for "no answer" handling
# ---------------------------------------------------------------------------
EMPTY_KB_TEXT = (
    "## From your knowledge base\n"
    "I'm sorry. I'm unable to help you with your query.\n\n"
    f"Feel free to email us directly at **{SUPPORT_EMAIL}**\n\n"
    f"Or call us at **{SUPPORT_PHONE}**"
)

_EMPTY_KB_TEXT_AR = (
    "## From your knowledge base\n"
    "عذراً، أنا غير قادر على مساعدتك في استفسارك.\n\n"
    f"يمكنك التواصل معنا مباشرة عبر البريد الإلكتروني **{SUPPORT_EMAIL}**\n\n"
    f"أو الهاتف: **{SUPPORT_PHONE}**"
)


def _empty_kb_text(language: str) -> str:
    return _EMPTY_KB_TEXT_AR if language == "ar" else EMPTY_KB_TEXT


# ---------------------------------------------------------------------------
# Single-shot answer (non-streaming) — kept for /ask compatibility
# ---------------------------------------------------------------------------
def _call_gemini_sync(prompt: str) -> str:
    model = _get_gemini()
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 1000},
    )
    try:
        return (response.text or "").strip()
    except Exception:
        parts = []
        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(cand.content, "parts", []) or []:
                if getattr(part, "text", None):
                    parts.append(part.text)
        return "\n".join(parts).strip()


async def answer_question(
    db: AsyncSession,
    session_id: UUID,
    question: str,
    language: str = "en",
) -> QAResult:
    """Non-streaming end-to-end Q&A. Returns the full QAResult."""
    # --- Small-talk shortcut: skip retrieval and LLM entirely ---
    convo_category = smalltalk.detect(question)
    if convo_category:
        return QAResult(
            answer=smalltalk.reply_for(convo_category, language=language),
            was_answered=True,
        )

    # --- Cache shortcut: same question + same language asked recently? ---
    cached = answer_cache.get(question, language)
    if cached is not None:
        logger.info("Cache hit for question (lang=%s).", language)
        return QAResult(
            answer=cached["answer"],
            was_answered=cached["was_answered"],
            chunks_used=cached.get("chunks_used", []),
            documents_referenced=cached.get("documents_referenced", []),
            document_filenames=cached.get("document_filenames", []),
            similarity_scores=cached.get("similarity_scores", []),
        )

    # Typo-correct the retrieval query (does NOT affect what Gemini sees)
    retrieval_query = correct_typos(question, language)

    question_embedding = await embed_query(retrieval_query)
    chunks = await retrieve_chunks(db, retrieval_query, question_embedding, TOP_K_CHUNKS)
    history = await load_recent_history(db, session_id)

    if not chunks:
        return QAResult(
            answer=_empty_kb_text(language),
            was_answered=False,
            question_embedding=question_embedding,
        )

    top_similarity = max((c.similarity for c in chunks), default=0.0)
    kb_has_signal = top_similarity >= SIMILARITY_THRESHOLD

    prompt = _build_user_prompt(question, chunks, history, language=language)
    try:
        answer_text = await asyncio.to_thread(_call_gemini_sync, prompt)
        if not answer_text:
            answer_text = _empty_kb_text(language)
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise

    # Strip any inline source citations the model may have produced.
    answer_text = strip_inline_citations(answer_text)

    # Detect no-answer phrasings and FORCE the canonical contact-us message.
    # This guarantees the user always sees the same polished message.
    if looks_like_no_answer(answer_text, language) or not kb_has_signal:
        answer_text = _empty_kb_text(language)
        was_answered = False
    else:
        was_answered = True

    result = QAResult(
        answer=answer_text,
        was_answered=was_answered,
        chunks_used=[str(c.chunk_id) for c in chunks],
        documents_referenced=list({str(c.document_id) for c in chunks}),
        document_filenames=list({c.document_filename for c in chunks}),
        similarity_scores=[round(c.similarity, 4) for c in chunks],
        question_embedding=question_embedding,
    )

    # Only cache positive answers — we want unanswered questions to
    # re-try in case the admin has added documents in the meantime.
    if was_answered:
        answer_cache.set(
            question,
            language,
            {
                "answer": result.answer,
                "was_answered": result.was_answered,
                "chunks_used": result.chunks_used,
                "documents_referenced": result.documents_referenced,
                "document_filenames": result.document_filenames,
                "similarity_scores": result.similarity_scores,
            },
        )
    return result


# ---------------------------------------------------------------------------
# Streaming answer
# ---------------------------------------------------------------------------
@dataclass
class StreamPreamble:
    """Metadata emitted before the first answer token starts streaming."""
    question_embedding: List[float]
    chunks: List[RetrievedChunk]
    history: List[SearchHistory]
    was_answered_initial: bool  # based on retrieval quality, before LLM
    smalltalk_reply: Optional[str] = None  # set if the question was small-talk
    language: str = "en"
    cached_reply: Optional[str] = None     # set if we have a cached answer
    cached_documents: List[str] = field(default_factory=list)
    cached_doc_ids: List[str] = field(default_factory=list)
    cached_chunks_used: List[str] = field(default_factory=list)
    cached_similarity: List[float] = field(default_factory=list)
    question: str = ""


async def prepare_stream(
    db: AsyncSession,
    session_id: UUID,
    question: str,
    language: str = "en",
) -> StreamPreamble:
    # --- Small-talk shortcut ---
    convo_category = smalltalk.detect(question)
    if convo_category:
        return StreamPreamble(
            question_embedding=[],
            chunks=[],
            history=[],
            was_answered_initial=True,
            smalltalk_reply=smalltalk.reply_for(convo_category, language=language),
            language=language,
            question=question,
        )

    # --- Cache shortcut: same question + same language asked recently? ---
    cached = answer_cache.get(question, language)
    if cached is not None:
        logger.info("Cache hit for streamed question (lang=%s).", language)
        return StreamPreamble(
            question_embedding=[],
            chunks=[],
            history=[],
            was_answered_initial=cached.get("was_answered", True),
            language=language,
            cached_reply=cached["answer"],
            cached_documents=cached.get("document_filenames", []),
            cached_doc_ids=cached.get("documents_referenced", []),
            cached_chunks_used=cached.get("chunks_used", []),
            cached_similarity=cached.get("similarity_scores", []),
            question=question,
        )

    # Typo-correct the retrieval query (Gemini still sees the original)
    retrieval_query = correct_typos(question, language)
    question_embedding = await embed_query(retrieval_query)
    chunks = await retrieve_chunks(db, retrieval_query, question_embedding, TOP_K_CHUNKS)
    history = await load_recent_history(db, session_id)
    top_similarity = max((c.similarity for c in chunks), default=0.0)
    return StreamPreamble(
        question_embedding=question_embedding,
        chunks=chunks,
        history=history,
        was_answered_initial=top_similarity >= SIMILARITY_THRESHOLD,
        language=language,
        question=question,
    )


def _stream_gemini_sync(prompt: str):
    """Yields successive text chunks from Gemini as they arrive."""
    model = _get_gemini()
    response_stream = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 1000},
        stream=True,
    )
    for ev in response_stream:
        try:
            t = ev.text
        except Exception:
            t = ""
            for cand in getattr(ev, "candidates", []) or []:
                for part in getattr(cand.content, "parts", []) or []:
                    if getattr(part, "text", None):
                        t += part.text
        if t:
            yield t


async def stream_answer(
    preamble: StreamPreamble,
    question: str,
) -> AsyncIterator[str]:
    """
    Async generator that yields successive text chunks from the LLM,
    stripping any inline source/page citations as they pass through.

    If the question was small-talk, yields the canned reply word-by-word.
    If the knowledge base is empty, yields the EMPTY_KB_TEXT in one chunk.
    """
    # --- Small-talk shortcut: stream the canned reply, no LLM call ---
    if preamble.smalltalk_reply:
        words = preamble.smalltalk_reply.split(" ")
        for i, w in enumerate(words):
            chunk = (w + " ") if i < len(words) - 1 else w
            yield chunk
            await asyncio.sleep(0.015)
        return

    # --- Cache hit: stream the cached answer back, no Gemini call ---
    if preamble.cached_reply:
        # Emit in ~80-char chunks so the UI updates smoothly without
        # making the cached response feel artificially slow.
        text = preamble.cached_reply
        CHUNK_SIZE = 80
        for i in range(0, len(text), CHUNK_SIZE):
            yield text[i : i + CHUNK_SIZE]
            await asyncio.sleep(0.005)
        return

    if not preamble.chunks:
        yield _empty_kb_text(preamble.language)
        return

    prompt = _build_user_prompt(
        question, preamble.chunks, preamble.history, language=preamble.language,
    )

    # Run the sync generator in a worker thread; pipe pieces back via a queue
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def producer():
        try:
            for piece in _stream_gemini_sync(prompt):
                asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
        except Exception as exc:
            logger.error("Gemini streaming error: %s", exc)
            asyncio.run_coroutine_threadsafe(queue.put(f"\n\n_[error: {exc}]_"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    asyncio.create_task(asyncio.to_thread(producer))

    # ---- Probe-and-decide buffer ----
    # Buffer the first ~200 chars from Gemini BEFORE yielding anything,
    # so we can detect "I'm sorry, I can't help" responses and swap them
    # for the canonical contact-us message — without the user briefly
    # seeing the wrong text. Adds ~200-400 ms of initial delay only.
    PROBE_CHARS = 200
    probe = []
    probe_len = 0
    decided = False

    # Streaming citation filter buffer (after we've decided)
    buf = ""
    LOOKBEHIND = 200

    while True:
        piece = await queue.get()
        if piece is None:
            break

        if not decided:
            probe.append(piece)
            probe_len += len(piece)
            if probe_len < PROBE_CHARS:
                continue
            # We have enough — decide
            decided = True
            combined = "".join(probe)
            if looks_like_no_answer(combined, preamble.language):
                # Replace with canonical, drain producer, stop.
                yield _empty_kb_text(preamble.language)
                while True:
                    p = await queue.get()
                    if p is None:
                        break
                return
            # Not no-answer — flush probe into the citation filter
            buf = combined
            continue

        buf += piece
        if "]" in piece or ")" in piece or len(buf) > LOOKBEHIND * 4:
            cleaned = strip_inline_citations(buf)
            if len(cleaned) > LOOKBEHIND:
                emit = cleaned[:-LOOKBEHIND]
                buf = cleaned[-LOOKBEHIND:]
                if emit:
                    yield emit
            else:
                buf = cleaned

    # Stream ended before we decided (very short response)
    if not decided:
        combined = "".join(probe)
        if looks_like_no_answer(combined, preamble.language):
            yield _empty_kb_text(preamble.language)
            return
        buf = combined

    # Final flush: scrub any remaining tail
    final = strip_inline_citations(buf)
    if final:
        yield final
