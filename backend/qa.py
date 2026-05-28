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
from embeddings import embed_query
from models import DocumentChunk, Document, SearchHistory
from reranker import rerank

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
SYSTEM_PROMPT = """You are AuditAI, a senior audit knowledge assistant for professional auditors. Your job is to give answers that are accurate, well-structured, AND educational.

You always answer in TWO clearly labelled sections so the user can tell what came from the firm's documents vs your general expert knowledge:

## From your knowledge base
- Use ONLY the provided context to write this section.
- Start with a direct one-line answer.
- Use `##` sub-headings to organize multi-part answers when helpful.
- Use **bold** for key terms, standard names (e.g., **ISA 315**), thresholds, and important figures.
- Use bullet lists (`-`) for collections and numbered lists (`1.`) for sequences.
- Reference source documents naturally: "According to ISA 315..." or "As stated in the uploaded compliance guidelines...".
- Keep paragraphs short — 3-4 sentences each.
- If the provided context does not contain a useful answer, write exactly: "The knowledge base does not contain a specific answer to this question."

## Additional context
- Add your own expert audit knowledge to make the answer richer, more natural, and more useful.
- Provide background, real-world examples, related standards, common pitfalls, and how to apply the concept in practice.
- Be honest that this is supplementary background; never present these statements as if they came from the firm's documents.
- 2-4 short paragraphs is ideal. Use sub-bullets if helpful.
- Stay professional, accurate, and aligned with mainstream audit practice (ISA / IAASB / ASB / PCAOB as applicable).

## Key Takeaway
- One short paragraph synthesising the single most important point the auditor should walk away with.

Hard rules:
- NEVER put fabricated or invented facts in the "From your knowledge base" section. That section is strictly grounded in the provided context.
- NEVER refuse to answer when the documents are silent — provide an "Additional context" section with general background instead, and clearly state that the knowledge base does not address the question.
- If a previous Q&A in the conversation provides context for a follow-up ("what about...", "and the threshold?"), treat it as continuation of the same topic.
- Output must be valid Markdown.
"""


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
            dc.id            AS id,
            dc.document_id   AS document_id,
            dc.content       AS content,
            d.filename       AS filename,
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
async def retrieve_chunks(
    db: AsyncSession,
    question: str,
    question_embedding: List[float],
    top_k: int = TOP_K_CHUNKS,
) -> List[RetrievedChunk]:
    vector_hits = await _vector_search(db, question_embedding, INITIAL_CANDIDATES)
    try:
        fts_hits = await _fts_search(db, question, INITIAL_CANDIDATES)
    except Exception as exc:
        # If the FTS index is missing or query is unusual, fall back gracefully.
        logger.warning("FTS search failed (%s) — using vector results only.", exc)
        fts_hits = []

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
) -> str:
    context_blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[Source {i} — {c.document_filename}]\n{c.content}")
    context_text = (
        "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context available)"
    )

    return (
        f"{_format_history(history)}"
        "Use the KNOWLEDGE BASE CONTEXT below for the 'From your knowledge base' section. "
        "Use your own expert audit knowledge for the 'Additional context' section. "
        "Always include both sections plus 'Key Takeaway' as instructed.\n\n"
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
    "The knowledge base is currently empty. Please contact your administrator "
    "to upload audit standards, regulations, or internal policies.\n\n"
    "## Additional context\n"
    "Once documents are uploaded, AuditAI can answer questions about their "
    "contents and provide supplementary background from broader audit practice."
)


# ---------------------------------------------------------------------------
# Single-shot answer (non-streaming) — kept for /ask compatibility
# ---------------------------------------------------------------------------
def _call_gemini_sync(prompt: str) -> str:
    model = _get_gemini()
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 2000},
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
) -> QAResult:
    """Non-streaming end-to-end Q&A. Returns the full QAResult."""
    question_embedding = await embed_query(question)
    chunks = await retrieve_chunks(db, question, question_embedding, TOP_K_CHUNKS)
    history = await load_recent_history(db, session_id)

    if not chunks:
        return QAResult(
            answer=EMPTY_KB_TEXT,
            was_answered=False,
            question_embedding=question_embedding,
        )

    top_similarity = max((c.similarity for c in chunks), default=0.0)
    kb_has_signal = top_similarity >= SIMILARITY_THRESHOLD

    prompt = _build_user_prompt(question, chunks, history)
    try:
        answer_text = await asyncio.to_thread(_call_gemini_sync, prompt)
        if not answer_text:
            answer_text = EMPTY_KB_TEXT
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise

    was_answered = kb_has_signal and "knowledge base does not contain" not in answer_text.lower()

    return QAResult(
        answer=answer_text,
        was_answered=was_answered,
        chunks_used=[str(c.chunk_id) for c in chunks],
        documents_referenced=list({str(c.document_id) for c in chunks}),
        document_filenames=list({c.document_filename for c in chunks}),
        similarity_scores=[round(c.similarity, 4) for c in chunks],
        question_embedding=question_embedding,
    )


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


async def prepare_stream(
    db: AsyncSession,
    session_id: UUID,
    question: str,
) -> StreamPreamble:
    question_embedding = await embed_query(question)
    chunks = await retrieve_chunks(db, question, question_embedding, TOP_K_CHUNKS)
    history = await load_recent_history(db, session_id)
    top_similarity = max((c.similarity for c in chunks), default=0.0)
    return StreamPreamble(
        question_embedding=question_embedding,
        chunks=chunks,
        history=history,
        was_answered_initial=top_similarity >= SIMILARITY_THRESHOLD,
    )


def _stream_gemini_sync(prompt: str):
    """Yields successive text chunks from Gemini as they arrive."""
    model = _get_gemini()
    response_stream = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 2000},
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
    Async generator that yields successive text chunks from the LLM.

    If the knowledge base is empty, yields the EMPTY_KB_TEXT in one chunk.
    """
    if not preamble.chunks:
        yield EMPTY_KB_TEXT
        return

    prompt = _build_user_prompt(question, preamble.chunks, preamble.history)

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

    while True:
        piece = await queue.get()
        if piece is None:
            break
        yield piece
