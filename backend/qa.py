"""
Q&A Engine — FREE STACK (Google Gemini).

Pipeline (per project brief, Section 7):
  1. Embed the user's question locally (sentence-transformers).
  2. pgvector cosine-similarity search on document_chunks.embedding
     to retrieve the top K most semantically relevant chunks.
  3. Build a strict-Markdown prompt with the chunks as context.
  4. Call Google Gemini to generate the answer.
  5. Return the answer + retrieval metadata to the caller.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_CHUNKS,
    SIMILARITY_THRESHOLD,
)
from embeddings import embed_text
from models import DocumentChunk, Document

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
    similarity: float  # 0..1, higher is more similar


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
# Formatting instructions sent with every prompt
# (Per project brief, Section 10)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are AuditAI, a senior audit knowledge assistant. You answer questions from auditors using ONLY the context provided to you from the firm's knowledge base. You must never guess, fabricate information, or use knowledge from outside the provided context.

You MUST follow these formatting rules on every response:

1. Start with a direct, one-line answer to the question.
2. Use `##` Markdown headings to organize multi-part answers into clear sections.
3. Use `**bold**` for all key terms, standard names (e.g., **ISA 315**), thresholds, and important figures.
4. Use bullet lists (`-`) for collections of items.
5. Use numbered lists (`1.`) for steps or sequences.
6. Keep paragraphs short — maximum 3-4 sentences each.
7. Reference source documents naturally in the text: "According to ISA 315..." or "As stated in the uploaded compliance guidelines...".
8. End complex answers with a `## Key Takeaway` section summarizing the most important point.
9. When the answer is NOT in the provided context, respond EXACTLY with: "Based on the documents currently in the knowledge base, I was unable to find a specific answer to this question. You may want to consult [relevant authority] or ask your administrator to upload the relevant documentation."
10. Never guess. Never fabricate. Never use knowledge outside the uploaded documents."""


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------
async def retrieve_chunks(
    db: AsyncSession,
    question_embedding: List[float],
    top_k: int = TOP_K_CHUNKS,
) -> List[RetrievedChunk]:
    """
    Cosine-similarity search using pgvector's `<=>` operator.
    `<=>` is cosine distance (0 = identical), so similarity = 1 - distance.
    """
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
        .limit(top_k)
    )
    result = await db.execute(stmt)
    rows = result.all()

    out: List[RetrievedChunk] = []
    for row in rows:
        out.append(
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                document_filename=row.filename,
                content=row.content,
                similarity=float(1.0 - float(row.dist)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_user_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Assemble the user-turn prompt with context blocks + the question."""
    context_blocks: List[str] = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[Source {i} — {c.document_filename}]\n{c.content}")
    context_text = (
        "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context available)"
    )

    return (
        "Below is the relevant context retrieved from the firm's audit knowledge base. "
        "Use ONLY this context to answer the question that follows.\n\n"
        "=== KNOWLEDGE BASE CONTEXT ===\n"
        f"{context_text}\n"
        "=== END CONTEXT ===\n\n"
        f"Question: {question}\n\n"
        "Remember: follow ALL formatting rules. If the answer is not present in the "
        "context above, use the exact fallback sentence specified in your instructions."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
NO_ANSWER_TEXT = (
    "Based on the documents currently in the knowledge base, I was unable to "
    "find a specific answer to this question. You may want to consult the "
    "relevant audit authority or ask your administrator to upload the relevant "
    "documentation."
)

EMPTY_KB_TEXT = (
    "The knowledge base is currently empty. Please contact your administrator."
)


def _call_gemini_sync(prompt: str) -> str:
    """Synchronous Gemini call — wrap in asyncio.to_thread from async code."""
    model = _get_gemini()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1500,
        },
    )
    # response.text raises if the response was blocked; guard it
    try:
        return (response.text or "").strip()
    except Exception:
        # Try to assemble manually from candidates
        try:
            parts = []
            for cand in getattr(response, "candidates", []) or []:
                for part in getattr(cand.content, "parts", []) or []:
                    if getattr(part, "text", None):
                        parts.append(part.text)
            return "\n".join(parts).strip()
        except Exception:
            return ""


async def answer_question(
    db: AsyncSession,
    question: str,
) -> QAResult:
    """End-to-end Q&A. Returns a QAResult; caller persists search_history."""

    # Step 1: embed question locally (free, fast)
    question_embedding = await embed_text(question)

    # Step 2: vector search
    chunks = await retrieve_chunks(db, question_embedding, top_k=TOP_K_CHUNKS)

    # If no chunks at all → KB is effectively empty for the active set
    if not chunks:
        return QAResult(
            answer=EMPTY_KB_TEXT,
            was_answered=False,
            question_embedding=question_embedding,
        )

    # Decide whether the top result is similar enough to be useful.
    top_similarity = chunks[0].similarity
    kb_has_signal = top_similarity >= SIMILARITY_THRESHOLD

    # Step 3 + 4: call Gemini
    user_prompt = _build_user_prompt(question, chunks)

    try:
        answer_text = await asyncio.to_thread(_call_gemini_sync, user_prompt)
        if not answer_text:
            answer_text = NO_ANSWER_TEXT
    except RuntimeError:
        # configuration / missing key — re-raise to the router
        raise
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise

    was_answered = kb_has_signal and not _looks_like_no_answer(answer_text)

    return QAResult(
        answer=answer_text,
        was_answered=was_answered,
        chunks_used=[str(c.chunk_id) for c in chunks],
        documents_referenced=list({str(c.document_id) for c in chunks}),
        document_filenames=list({c.document_filename for c in chunks}),
        similarity_scores=[round(c.similarity, 4) for c in chunks],
        question_embedding=question_embedding,
    )


def _looks_like_no_answer(text: str) -> bool:
    """Heuristic — did the model produce the fallback phrasing?"""
    lowered = text.lower()
    markers = [
        "unable to find a specific answer",
        "knowledge base is currently empty",
        "could not find an answer",
    ]
    return any(m in lowered for m in markers)
