"""
Support-desk router — the AI assist behind the support system's ticket reply
box. Called SERVER-TO-SERVER by the support backend (support-be), never by a
browser, so there is no session token and no CORS involvement.

Endpoints:
  POST /support/assist   stream a plain-text answer / reply draft (NDJSON)

Authentication is a shared secret in the ``X-Support-Secret`` header, matching
``SUPPORT_SHARED_SECRET`` (fail closed: unset ⇒ 503, exactly like the internal
router). It is deliberately a different secret from INTERNAL_SHARED_SECRET so
the support desk cannot reach /internal/*.

The event format mirrors /copilot/write: newline-delimited JSON objects with a
leading {"type":"meta"}, a run of {"type":"delta","text":…}, an optional
terminal {"type":"error","message":…}, and a final {"type":"done", …} that
carries the FULL normalised text — the consumer treats that as the canonical
result and the deltas as a live preview only.

NOTE: No `from __future__ import annotations` here — slowapi's decorator
interferes with FastAPI's resolution of stringified forward references (same
reason as routers/user.py).
"""
import hmac
import json
import logging
import time
import uuid
from typing import List, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

import config
from database import AsyncSessionLocal
from embeddings import embed_query
from query_preprocessor import correct_typos
from rate_limit import limiter
import qa
import schemas
import structured
import usage_meter
from prompts import support as support_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/support", tags=["support"])

# The NDJSON error line for a question that could not be looked up at all.
# English on purpose: the support desk's dictionary is keyed by the English
# text and translates it for its Arabic UI; the `language` this request
# carries is the reply language, not the agent's UI language.
MANUAL_UNAVAILABLE_MESSAGE = "The help manual could not be searched right now. Please try again in a moment."

_STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


def _support_caller_key(request: Request) -> str:
    """Rate-limit key: the support user (sent by support-be), else the IP.

    Every agent reaches us through the support backend's single egress IP, so a
    per-IP bucket would make all agents share one allowance."""
    return request.headers.get("x-support-user") or get_remote_address(request)


async def _require_support_secret(x_support_secret: str = Header(default="")) -> None:
    """Fail closed: if no secret is configured, the support API is disabled."""
    # Read at request time (not import time) so a deployment can set the secret
    # without a code change being visible here, and so tests can monkeypatch it.
    secret = config.SUPPORT_SHARED_SECRET
    if not secret:
        raise HTTPException(status_code=503, detail="Support assist is not configured.")
    if not hmac.compare_digest(x_support_secret or "", secret):
        raise HTTPException(status_code=403, detail="Invalid support secret.")


def _ndjson(obj: dict) -> str:
    # ensure_ascii=False keeps Arabic readable in logs and on the wire; the
    # consumer decodes UTF-8 either way.
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _excerpts_from(chunks: List["qa.RetrievedChunk"], language: str) -> List[dict]:
    """Shape retrieved chunks for the prompt builder, resolving help URLs here
    so prompts/support.py stays free of a qa dependency."""
    out: List[dict] = []
    for chunk in chunks:
        loc_parts = []
        if chunk.page_number:
            loc_parts.append(f"page {chunk.page_number}")
        if chunk.section_heading:
            loc_parts.append(f"section: {chunk.section_heading}")
        loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
        out.append(
            {
                "content": chunk.content,
                "source": f"{chunk.document_filename}{loc}",
                "help_url": qa.build_help_url(chunk.help_url, language),
            }
        )
    return out


async def _retrieve(query: str, language: str) -> Tuple[List["qa.RetrievedChunk"], bool]:
    """Help-manual chunks for the query, as ``(chunks, available)``.

    ``available`` is False when the lookup itself failed (database down, host
    unresolvable, embedding error) — as opposed to a lookup that ran and found
    nothing, which is ``([], True)``. Ask mode needs the difference: "the
    manual does not cover this" and "the manual could not be searched" call
    for different answers.

    Uses its OWN short-lived session rather than a request-scoped one: a failed
    connection leaves a SQLAlchemy session in a state whose close() raises, and
    with a `Depends(get_db)` session that raise lands in the dependency
    teardown — turning a database blip into a 500 before the stream starts.
    Confined here, the worst case is an ungrounded answer and a warning.
    """
    try:
        retrieval_query = correct_typos(query, language)
        embedding = await embed_query(retrieval_query)
        async with AsyncSessionLocal() as session:
            pool = await qa.retrieve_chunks(
                session, retrieval_query, embedding, config.SUPPORT_ASSIST_TOP_K * 2, language=language
            )
        # The KB mixes the help manual with ISA standards and other uploads;
        # help-manual chunks are the ones that carry a help_url, so they go
        # first (stable sort keeps the rerank order within each group).
        pool = sorted(pool, key=lambda c: 0 if c.help_url else 1)
        return pool[: config.SUPPORT_ASSIST_TOP_K], True
    except Exception as exc:  # retrieval is best-effort
        logger.warning("support assist: retrieval failed (%s)", exc)
        return [], False


def _reply_query(conversation: List[dict], ticket: dict) -> str:
    """What to look up for a one-click reply: the customer's latest message,
    else the ticket's subject and opening description."""
    for message in reversed(conversation):
        text = (message.get("text") or "").strip()
        if message.get("role") == "customer" and text:
            return text[:500]
    subject = (ticket.get("subject") or "").strip()
    description = (ticket.get("description") or "").strip()[:300]
    return f"{subject} {description}".strip()


def _sources_from(chunks: List["qa.RetrievedChunk"], language: str, limit: int = 3) -> List[dict]:
    """Help-manual pages behind the answer, one per help URL, for the UI's
    source chips. Chunks without a help_url (ISA standards, other uploads)
    are not user-clickable and are left out."""
    seen = set()
    sources: List[dict] = []
    for chunk in chunks:
        url = qa.build_help_url(chunk.help_url, language)
        if not url or url in seen:
            continue
        seen.add(url)
        title = (chunk.section_heading or chunk.document_filename or "").strip()
        sources.append({"title": title, "url": url})
        if len(sources) >= limit:
            break
    return sources


@router.post("/assist")
@limiter.limit(config.RATE_LIMIT_SUPPORT_ASSIST, key_func=_support_caller_key)
async def support_assist(
    request: Request,
    payload: schemas.SupportAssistRequest,
    _: None = Depends(_require_support_secret),
):
    """Stream a plain-text answer (ask) or reply draft / translation (write)."""
    started = time.perf_counter()
    req_id = (payload.request_id or "").strip() or uuid.uuid4().hex
    mode = payload.mode
    language = payload.language
    translate = bool(payload.translate) and mode == "write"
    draft = (payload.draft or "").strip()
    instruction = (payload.instruction or "").strip()
    question = (payload.question or "").strip()
    ticket = payload.ticket.model_dump()
    conversation = [m.model_dump() for m in payload.conversation]
    kb_articles = [a.model_dump() for a in payload.kb_articles]

    if mode == "reply":
        # One click in the desk: no draft, no instruction. The built-in
        # instruction says what a good next reply is; it runs through the
        # write path below exactly like an agent-typed instruction would.
        draft = ""
        instruction = support_prompt.AUTO_REPLY_INSTRUCTION

    # Belt and braces: support-be validates the same rules before calling.
    if mode == "ask" and not question:
        raise HTTPException(status_code=422, detail="A question is required in ask mode.")
    if translate and not draft:
        raise HTTPException(status_code=422, detail="Translate needs a draft.")
    if mode == "write" and not translate and not draft and not instruction:
        raise HTTPException(status_code=422, detail="Write needs a draft or an instruction.")

    caller = f"support:{payload.caller_user_id}" if payload.caller_user_id else None

    # Retrieval grounds a question, and a reply written from scratch. A polish
    # or a translation of the agent's own text needs none.
    # Retrieval grounds a question, and a reply written from scratch — but only
    # when the manual covers this ticket's product (help_manual); an aninvoice
    # reply must not be grounded in the 1audit manual. A polish or a
    # translation of the agent's own text needs none.
    chunks: List["qa.RetrievedChunk"] = []
    retrieval = "skipped"
    # A polish — a draft with no instruction — may not add facts (system prompt),
    # so reference material cannot be used in it: drop the support articles
    # rather than pay for them. A draft WITH an instruction ("add the export
    # steps") keeps them.
    if mode == "write" and draft and not instruction and not translate:
        kb_articles = []
        # Likewise the thread: a polish keeps the draft's meaning, so the only
        # context it can use is the message being answered (names, tone).
        latest_customer = next((m for m in reversed(conversation) if m.get("role") == "customer"), None)
        conversation = [latest_customer] if latest_customer else []
    needs_retrieval = payload.help_manual and (mode == "ask" or (not translate and not draft))
    if needs_retrieval:
        if mode == "ask":
            query = question
        elif mode == "reply":
            query = _reply_query(conversation, ticket)
        else:
            query = payload.instruction or (ticket.get("subject") or "")
        if query.strip():
            chunks, available = await _retrieve(query, language)
            retrieval = "ok" if available else "unavailable"

    if mode == "ask" and retrieval == "unavailable":
        # A question is answered from the manual or not at all. Without the
        # lookup the model could only produce the "I couldn't find this" line,
        # which the agent would read as "the manual does not cover it" — not
        # true. Say what happened, spend no tokens, and let them retry. (A
        # reply written from scratch still goes ahead ungrounded: the ticket
        # and the instruction carry it.)
        logger.warning("support assist: help manual unavailable, ask aborted req=%s caller=%s", req_id, caller)

        async def unavailable_stream():
            yield _ndjson(
                {
                    "type": "meta",
                    "mode": mode,
                    "translate": False,
                    "language": language,
                    "sources": [],
                    "confidence": 0.0,
                    "retrieval": retrieval,
                }
            )
            yield _ndjson({"type": "error", "message": MANUAL_UNAVAILABLE_MESSAGE})

        return StreamingResponse(unavailable_stream(), media_type="application/x-ndjson", headers=_STREAM_HEADERS)

    confidence = qa._confidence_score(chunks)[0] if chunks else 0.0
    excerpts = _excerpts_from(chunks, language)
    sources = _sources_from(chunks, language) if mode == "ask" else []

    if mode == "ask":
        system = support_prompt.system_prompt_ask(language)
        user_prompt = support_prompt.build_ask_prompt(
            question=question,
            excerpts=excerpts,
            kb_articles=kb_articles,
            ticket=ticket,
            conversation=conversation,
            language=language,
        )
        temperature = 0.2
        feature = "support_ask"
    elif translate:
        system = support_prompt.system_prompt_translate(language)
        user_prompt = support_prompt.build_write_prompt(
            draft=draft,
            instruction=None,
            translate=True,
            excerpts=[],
            kb_articles=[],
            ticket=ticket,
            conversation=[],
            language=language,
        )
        temperature = 0.1
        feature = "support_translate"
    else:
        system = support_prompt.system_prompt_write(
            language, ticket.get("agent_name"), ticket.get("customer_name")
        )
        user_prompt = support_prompt.build_write_prompt(
            draft=draft,
            instruction=instruction,
            translate=False,
            excerpts=excerpts,
            kb_articles=kb_articles,
            ticket=ticket,
            conversation=conversation,
            language=language,
        )
        temperature = 0.4
        feature = "support_reply" if mode == "reply" else "support_write"

    usage: dict = {}

    logger.info(
        "support assist start req=%s mode=%s translate=%s lang=%s caller=%s chunks=%d",
        req_id, mode, translate, language, caller, len(chunks),
    )

    async def _record(usage_dict: dict) -> None:
        # The desk is not a metered customer org: usage is RECORDED (ledger,
        # analytics) under its own org id, but ensure_credits is deliberately
        # not called — the default monthly allowance would 402 the desk.
        # Own session, same reasoning as _retrieve: bookkeeping must never be
        # able to break the answer.
        try:
            async with AsyncSessionLocal() as session:
                await usage_meter.record_usage(
                    session,
                    organization_id=config.SUPPORT_ASSIST_ORG_ID,
                    user_id=caller,
                    feature=feature,
                    tier="smart",
                    usage=usage_dict,
                    request_id=req_id,
                )
        except Exception as exc:  # never let bookkeeping break the answer
            logger.warning("support assist: usage record failed (req=%s): %s", req_id, exc)

    async def event_stream():
        yield _ndjson(
            {
                "type": "meta",
                "mode": mode,
                "translate": translate,
                "language": language,
                "sources": sources,
                "confidence": round(confidence, 3),
                "retrieval": retrieval,
            }
        )
        pieces: List[str] = []
        try:
            async for piece in structured.astream_text(
                system,
                user_prompt,
                temperature=temperature,
                max_output_tokens=config.SUPPORT_ASSIST_MAX_TOKENS,
                usage_out=usage,
            ):
                if piece:
                    pieces.append(piece)
                    yield _ndjson({"type": "delta", "text": piece})
        except Exception as exc:
            logger.exception("support assist streaming error (req=%s): %s", req_id, exc)
            message = (
                qa.friendly_llm_error(exc)
                if qa.is_quota_error(exc)
                else "Generation interrupted; please retry."
            )
            await _record(usage)
            yield _ndjson({"type": "error", "message": message})
            return

        text = support_prompt.guard_links(support_prompt.to_plain_text("".join(pieces)))
        answered = support_prompt.was_answered(text, language) if mode == "ask" else True
        await _record(usage)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "support assist done req=%s mode=%s lang=%s ms=%d in=%s out=%s answered=%s",
            req_id, mode, language, elapsed_ms, usage.get("input"), usage.get("output"), answered,
        )
        yield _ndjson(
            {
                "type": "done",
                "text": text,
                "was_answered": answered,
                "response_time_ms": elapsed_ms,
                "usage": {
                    "input": int(usage.get("input", 0) or 0),
                    "output": int(usage.get("output", 0) or 0),
                },
            }
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers=_STREAM_HEADERS)
