"""
Structured-output + tool-calling helpers for Gemini.

Two reusable primitives the copilot features build on:

  - generate_structured(prompt, schema)            -> validated Pydantic object
  - run_tool_loop(system, user, tools, tool_impls) -> ToolLoopResult(answer, tools_used)

Provider isolation
------------------
Every Gemini-specific detail lives behind these functions and the small
``_make_model`` factory, so swapping the LLM later means editing only this
file. We reuse the SAME model-failover list (``config.GEMINI_MODELS``), the
SAME quota detection (``qa.is_quota_error``) and the SAME cooldown behaviour as
the chat engine, but build clients with a caller-supplied system instruction
(NOT the audit-chat system prompt baked into qa._get_gemini).

Both functions are synchronous (they do blocking Gemini calls). Async callers
should wrap them with ``await asyncio.to_thread(...)`` exactly as qa.py does for
``_call_gemini_sync``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from config import (
    GEMINI_API_KEY,
    GEMINI_MODELS,
    GEMINI_MODEL_COOLDOWN_SEC,
)
from qa import is_quota_error  # identical quota/rate-limit detection

logger = logging.getLogger(__name__)

# gemini-flash-latest / 2.5-flash are "thinking" models whose reasoning tokens
# count toward max_output_tokens. A small cap can make the model spend the whole
# budget thinking and return NO answer text, so default generously.
DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Per-model quota cooldown registry (mirrors qa.py; kept separate so the two
# modules track cooldowns independently — it is only a soft optimisation).
_model_cooldown_until: Dict[str, float] = {}
_gemini_configured = False


# ---------------------------------------------------------------------------
# Provider-agnostic tool description
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """A function the model may call.

    ``parameters`` is a JSON-Schema object (``{"type": "object", "properties": …}``).
    Use an empty-properties object for a no-argument tool.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass
class ToolCall:
    """One function call the model made, in execution order."""
    name: str
    args: Dict[str, Any]


@dataclass
class ToolLoopResult:
    answer: str
    tools_used: List[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gemini client plumbing (failover + cooldown), system-instruction aware
# ---------------------------------------------------------------------------
def _ensure_configured() -> None:
    global _gemini_configured
    if _gemini_configured:
        return
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
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_configured = True


def _make_model(
    model_name: str,
    *,
    system_instruction: Optional[str] = None,
    tools: Optional[list] = None,
    generation_config: Optional[dict] = None,
):
    """Build a Gemini GenerativeModel. Isolated so tests can monkeypatch it."""
    _ensure_configured()
    import google.generativeai as genai

    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        tools=tools,
        generation_config=generation_config,
    )


def _model_order() -> List[str]:
    """GEMINI_MODELS with currently-cooling-down models moved to the end."""
    now = time.monotonic()
    fresh, cooled = [], []
    for name in GEMINI_MODELS:
        (cooled if _model_cooldown_until.get(name, 0) > now else fresh).append(name)
    return fresh + cooled


def _mark_quota_exhausted(model_name: str) -> None:
    _model_cooldown_until[model_name] = time.monotonic() + GEMINI_MODEL_COOLDOWN_SEC
    logger.info(
        "structured: model %s hit quota — cooling down for %ds",
        model_name,
        GEMINI_MODEL_COOLDOWN_SEC,
    )


def _run_with_failover(make_call: Callable[[str], Any]) -> Any:
    """Call ``make_call(model_name)`` against each model in priority order,
    failing over to the next model on quota errors. Re-raises the last quota
    error if every model is exhausted; non-quota errors propagate immediately."""
    order = _model_order()
    primary = order[0]
    last_exc: Optional[Exception] = None
    for name in order:
        try:
            result = make_call(name)
        except Exception as exc:
            if is_quota_error(exc):
                _mark_quota_exhausted(name)
                last_exc = exc
                continue
            raise
        if name != primary:
            logger.info("structured: fell back from %s to %s", primary, name)
        return result
    assert last_exc is not None
    raise last_exc


def _extract_text(response) -> str:
    """Robust text extraction (mirrors qa._extract_text). Returns '' if the
    response carried no text part (e.g. truncated by max_output_tokens)."""
    try:
        return (response.text or "").strip()
    except Exception:
        parts = []
        for cand in getattr(response, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    parts.append(part.text)
        return "\n".join(parts).strip()


def _extract_function_calls(response) -> List[tuple]:
    """Return [(name, args_dict), ...] for any function-call parts in the
    model's latest turn. Empty list means the model produced a final answer."""
    import google.generativeai as genai

    calls: List[tuple] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return calls
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", []) or []:
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", ""):
            try:
                d = genai.protos.FunctionCall.to_dict(fc)
                args = d.get("args") or {}
            except Exception:
                args = dict(getattr(fc, "args", {}) or {})
            calls.append((fc.name, args))
    return calls


def _sanitize_tool_result(result: Any) -> dict:
    """A FunctionResponse payload must be a JSON-object (proto Struct). Wrap
    non-dicts and round-trip through JSON so only JSON-native types remain."""
    if not isinstance(result, dict):
        result = {"result": result}
    try:
        return json.loads(json.dumps(result, default=str))
    except (TypeError, ValueError):
        return {"result": str(result)}


# ---------------------------------------------------------------------------
# (1) Structured output
# ---------------------------------------------------------------------------
def generate_structured(
    prompt: str,
    schema: Type[BaseModel],
    temperature: float = 0.0,
    *,
    system: Optional[str] = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> BaseModel:
    """Ask Gemini for JSON matching ``schema`` and return the parsed Pydantic
    object. Uses response_schema for constrained decoding, with model failover.

    Raises RuntimeError if the model returned no parseable JSON (e.g. the output
    was truncated by ``max_output_tokens``)."""
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    def call(model_name: str):
        model = _make_model(
            model_name,
            system_instruction=system,
            generation_config=generation_config,
        )
        return model.generate_content(prompt)

    response = _run_with_failover(call)
    text = _extract_text(response)
    if not text:
        raise RuntimeError(
            "Gemini returned no structured output (possibly truncated by "
            "max_output_tokens, or blocked). Try a larger token budget."
        )
    return schema.model_validate_json(text)


# ---------------------------------------------------------------------------
# (2) Tool-calling loop
# ---------------------------------------------------------------------------
def _run_tool_loop_once(
    model_name: str,
    system: str,
    user: str,
    tools: List[ToolSpec],
    tool_impls: Dict[str, Callable[..., Any]],
    max_steps: int,
    temperature: float,
    max_output_tokens: int,
) -> ToolLoopResult:
    """One full function-calling conversation on a single model. Raises on
    quota/other errors (the caller decides whether to fail over)."""
    import google.generativeai as genai

    fn_decls = [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in tools
    ]
    sdk_tools = [{"function_declarations": fn_decls}] if fn_decls else None
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    model = _make_model(
        model_name,
        system_instruction=system,
        tools=sdk_tools,
        generation_config=generation_config,
    )
    chat = model.start_chat()
    response = chat.send_message(user)
    tools_used: List[ToolCall] = []

    for _step in range(max_steps):
        fcalls = _extract_function_calls(response)
        if not fcalls:
            return ToolLoopResult(answer=_extract_text(response), tools_used=tools_used)

        response_parts = []
        for name, args in fcalls:
            tools_used.append(ToolCall(name=name, args=args))
            impl = tool_impls.get(name)
            if impl is None:
                result: Any = {"error": f"Unknown tool '{name}'"}
            else:
                try:
                    result = impl(**args)
                except Exception as exc:  # surface tool failure to the model
                    logger.exception("run_tool_loop: tool %s raised", name)
                    result = {"error": str(exc)}
            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name, response=_sanitize_tool_result(result)
                    )
                )
            )
        response = chat.send_message(response_parts)

    logger.warning(
        "run_tool_loop hit max_steps=%d; returning best-effort answer", max_steps
    )
    return ToolLoopResult(answer=_extract_text(response), tools_used=tools_used)


def run_tool_loop(
    system: str,
    user: str,
    tools: List[ToolSpec],
    tool_impls: Dict[str, Callable[..., Any]],
    max_steps: int = 5,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> ToolLoopResult:
    """Run a function-calling conversation until the model returns a final text
    answer (or ``max_steps`` is reached).

    The model is given ``tools``; whenever it asks to call one, the matching
    callable in ``tool_impls`` is executed and its (JSON-serialisable) result is
    fed back. Returns the final answer plus the ordered list of tool calls made.

    Model failover happens per-attempt: we try each model in priority order and
    move to the next on a quota error OR an EMPTY final answer. The latter
    matters because the cheap "lite" models sometimes return no text after a
    function response (finish_reason STOP, zero output tokens) — escalating to a
    stronger model recovers a real answer. Tools are read-only and idempotent, so
    re-running them on the next model is safe.
    """
    order = _model_order()
    primary = order[0]
    last_exc: Optional[Exception] = None
    last_result: Optional[ToolLoopResult] = None

    for model_name in order:
        try:
            result = _run_tool_loop_once(
                model_name, system, user, tools, tool_impls,
                max_steps, temperature, max_output_tokens,
            )
        except Exception as exc:
            if is_quota_error(exc):
                _mark_quota_exhausted(model_name)
                last_exc = exc
                continue
            raise

        if (result.answer or "").strip():
            if model_name != primary:
                logger.info("run_tool_loop: fell back from %s to %s", primary, model_name)
            return result

        logger.info(
            "run_tool_loop: %s returned an empty answer; trying next model", model_name
        )
        last_result = result

    # Every model was quota-exhausted before producing anything → surface quota.
    if last_result is None and last_exc is not None:
        raise last_exc
    # Otherwise return the last (empty) result; the caller maps empty → friendly error.
    return last_result if last_result is not None else ToolLoopResult(answer="", tools_used=[])


# ---------------------------------------------------------------------------
# (3) Streaming free-text generation (custom system instruction)
# ---------------------------------------------------------------------------
def _chunk_text(ev) -> str:
    """Extract text from one streaming event (mirrors qa._chunk_text)."""
    try:
        return ev.text or ""
    except Exception:
        t = ""
        for cand in getattr(ev, "candidates", []) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    t += part.text
        return t


def stream_text(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
):
    """Sync generator yielding successive text chunks from Gemini using a
    caller-supplied system instruction.

    Model failover only kicks in BEFORE the first chunk is yielded; once content
    has reached the caller, a mid-stream quota error propagates (swapping models
    would lose the partial output). Mirrors qa._stream_gemini_sync."""
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    order = _model_order()
    primary = order[0]
    last_exc: Optional[Exception] = None
    for name in order:
        try:
            model = _make_model(
                name, system_instruction=system, generation_config=generation_config
            )
            response_stream = model.generate_content(prompt, stream=True)
            iterator = iter(response_stream)
            first_ev = next(iterator)  # forces the API call (raises 429 here)
        except StopIteration:
            continue
        except Exception as exc:
            if is_quota_error(exc):
                _mark_quota_exhausted(name)
                last_exc = exc
                continue
            raise
        if name != primary:
            logger.info("structured.stream_text: fell back from %s to %s", primary, name)
        t = _chunk_text(first_ev)
        if t:
            yield t
        for ev in iterator:
            t = _chunk_text(ev)
            if t:
                yield t
        return
    assert last_exc is not None
    raise last_exc


@dataclass
class _StreamError:
    exc: Exception


async def astream_text(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> AsyncIterator[str]:
    """Async wrapper around ``stream_text``: runs the blocking generator in a
    worker thread and yields chunks as they arrive. Re-raises any producer
    exception in the consuming coroutine so callers can surface a clean error."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for piece in stream_text(
                system, prompt, temperature=temperature, max_output_tokens=max_output_tokens
            ):
                asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
        except Exception as exc:  # forward to consumer
            asyncio.run_coroutine_threadsafe(queue.put(_StreamError(exc)), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # sentinel

    asyncio.create_task(asyncio.to_thread(producer))

    while True:
        piece = await queue.get()
        if piece is None:
            break
        if isinstance(piece, _StreamError):
            raise piece.exc
        yield piece
