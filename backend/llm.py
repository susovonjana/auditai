"""
LLM provider layer — AWS Bedrock (Claude) via the AnthropicBedrock client.

This is the SINGLE seam every generative call in the app goes through. The rest
of the codebase (qa.py, structured.py, query_expander.py) talks to these
primitives, so swapping providers later means editing only this file.

Primitives
----------
  - complete_text(system, prompt, *, tier)            -> LlmResult(text, usage)
  - stream_text(system, prompt, *, tier, usage_out)   -> generator[str]
  - complete_structured(system, prompt, schema, ...)  -> StructuredResult(value, usage)
  - run_tool_loop(system, user, tools, tool_impls, …) -> ToolLoopResult

Auth
----
The AnthropicBedrock client resolves AWS credentials via the standard chain:
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (set in .env.local for dev) or the ECS
task role in production. Region comes from config.BEDROCK_REGION (eu-west-1).

Tiers
-----
  - "smart" (Sonnet-class) — generation, chat, TB Tier-3 (quality-sensitive).
  - "fast"  (Haiku-class)  — intent classify, query expansion, translation.
Each tier is a small failover list; a throttled model cools down briefly so the
next request skips it.

Usage
-----
Every primitive reports token usage (input/output + model id) so callers can
meter per-org spend (see usage_meter.py). Streaming/loop primitives accept an
optional ``usage_out`` dict they fill with {"input", "output", "model"}.

All four primitives are SYNCHRONOUS (blocking Bedrock calls). Async callers wrap
them with ``await asyncio.to_thread(...)`` exactly as qa.py / structured.py do.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from config import (
    BEDROCK_MAX_RETRIES,
    BEDROCK_MODEL_COOLDOWN_SEC,
    BEDROCK_REGION,
    BEDROCK_TIER_MODELS,
)

logger = logging.getLogger(__name__)

# Claude models are not "thinking" by default here, but keep a generous default
# so multi-step tool loops and long procedures never truncate mid-answer.
DEFAULT_MAX_OUTPUT_TOKENS = 4096


# ---------------------------------------------------------------------------
# Provider-agnostic tool + result containers
# (defined here so llm.py is the lowest layer; structured.py re-exports them,
#  preserving `from structured import ToolSpec, ToolLoopResult` for callers.)
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


@dataclass
class LlmUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class LlmResult:
    text: str
    usage: LlmUsage


@dataclass
class StructuredResult:
    value: BaseModel
    usage: LlmUsage


# ---------------------------------------------------------------------------
# Bedrock client (cached)
# ---------------------------------------------------------------------------
_client_cache = None


def _client():
    """Return a cached AnthropicBedrock client. Credentials/region resolve via
    the standard AWS chain; nothing here holds secrets."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    try:
        from anthropic import AnthropicBedrock
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "anthropic[bedrock] is not installed. Run: pip install -r requirements.txt"
        ) from exc
    _client_cache = AnthropicBedrock(aws_region=BEDROCK_REGION)
    return _client_cache


# ---------------------------------------------------------------------------
# Throttle detection + per-model cooldown / failover
# ---------------------------------------------------------------------------
def is_throttle_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a Bedrock throttle / rate-limit (HTTP 429).
    Bedrock surfaces this as anthropic.RateLimitError or a ThrottlingException;
    match by type name, status code, and message so we stay robust across SDK
    versions."""
    name = type(exc).__name__
    if name in ("RateLimitError", "ThrottlingException", "TooManyRequestsException"):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return (
        "throttl" in msg
        or "too many requests" in msg
        or "rate limit" in msg
        or "(429)" in msg
        or msg.startswith("429 ")
    )


# Backwards-compat alias: callers historically imported "quota" detection.
is_quota_error = is_throttle_error

_cooldown_until: Dict[str, float] = {}


def _tier_models(tier: str) -> List[str]:
    return BEDROCK_TIER_MODELS.get(tier) or BEDROCK_TIER_MODELS["smart"]


def _model_order(tier: str) -> List[str]:
    """Tier's model list with currently-cooling-down ids moved to the end."""
    now = time.monotonic()
    fresh, cooled = [], []
    for m in _tier_models(tier):
        (cooled if _cooldown_until.get(m, 0) > now else fresh).append(m)
    return fresh + cooled


def _mark_cooldown(model: str) -> None:
    _cooldown_until[model] = time.monotonic() + BEDROCK_MODEL_COOLDOWN_SEC
    logger.info(
        "llm: model %s throttled — cooling down for %ds", model, BEDROCK_MODEL_COOLDOWN_SEC
    )


def _backoff_seconds(attempt: int) -> float:
    # Exponential with jitter: 0.5, 1, 2 … capped at 8s.
    return min(8.0, (0.5 * (2 ** attempt))) + random.uniform(0.0, 0.4)


def _base_kwargs(
    *, system: Optional[str], messages: list, max_tokens: int, temperature: float
) -> dict:
    kwargs: dict = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        # Prompt caching: the system prompt is static and identical across every
        # request that uses it, so cache it as an ephemeral (5-min TTL) prefix.
        # Bedrock/Claude only creates a cache entry when the prefix meets the
        # model's minimum cacheable length, so this is a harmless no-op for the
        # short fast-tier prompts and a real input-token saving for the large
        # chat / tool-loop system prompts reused within the window.
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    return kwargs


def _messages_create(
    *,
    tier: str,
    system: Optional[str],
    messages: list,
    max_tokens: int,
    temperature: float,
    tools: Optional[list] = None,
    tool_choice: Optional[dict] = None,
):
    """Blocking ``messages.create`` with per-model retry (transient throttling)
    then failover to the next model in the tier. Returns (response, model_id).
    Re-raises the last throttle error if every model is exhausted; non-throttle
    errors propagate immediately."""
    client = _client()
    order = _model_order(tier)
    primary = order[0]
    last_exc: Optional[Exception] = None

    for model in order:
        attempt = 0
        while True:
            try:
                kwargs = _base_kwargs(
                    system=system, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                if tools is not None:
                    kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                resp = client.messages.create(model=model, **kwargs)
                if model != primary:
                    logger.info("llm[%s]: fell back from %s to %s", tier, primary, model)
                return resp, model
            except Exception as exc:
                if is_throttle_error(exc):
                    if attempt < BEDROCK_MAX_RETRIES:
                        attempt += 1
                        time.sleep(_backoff_seconds(attempt))
                        continue
                    _mark_cooldown(model)
                    last_exc = exc
                    break  # move on to the next model in the tier
                raise
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
def _text_from(resp) -> str:
    """Concatenate the text content blocks of a Messages response."""
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _usage_from(resp, model: str) -> LlmUsage:
    u = getattr(resp, "usage", None)
    return LlmUsage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        model=model,
    )


def _sanitize_tool_result(result: Any) -> dict:
    """A tool_result payload should be JSON. Wrap non-dicts and round-trip so
    only JSON-native types remain."""
    if not isinstance(result, dict):
        result = {"result": result}
    try:
        return json.loads(json.dumps(result, default=str))
    except (TypeError, ValueError):
        return {"result": str(result)}


# ---------------------------------------------------------------------------
# (1) Free-text completion
# ---------------------------------------------------------------------------
def complete_text(
    system: Optional[str],
    prompt: str,
    *,
    tier: str = "smart",
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> LlmResult:
    resp, model = _messages_create(
        tier=tier,
        system=system or None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return LlmResult(text=_text_from(resp), usage=_usage_from(resp, model))


# ---------------------------------------------------------------------------
# (2) Structured output via forced tool-use (replaces Gemini response_schema)
# ---------------------------------------------------------------------------
_STRUCTURED_TOOL_NAME = "emit"


def complete_structured(
    system: Optional[str],
    prompt: str,
    schema: Type[BaseModel],
    *,
    tier: str = "smart",
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> StructuredResult:
    """Force the model to call a single tool whose input_schema is ``schema``,
    then validate the tool input into the Pydantic object. This is the robust
    Bedrock equivalent of Gemini's constrained JSON decoding."""
    tool = {
        "name": _STRUCTURED_TOOL_NAME,
        "description": "Return the result as structured data matching the schema.",
        "input_schema": schema.model_json_schema(),
    }
    resp, model = _messages_create(
        tier=tier,
        system=system or None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        tools=[tool],
        tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
    )
    payload = None
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _STRUCTURED_TOOL_NAME:
            payload = block.input
            break
    if payload is None:
        raise RuntimeError(
            "Bedrock returned no structured tool_use output (possibly truncated "
            "by max_tokens, or blocked). Try a larger token budget."
        )
    return StructuredResult(value=schema.model_validate(payload), usage=_usage_from(resp, model))


# ---------------------------------------------------------------------------
# (3) Tool-calling loop (native Anthropic tool use)
# ---------------------------------------------------------------------------
def run_tool_loop(
    system: str,
    user: str,
    tools: List[ToolSpec],
    tool_impls: Dict[str, Callable[..., Any]],
    max_steps: int = 5,
    *,
    tier: str = "smart",
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    force_first_call: bool = False,
    usage_out: Optional[dict] = None,
) -> ToolLoopResult:
    """Run a function-calling conversation until the model returns a final text
    answer (or ``max_steps`` is reached). The matching callable in ``tool_impls``
    is executed for each tool the model asks for, and its JSON-serialisable
    result is fed back as a tool_result block.

    ``force_first_call`` forces a tool call on the opening turn (tool_choice
    "any"), so the model can't answer a data question from assumption before
    looking anything up. Subsequent turns use auto tool choice.
    """
    anthropic_tools = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters or {"type": "object", "properties": {}},
        }
        for t in tools
    ]
    # Prompt caching: tool definitions are static across the loop's steps (and
    # across requests), so mark the last tool as a cache breakpoint — every step
    # after the first reuses the cached tool block instead of re-sending it.
    if anthropic_tools:
        anthropic_tools[-1] = {
            **anthropic_tools[-1], "cache_control": {"type": "ephemeral"}
        }
    messages: list = [{"role": "user", "content": user}]
    tools_used: List[ToolCall] = []
    total_in = total_out = 0
    used_model = ""
    resp = None

    for step in range(max_steps):
        tool_choice = None
        if force_first_call and step == 0 and anthropic_tools:
            tool_choice = {"type": "any"}
        resp, used_model = _messages_create(
            tier=tier,
            system=system or None,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=anthropic_tools or None,
            tool_choice=tool_choice,
        )
        u = _usage_from(resp, used_model)
        total_in += u.input_tokens
        total_out += u.output_tokens

        if getattr(resp, "stop_reason", None) == "tool_use":
            # Echo the assistant's tool-use turn back, then answer each tool call.
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = block.name
                args = dict(block.input or {})
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
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(_sanitize_tool_result(result), default=str),
                })
            messages.append({"role": "user", "content": results})
            continue

        # Final text answer.
        if usage_out is not None:
            usage_out.update(input=total_in, output=total_out, model=used_model)
        return ToolLoopResult(answer=_text_from(resp), tools_used=tools_used)

    logger.warning("run_tool_loop hit max_steps=%d; returning best-effort answer", max_steps)
    if usage_out is not None:
        usage_out.update(input=total_in, output=total_out, model=used_model)
    return ToolLoopResult(
        answer=_text_from(resp) if resp is not None else "",
        tools_used=tools_used,
    )


# ---------------------------------------------------------------------------
# (4) Streaming free-text generation
# ---------------------------------------------------------------------------
def stream_text(
    system: Optional[str],
    prompt: str,
    *,
    tier: str = "smart",
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    usage_out: Optional[dict] = None,
):
    """Sync generator yielding successive text chunks from Bedrock.

    Model failover only kicks in BEFORE the first chunk is yielded; once content
    has reached the caller, a mid-stream throttle propagates (swapping models
    would lose the partial output). If ``usage_out`` is given it is filled with
    {"input", "output", "model"} once the stream completes."""
    client = _client()
    order = _model_order(tier)
    primary = order[0]
    last_exc: Optional[Exception] = None

    for model in order:
        mgr = None
        try:
            kwargs = _base_kwargs(
                system=system or None,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            mgr = client.messages.stream(model=model, **kwargs)
            stream = mgr.__enter__()
            text_iter = iter(stream.text_stream)
            first = next(text_iter)  # forces the request; raises 429 here
        except StopIteration:
            # Model produced no text — close and try the next model.
            if mgr is not None:
                try:
                    mgr.__exit__(None, None, None)
                except Exception:
                    pass
            continue
        except Exception as exc:
            if mgr is not None:
                try:
                    mgr.__exit__(type(exc), exc, None)
                except Exception:
                    pass
            if is_throttle_error(exc):
                _mark_cooldown(model)
                last_exc = exc
                continue
            raise

        # Committed to this model.
        if model != primary:
            logger.info("llm[%s].stream: fell back from %s to %s", tier, primary, model)
        try:
            if first:
                yield first
            for piece in text_iter:
                if piece:
                    yield piece
            if usage_out is not None:
                try:
                    final = stream.get_final_message()
                    u = getattr(final, "usage", None)
                    usage_out.update(
                        input=int(getattr(u, "input_tokens", 0) or 0),
                        output=int(getattr(u, "output_tokens", 0) or 0),
                        model=model,
                    )
                except Exception:
                    pass
        finally:
            try:
                mgr.__exit__(None, None, None)
            except Exception:
                pass
        return

    # Every model threw a throttle before producing any text.
    if last_exc is not None:
        raise last_exc
    return
