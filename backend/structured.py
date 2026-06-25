"""
Structured-output + tool-calling helpers (AWS Bedrock / Claude).

Thin, stable wrappers over the provider seam in ``llm.py``. The copilot features
import the SAME names as before:

  - generate_structured(prompt, schema)            -> validated Pydantic object
  - run_tool_loop(system, user, tools, tool_impls) -> ToolLoopResult(answer, tools_used)
  - stream_text / astream_text(system, prompt)     -> streamed text chunks
  - ToolSpec / ToolCall / ToolLoopResult           -> tool description dataclasses

All Bedrock specifics (client, model failover, throttle handling, forced
tool-use for JSON, native tool loop) live in ``llm.py``; this module only
adapts argument names (``max_output_tokens`` ↔ ``max_tokens``), picks the cost
tier per feature, and provides the async streaming wrapper.

Each wrapper accepts an optional ``usage_out`` dict it fills with
{"input", "output", "model"} so callers can meter per-org token spend
(see usage_meter.py). The sync wrappers do blocking Bedrock calls — async
callers wrap them with ``await asyncio.to_thread(...)``.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

import llm
# Re-export the tool/result dataclasses so existing callers keep importing them
# from `structured` (they now live in the lower llm layer).
from llm import ToolSpec, ToolCall, ToolLoopResult, DEFAULT_MAX_OUTPUT_TOKENS

__all__ = [
    "ToolSpec",
    "ToolCall",
    "ToolLoopResult",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "generate_structured",
    "run_tool_loop",
    "stream_text",
    "astream_text",
]

logger = logging.getLogger(__name__)


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
    tier: str = "smart",
    usage_out: Optional[dict] = None,
) -> BaseModel:
    """Ask the model for JSON matching ``schema`` (via forced tool-use) and
    return the parsed Pydantic object. Raises RuntimeError if the model returned
    no usable structured output."""
    result = llm.complete_structured(
        system,
        prompt,
        schema,
        tier=tier,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    if usage_out is not None:
        usage_out.update(
            input=result.usage.input_tokens,
            output=result.usage.output_tokens,
            model=result.usage.model,
        )
    return result.value


# ---------------------------------------------------------------------------
# (2) Tool-calling loop
# ---------------------------------------------------------------------------
def run_tool_loop(
    system: str,
    user: str,
    tools: List[ToolSpec],
    tool_impls: Dict[str, Callable[..., Any]],
    max_steps: int = 5,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    force_first_call: bool = False,
    tier: str = "smart",
    usage_out: Optional[dict] = None,
) -> ToolLoopResult:
    """Run a function-calling conversation until the model returns a final text
    answer. See llm.run_tool_loop for the mechanics."""
    return llm.run_tool_loop(
        system,
        user,
        tools,
        tool_impls,
        max_steps,
        tier=tier,
        temperature=temperature,
        max_tokens=max_output_tokens,
        force_first_call=force_first_call,
        usage_out=usage_out,
    )


# ---------------------------------------------------------------------------
# (3) Streaming free-text generation
# ---------------------------------------------------------------------------
def stream_text(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    tier: str = "smart",
    usage_out: Optional[dict] = None,
):
    """Sync generator yielding successive text chunks from the model using a
    caller-supplied system instruction."""
    yield from llm.stream_text(
        system,
        prompt,
        tier=tier,
        temperature=temperature,
        max_tokens=max_output_tokens,
        usage_out=usage_out,
    )


@dataclass
class _StreamError:
    exc: Exception


async def astream_text(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    tier: str = "smart",
    usage_out: Optional[dict] = None,
) -> AsyncIterator[str]:
    """Async wrapper around ``stream_text``: runs the blocking generator in a
    worker thread and yields chunks as they arrive. Re-raises any producer
    exception in the consuming coroutine so callers can surface a clean error."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for piece in stream_text(
                system,
                prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tier=tier,
                usage_out=usage_out,
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
