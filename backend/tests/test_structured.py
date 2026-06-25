"""Unit tests for structured.py (offline — Bedrock is mocked at llm._messages_create).

The provider seam lives in llm.py; structured.py is thin wrappers over it. We
stub llm._messages_create to return fake Anthropic-Messages-shaped responses, so
no AWS/Bedrock call (and no anthropic import) happens.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

import llm
import structured


# --------------------------------------------------------------------------
# Fakes: stand in for an Anthropic Messages response without any network call.
# --------------------------------------------------------------------------
class _Usage:
    def __init__(self, i: int = 10, o: int = 5):
        self.input_tokens = i
        self.output_tokens = o


class _TextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _ToolUseBlock:
    type = "tool_use"

    def __init__(self, name: str, input: dict, id: str = "tool_1"):
        self.name = name
        self.input = input
        self.id = id


class _Resp:
    """Minimal stand-in for anthropic's Message response object."""
    def __init__(self, content, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


def _script(monkeypatch, responses):
    """Make llm._messages_create return the given responses in order."""
    state = {"i": 0}

    def fake_messages_create(**kwargs):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        return r, "fake-model"

    monkeypatch.setattr(llm, "_messages_create", fake_messages_create)


# --------------------------------------------------------------------------
# generate_structured (forced tool-use)
# --------------------------------------------------------------------------
class Person(BaseModel):
    name: str
    age: int


def test_generate_structured_parses_toy_schema(monkeypatch):
    _script(monkeypatch, [_Resp([_ToolUseBlock("emit", {"name": "John", "age": 42})])])
    out = structured.generate_structured("Extract: John is 42.", Person)
    assert isinstance(out, Person)
    assert out.name == "John"
    assert out.age == 42


def test_generate_structured_empty_output_raises(monkeypatch):
    # No tool_use block in the response → no structured payload → RuntimeError.
    _script(monkeypatch, [_Resp([_TextBlock("")])])
    with pytest.raises(RuntimeError):
        structured.generate_structured("Extract nothing", Person)


def test_generate_structured_fills_usage_out(monkeypatch):
    _script(monkeypatch, [_Resp([_ToolUseBlock("emit", {"name": "Jo", "age": 9})])])
    usage: dict = {}
    structured.generate_structured("x", Person, usage_out=usage)
    assert usage["input"] == 10 and usage["output"] == 5
    assert usage["model"] == "fake-model"


# --------------------------------------------------------------------------
# run_tool_loop (native tool use)
# --------------------------------------------------------------------------
def test_run_tool_loop_executes_tool_then_answers(monkeypatch):
    _script(monkeypatch, [
        _Resp([_ToolUseBlock("get_total", {"account": "cash"}, id="t1")], stop_reason="tool_use"),
        _Resp([_TextBlock("The cash total is 500.00 SAR.")], stop_reason="end_turn"),
    ])

    seen = {}

    def get_total(account):
        seen["account"] = account
        return {"amount": 500.0, "currency": "SAR"}

    tools = [
        structured.ToolSpec(
            name="get_total",
            description="Total for an account",
            parameters={
                "type": "object",
                "properties": {"account": {"type": "string"}},
                "required": ["account"],
            },
        )
    ]
    result = structured.run_tool_loop(
        "You answer from tools.", "What is the cash total?", tools, {"get_total": get_total}
    )

    assert seen["account"] == "cash"
    assert result.answer == "The cash total is 500.00 SAR."
    assert len(result.tools_used) == 1
    assert result.tools_used[0].name == "get_total"
    assert result.tools_used[0].args == {"account": "cash"}


def test_run_tool_loop_unknown_tool_is_reported_to_model(monkeypatch):
    _script(monkeypatch, [
        _Resp([_ToolUseBlock("does_not_exist", {}, id="t1")], stop_reason="tool_use"),
        _Resp([_TextBlock("Sorry, I can't answer that.")], stop_reason="end_turn"),
    ])

    result = structured.run_tool_loop("sys", "do something", [], {})
    # The unknown call is still recorded, and the loop still terminates cleanly.
    assert result.tools_used[0].name == "does_not_exist"
    assert result.answer == "Sorry, I can't answer that."
