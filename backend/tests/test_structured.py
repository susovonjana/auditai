"""Unit tests for structured.py (offline — Gemini is mocked via _make_model)."""
from __future__ import annotations

import google.generativeai as genai
import pytest
from pydantic import BaseModel

import structured


# --------------------------------------------------------------------------
# Fakes: stand in for genai.GenerativeModel without any network call.
# --------------------------------------------------------------------------
class _Resp:
    """A plain text response (structured-output path)."""
    def __init__(self, text: str):
        self.text = text


class _Part:
    def __init__(self, function_call=None, text=None):
        self.function_call = function_call
        self.text = text


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Cand:
    def __init__(self, parts):
        self.content = _Content(parts)


class _RespFnCall:
    """A response that asks for tool calls; .text raises like the real SDK."""
    def __init__(self, parts):
        self.candidates = [_Cand(parts)]

    @property
    def text(self):
        raise ValueError("response has no text part (function call)")


class _RespText:
    def __init__(self, text: str):
        self._text = text
        self.candidates = [_Cand([_Part(text=text)])]

    @property
    def text(self):
        return self._text


class _Chat:
    def __init__(self, scripted):
        self._scripted = scripted
        self._i = 0

    def send_message(self, content, **kwargs):
        resp = self._scripted[self._i]
        self._i += 1
        return resp


class _StructuredModel:
    def __init__(self, resp):
        self._resp = resp

    def generate_content(self, prompt, **kwargs):
        return self._resp


class _ToolModel:
    def __init__(self, scripted):
        self._scripted = scripted

    def start_chat(self):
        return _Chat(self._scripted)


# --------------------------------------------------------------------------
# generate_structured
# --------------------------------------------------------------------------
class Person(BaseModel):
    name: str
    age: int


def test_generate_structured_parses_toy_schema(monkeypatch):
    monkeypatch.setattr(
        structured, "_make_model",
        lambda *a, **k: _StructuredModel(_Resp('{"name": "John", "age": 42}')),
    )
    out = structured.generate_structured("Extract: John is 42.", Person)
    assert isinstance(out, Person)
    assert out.name == "John"
    assert out.age == 42


def test_generate_structured_empty_output_raises(monkeypatch):
    monkeypatch.setattr(
        structured, "_make_model",
        lambda *a, **k: _StructuredModel(_Resp("")),
    )
    with pytest.raises(RuntimeError):
        structured.generate_structured("Extract nothing", Person)


# --------------------------------------------------------------------------
# run_tool_loop
# --------------------------------------------------------------------------
def test_run_tool_loop_executes_tool_then_answers(monkeypatch):
    fc = genai.protos.FunctionCall(name="get_total", args={"account": "cash"})
    scripted = [
        _RespFnCall([_Part(function_call=fc)]),       # step 1: model calls tool
        _RespText("The cash total is 500.00 SAR."),   # step 2: model answers
    ]
    monkeypatch.setattr(structured, "_make_model", lambda *a, **k: _ToolModel(scripted))

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
    fc = genai.protos.FunctionCall(name="does_not_exist", args={})
    scripted = [
        _RespFnCall([_Part(function_call=fc)]),
        _RespText("Sorry, I can't answer that."),
    ]
    monkeypatch.setattr(structured, "_make_model", lambda *a, **k: _ToolModel(scripted))

    result = structured.run_tool_loop("sys", "do something", [], {})
    # The unknown call is still recorded, and the loop still terminates cleanly.
    assert result.tools_used[0].name == "does_not_exist"
    assert result.answer == "Sorry, I can't answer that."
