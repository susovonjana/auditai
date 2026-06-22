"""Unit tests for copilot_tools (offline — requests is mocked)."""
import copilot_tools
from copilot_tools import CopilotContext, build_tool_impls, TOOL_SPECS


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_tool_impl_sends_grant_and_scopes_to_file(monkeypatch):
    calls = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.update(url=url, params=params, headers=headers)
        return _FakeResp(200, {
            "message": "ok", "success": True,
            "data": {"accounts": [{"account_name": "Trade receivables", "cy_amount": 1000}]},
        })

    monkeypatch.setattr(copilot_tools.requests, "get", fake_get)
    ctx = CopilotContext(audit_file_id=56, grant="GRANT123", base_url="http://be/api/v1/internal")
    out = build_tool_impls(ctx)["get_trial_balance"]("recv")

    assert out["accounts"][0]["account_name"] == "Trade receivables"  # unwrapped .data
    assert "/copilot/audit_files/56/trial_balance" in calls["url"]
    assert calls["headers"]["X-Copilot-Grant"] == "GRANT123"
    assert calls["params"]["contains"] == "recv"


def test_grant_and_file_id_are_not_model_visible():
    for spec in TOOL_SPECS:
        props = spec.parameters.get("properties", {})
        assert "audit_file_id" not in props
        assert "grant" not in props


def test_get_drops_none_params(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return _FakeResp(200, {"data": {"ok": True}})

    monkeypatch.setattr(copilot_tools.requests, "get", fake_get)
    ctx = CopilotContext(1, "g")
    build_tool_impls(ctx)["get_procedure_results"](coa_original_id=None, account=None)
    assert captured["params"] == {}  # None values stripped before sending


def test_http_error_returns_error_dict(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResp(403, {"message": "Invalid grant"})

    monkeypatch.setattr(copilot_tools.requests, "get", fake_get)
    ctx = CopilotContext(1, "bad")
    out = build_tool_impls(ctx)["get_risks"]()
    assert "error" in out
