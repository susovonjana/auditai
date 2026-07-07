"""
Agent tool registry.

The runtime never imports tool implementations directly — it asks this registry
for the callables a given step needs, built against THIS request's
``CopilotContext`` (which carries the audit_file_id + grant). Read tools reuse
``copilot_tools.build_tool_impls`` verbatim; write tools are registered later
(per agent) and are the ONLY ones marked ``requires_approval`` so the runtime
can hold them at a checkpoint.

A write tool's ``build(ctx)`` must return a callable that closes over the
audit_file_id + grant (never model-visible args), exactly like ``CopilotContext``
does for reads — see ``agent/definitions/*`` for examples.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from copilot_tools import CopilotContext, build_tool_impls


class ToolKind(enum.Enum):
    READ = "read"
    COMPUTE = "compute"
    ANALYSIS = "analysis"
    WRITE = "write"


@dataclass
class RegisteredTool:
    """A tool the runtime can run. ``build(ctx)`` returns the bound callable."""
    name: str
    kind: ToolKind
    requires_approval: bool
    build: Callable[[CopilotContext], Callable[..., Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def impls_for(
        self, ctx: CopilotContext, names: List[str]
    ) -> Dict[str, Callable[..., Any]]:
        """Return {name: bound callable} for the requested tools, built against
        this request's context. Raises KeyError for an unregistered name."""
        out: Dict[str, Callable[..., Any]] = {}
        for n in names:
            tool = self._tools.get(n)
            if tool is None:
                raise KeyError(f"tool not registered: {n}")
            out[n] = tool.build(ctx)
        return out


# Module-wide singleton the runtime and definitions share.
REGISTRY = ToolRegistry()


def _read_builder(name: str) -> Callable[[CopilotContext], Callable[..., Any]]:
    # build_tool_impls only defines closures (no I/O), so calling it per tool is
    # cheap; binding `name` as a default keeps each closure pointing at its tool.
    def build(ctx: CopilotContext, _name: str = name) -> Callable[..., Any]:
        return build_tool_impls(ctx)[_name]
    return build


# Pre-register EVERY existing read tool (not a hard-coded subset). The key list is
# derived from build_tool_impls itself so it stays in sync if tools are added.
# A throwaway context is safe here: build_tool_impls does no network I/O, it only
# returns the closures whose names we need.
READ_TOOL_NAMES: List[str] = list(build_tool_impls(CopilotContext(0, "")).keys())
for _name in READ_TOOL_NAMES:
    REGISTRY.register(
        RegisteredTool(
            name=_name,
            kind=ToolKind.READ,
            requires_approval=False,
            build=_read_builder(_name),
        )
    )


# ---------------------------------------------------------------------------
# WRITE tools — each POSTs to a grant-scoped 1audit-be endpoint via ctx.post,
# with the audit_file_id + grant living in the closure (never model-visible).
# The runtime holds these at an approval checkpoint and runs them once, only
# after the auditor approves, with the (possibly edited) proposed_write payload.
# ---------------------------------------------------------------------------
# Writes get a generous read timeout: they run ONCE after human approval inside
# a be transaction (a bulk draft against the remote DB can take tens of seconds),
# and a premature client timeout is worse than waiting — the be transaction
# still commits, leaving changes the run never recorded (learned the hard way).
_WRITE_READ_TIMEOUT_SEC = 180


def _post_builder(endpoint: str) -> Callable[[CopilotContext], Callable[..., Any]]:
    def build(ctx: CopilotContext) -> Callable[..., Any]:
        def tool(payload: dict) -> Any:
            return ctx.post(endpoint, payload or {}, read_timeout=_WRITE_READ_TIMEOUT_SEC)
        return tool
    return build


# Engagement build-out writes (1audit-be grant-scoped internal endpoints).
REGISTRY.register(RegisteredTool(
    "persist_tb_mappings", ToolKind.WRITE, True,
    _post_builder("trial_balance/batch_update_accounts_mapping"),
))
REGISTRY.register(RegisteredTool(
    "create_lead_schedules", ToolKind.WRITE, True,
    _post_builder("trial_balance/bulk_create_lead_sheets"),
))

# Substantive testing write (draft per-sample testing conclusions back to the file).
REGISTRY.register(RegisteredTool(
    "save_test_conclusions", ToolKind.WRITE, True,
    _post_builder("sampling/save_test_conclusions"),
))


def _wp_post_builder(endpoint_tpl: str) -> Callable[[CopilotContext], Callable[..., Any]]:
    """Like ``_post_builder`` but for endpoints scoped to ONE working paper:
    the payload carries ``working_paper_id`` (placed there by the definition's
    ``prepare_write``, never invented by the model), which is popped into the
    URL path instead of being sent in the body."""
    def build(ctx: CopilotContext) -> Callable[..., Any]:
        def tool(payload: dict) -> Any:
            body = dict(payload or {})
            wp_id = body.pop("working_paper_id", None)
            try:
                wp_id = int(wp_id)
            except (TypeError, ValueError):
                return {"error": "working_paper_id missing from write payload"}
            return ctx.post(endpoint_tpl.format(wp_id=wp_id), body, read_timeout=_WRITE_READ_TIMEOUT_SEC)
        return tool
    return build


# Procedure build-out writes: create the approved procedure tree in one
# transaction, and undo a whole run (both scoped to one working paper).
REGISTRY.register(RegisteredTool(
    "bulk_create_program_sections", ToolKind.WRITE, True,
    _wp_post_builder("working_papers/{wp_id}/program_sections/bulk_create"),
))
REGISTRY.register(RegisteredTool(
    "undo_program_sections_run", ToolKind.WRITE, True,
    _wp_post_builder("working_papers/{wp_id}/program_sections/undo_run"),
))
REGISTRY.register(RegisteredTool(
    "restore_program_sections_run", ToolKind.WRITE, True,
    _wp_post_builder("working_papers/{wp_id}/program_sections/restore_run"),
))
