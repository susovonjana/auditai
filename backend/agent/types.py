"""
Agent definition interface + the per-run context object.

An ``AgentDefinition`` is the "recipe" for one agent: the steps it runs and how
it turns the gathered data into a final structured result. The runtime executes
the recipe, pausing at write checkpoints. A definition does NO persistence and
NO auth — it only describes work and runs the compute/analysis logic.

Numbers rule: ``execute_step`` / ``synthesize`` may call the LLM only for prose,
proposals, or recommendations — never to compute or invent a financial figure
(those come from read tools or pure-Python compute steps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from copilot_tools import CopilotContext


@dataclass
class PlannedStep:
    """One step in an agent's plan. ``type`` is read|compute|analysis|write.
    For read/write steps ``tool`` names a registered tool and ``args`` are its
    call kwargs. For compute/analysis steps ``tool`` names a handler the
    definition's ``execute_step`` dispatches on."""
    title: str
    type: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False


@dataclass
class StepResult:
    """The recorded output of a completed step (kept in run order)."""
    idx: int
    title: str
    type: str
    tool: Optional[str]
    args: Dict[str, Any]
    output: Any


@dataclass
class RunContext:
    """Everything a definition's callables may read for one run. Rebuilt per
    request from this request's grant + the persisted step outputs."""
    copilot: CopilotContext
    audit_file_id: int
    language: str = "en"
    organization_id: Optional[str] = None
    results: List[StepResult] = field(default_factory=list)
    # filled by the most recent LLM call so the runtime can meter run credits
    usage_out: Dict[str, Any] = field(default_factory=dict)
    # the persisted run id (set by the runtime) — write payloads embed it so a
    # whole run's DB writes can be traced/undone as one unit (ai_run_id stamp)
    run_id: Optional[str] = None
    # scope param for working-paper-scoped agents (procedure_buildout). Only
    # needed at START (build_plan bakes it into step args, which persist).
    working_paper_id: Optional[int] = None
    # firm house-style examples (proc_memory), fetched by the router at start —
    # only the drafting analysis step reads them, which runs before any pause.
    style_examples: List[str] = field(default_factory=list)
    # the granted file is an audit file TEMPLATE — wp-scoped agents plan/draft
    # the firm's standard program (a skeleton has no client or risk data). Only
    # needed at START: build_plan and the drafting step run before any pause.
    is_template: bool = False

    def find(self, tool: str, **arg_match: Any) -> Optional[Any]:
        """Return the output of an earlier step by tool name (and optional arg
        match, e.g. find('get_financial_statement', statement_type='balance_sheet'))."""
        for r in self.results:
            if r.tool == tool and all(r.args.get(k) == v for k, v in arg_match.items()):
                return r.output
        return None


@runtime_checkable
class AgentDefinition(Protocol):
    agent_type: str
    allowed_tools: List[str]

    def default_goal(self, audit_file_id: int) -> str:
        """Human-readable goal shown when the caller doesn't supply one."""
        ...

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        """Return the ordered steps. v1 definitions return a STATIC list."""

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        """Run a compute or analysis step and return its JSON-safe output.
        Read/write steps never reach here — the runtime runs those via the tool
        registry."""

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        """Produce the final structured result_summary from all step outputs."""


# Every agent registers itself here (agent_type -> definition instance).
AGENT_DEFINITIONS: Dict[str, AgentDefinition] = {}


def register_definition(definition: AgentDefinition) -> None:
    AGENT_DEFINITIONS[definition.agent_type] = definition


_LANG_NAME = {"en": "English", "ar": "Arabic"}


def language_directive(language: Optional[str]) -> str:
    """System-prompt suffix pinning the LLM's prose to the RUNNING user's UI
    language (ctx.language). Every agent that generates human-readable text —
    findings, reasons, summaries, recommendations — appends this so an Arabic
    user reads Arabic and an English user reads English, regardless of which
    language dominates the file's data. Explicit for English too: Arabic-heavy
    file data would otherwise pull the model into Arabic."""
    name = _LANG_NAME.get((language or "en").lower(), "English")
    return (
        f"\n\nLANGUAGE: The auditor reading your output works in {name}. Write every "
        f"human-readable text you produce — summaries, reasons, findings, notes, "
        f"explanations, recommendations — in {name}. Do NOT translate identifiers, "
        f"codes, statuses, or account/working-paper names quoted from the data; keep "
        f"those exactly as they appear."
    )
