"""Unit tests for the Procedure Build-Out agent (offline — no Bedrock, no be).

The LLM call itself is not exercised; we test the pure-Python structure
discipline — tree flattening (titles group by order, procedures nest),
response-set validation, HTML sanitization, depth/size caps, and the write
checkpoint payload (working_paper_id + ai_run_id stamp + sections shape).
These are the guarantees that keep a hallucinated structure out of the file.
"""
from __future__ import annotations

import pytest

from agent.definitions.procedure_buildout import (
    ProcedureBuildOutAgent,
    ProcedurePlan,
    ResponseSetProposal,
    SectionNode,
    StepNode,
    SubStepNode,
    flatten_plan,
    sanitize_html,
    summarize_wp_config,
    validate_response_sets,
    _MAX_TOP_LEVEL,
)
from agent.types import PlannedStep, RunContext, StepResult
from copilot_tools import CopilotContext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ctx(**kwargs) -> RunContext:
    ctx = RunContext(copilot=CopilotContext(0, ""), audit_file_id=56, **kwargs)
    return ctx


def _plan_tree() -> list:
    return [
        SectionNode(
            section_type="title",
            title="Revenue — Occurrence & Cut-off",
            rationale="groups the occurrence work",
            children=[
                StepNode(
                    procedure_html="<ol><li>Select the last 10 invoices before year-end.</li></ol>",
                    assertions=["Occurrence"],
                    response_sets=[
                        ResponseSetProposal(type="picklist", placeholder="Result",
                                            options=["Satisfactory", "Exception noted", "N/A"]),
                        ResponseSetProposal(type="text_area", placeholder="Work performed"),
                    ],
                    rationale="verdict is a fixed set",
                    children=[
                        SubStepNode(
                            procedure_html="<p>Trace each invoice to its delivery document.</p>",
                            response_sets=[ResponseSetProposal(type="date", placeholder="Traced on")],
                            rationale="a date is the natural answer",
                        ),
                    ],
                ),
            ],
        ),
        SectionNode(
            section_type="procedure",
            procedure_html="<p>Conclude on the area.</p>",
            response_sets=[ResponseSetProposal(type="text_area", placeholder="Conclusion")],
            rationale="narrative",
        ),
    ]


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------
def test_flatten_titles_group_by_order_and_procedures_nest():
    out = flatten_plan(_plan_tree(), assertions_enabled=True)
    rows = out["sections"]

    # title first, then its procedure at TOP level (titles never parent),
    # then the sub-step nested under the procedure, then the standalone one.
    assert [r["section_type"] for r in rows] == [1, 2, 2, 2]
    title, proc, sub, standalone = rows
    assert title["parent_temp_id"] is None
    assert proc["parent_temp_id"] is None          # flattened below the title
    assert sub["parent_temp_id"] == proc["temp_id"]  # real nesting
    assert standalone["parent_temp_id"] is None

    # temp ids are unique and parents appear before children
    ids = [r["temp_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert ids.index(proc["temp_id"]) < ids.index(sub["temp_id"])

    # content mapped correctly
    assert title["title"] == "Revenue — Occurrence & Cut-off"
    assert "Select the last 10 invoices" in proc["procedure"]
    assert proc["assertions"] == ["Occurrence"]
    assert [rs["type"] for rs in proc["response_sets"]] == ["picklist", "text_area"]
    assert proc["response_sets"][0]["options"] == ["Satisfactory", "Exception noted", "N/A"]

    assert out["counts"]["total_sections"] == 4
    assert out["counts"]["titles"] == 1
    assert out["counts"]["procedures"] == 3
    assert out["counts"]["response_types"] == {"picklist": 1, "text_area": 2, "date": 1}


def test_flatten_strips_assertions_when_disabled():
    out = flatten_plan(_plan_tree(), assertions_enabled=False)
    assert all("assertions" not in r for r in out["sections"])


def test_flatten_caps_top_level_and_notes_it():
    nodes = [
        SectionNode(section_type="procedure", procedure_html=f"<p>Step {i}</p>")
        for i in range(_MAX_TOP_LEVEL + 5)
    ]
    out = flatten_plan(nodes, assertions_enabled=False)
    assert len(out["sections"]) == _MAX_TOP_LEVEL
    assert any("top-level" in n for n in out["notes"])


def test_flatten_drops_empty_nodes_with_notes():
    nodes = [
        SectionNode(section_type="title", title="   "),
        SectionNode(section_type="procedure", procedure_html="<p>   </p>"),
        SectionNode(section_type="procedure", procedure_html="<p>Real step</p>"),
    ]
    out = flatten_plan(nodes, assertions_enabled=False)
    assert len(out["sections"]) == 1
    assert "Real step" in out["sections"][0]["procedure"]
    assert len(out["notes"]) == 2


def test_flatten_bilingual_two_field_pairs():
    """Every two-field pair (procedure/procedure_sl, title/title_sl,
    description/description_sl, placeholder/placeholder_sl, option title/title_sl)
    flows through when bilingual — and is STRIPPED when the file is single-language."""
    nodes = [
        SectionNode(section_type="comment",
                    procedure_html="<p>Objective: evidence over receivables.</p>",
                    procedure_html_sl="<p>الهدف: أدلة حول الذمم المدينة.</p>"),
        SectionNode(section_type="title", title="Cut-off", title_sl="الفصل الزمني"),
        SectionNode(
            section_type="procedure",
            procedure_html="<p>Trace invoices to GRNs.</p>",
            procedure_html_sl="<p>تتبع الفواتير إلى إشعارات الاستلام.</p>",
            response_sets=[ResponseSetProposal(
                type="picklist", placeholder="Result", placeholder_sl="النتيجة",
                options=["Completed no exceptions", "Completed with exceptions"],
                options_sl=["مكتمل دون ملاحظات", "مكتمل مع ملاحظات"],
            )],
        ),
    ]
    out = flatten_plan(nodes, assertions_enabled=False, bilingual=True)
    comment, title, proc = out["sections"]
    assert "الهدف" in comment["description_sl"]
    assert title["title_sl"] == "الفصل الزمني"
    assert "تتبع" in proc["procedure_sl"]
    rs = proc["response_sets"][0]
    assert rs["placeholder_sl"] == "النتيجة"
    assert rs["options_sl"] == ["مكتمل دون ملاحظات", "مكتمل مع ملاحظات"]
    # preview carries both languages for the checkpoint UI
    assert out["preview"][0]["html_sl"]
    assert out["preview"][1]["text_sl"] == "الفصل الزمني"

    # single-language: the same tree loses every _sl field
    out_en = flatten_plan(nodes, assertions_enabled=False, bilingual=False)
    c2, t2, p2 = out_en["sections"]
    assert "description_sl" not in c2 and "title_sl" not in t2 and "procedure_sl" not in p2
    assert "placeholder_sl" not in p2["response_sets"][0]
    assert "options_sl" not in p2["response_sets"][0]


def test_options_sl_pairs_by_index_and_follows_drops():
    notes: list = []
    out = validate_response_sets(
        [{"type": "multi_picklist", "placeholder": "Evidence",
          "options": ["Invoice", "", "Invoice", "GRN"],
          "options_sl": ["فاتورة", "x", "فاتورة مكررة", "إشعار استلام"]}],
        notes, "node 1", bilingual=True,
    )
    # empty + duplicate primaries dropped; translations follow their option
    assert out[0]["options"] == ["Invoice", "GRN"]
    assert out[0]["options_sl"] == ["فاتورة", "إشعار استلام"]


def test_summarize_wp_config_languages():
    base = {"working_paper": {"name": "Revenue", "config": None,
                              "languages": {"primary": "en", "secondary": "ar", "use_secondary": True}},
            "items": []}
    facts = summarize_wp_config(base)
    assert facts["bilingual"] is True
    assert facts["primary_language"] == "en" and facts["secondary_language"] == "ar"

    base["working_paper"]["languages"]["use_secondary"] = False
    facts2 = summarize_wp_config(base)
    assert facts2["bilingual"] is False and facts2["secondary_language"] is None

    base["working_paper"]["languages"] = None
    facts3 = summarize_wp_config(base)
    assert facts3["bilingual"] is False and facts3["primary_language"] == "en"


def test_flatten_objective_comment():
    nodes = [
        SectionNode(section_type="comment",
                    procedure_html="<p>Objective: To obtain sufficient appropriate audit evidence.</p>",
                    rationale="house style opener"),
        SectionNode(section_type="procedure", procedure_html="<p>Do the work.</p>"),
    ]
    out = flatten_plan(nodes, assertions_enabled=False)
    comment, proc = out["sections"]
    assert comment["section_type"] == 3
    assert comment["parent_temp_id"] is None
    assert "Objective" in comment["description"]
    assert "response_sets" not in comment
    assert out["counts"]["comments"] == 1
    assert out["preview"][0]["kind"] == "comment"


# ---------------------------------------------------------------------------
# response sets
# ---------------------------------------------------------------------------
def test_response_set_validation_drops_invalid():
    notes: list = []
    out = validate_response_sets(
        [
            {"type": "picklist", "placeholder": "Result", "options": ["Only one"]},
            {"type": "signature", "placeholder": "nope"},
            {"type": "multi_picklist", "placeholder": "Evidence",
             "options": ["Invoice", "GRN", "Invoice", ""]},
            {"type": "text", "placeholder": "<b>Ref</b> no."},
        ],
        notes, "node 1",
    )
    # picklist with 1 option and unknown type dropped; duplicates de-duped
    assert [r["type"] for r in out] == ["multi_picklist", "text"]
    assert out[0]["options"] == ["Invoice", "GRN"]
    assert out[1]["placeholder"] == "Ref no."   # tags stripped
    assert len(notes) == 2


# ---------------------------------------------------------------------------
# HTML sanitization
# ---------------------------------------------------------------------------
def test_sanitize_html_whitelist():
    dirty = (
        '<script>alert(1)</script><h1>Heading</h1>'
        '<p onclick="x()">Inspect <strong>contracts</strong></p>'
        '<img src=x><ol><li>Step</li></ol><!-- note -->'
    )
    clean = sanitize_html(dirty)
    assert "<script" not in clean and "alert" not in clean
    assert "<h1>" not in clean and "Heading" in clean       # tag stripped, text kept
    assert '<p>Inspect <strong>contracts</strong></p>' in clean  # attributes removed
    assert "<img" not in clean
    assert "<ol><li>Step</li></ol>" in clean
    assert "<!--" not in clean


# ---------------------------------------------------------------------------
# config summary
# ---------------------------------------------------------------------------
def test_summarize_wp_config():
    facts = summarize_wp_config({
        "working_paper": {
            "name": "Revenue", "type": 1,
            "config": {"settings": {
                "enable_assertions": True, "enable_signature": True,
                "response_sets": [{"type": "text_area", "placeholder": "Enter the response"}],
            }},
        },
        "items": [{"kind": "title", "text": "Opening balances"}, {"kind": "procedure", "question": "…"}],
    })
    assert facts["working_paper_name"] == "Revenue"
    assert facts["assertions_enabled"] is True
    assert facts["default_response_sets"] == [{"type": "text_area", "placeholder": "Enter the response"}]
    assert facts["existing_section_count"] == 2
    assert facts["existing_titles"] == ["Opening balances"]
    assert facts["is_empty"] is False


def test_summarize_wp_config_empty_wp():
    facts = summarize_wp_config({"working_paper": {"name": "Payroll", "config": None}, "items": []})
    assert facts["is_empty"] is True
    assert facts["assertions_enabled"] is False
    assert facts["default_response_sets"] == []


# ---------------------------------------------------------------------------
# the plan + write checkpoint
# ---------------------------------------------------------------------------
def test_build_plan_requires_working_paper_and_gates_the_write():
    agent = ProcedureBuildOutAgent()
    with pytest.raises(ValueError):
        agent.build_plan(_ctx())

    plan = agent.build_plan(_ctx(working_paper_id=77))
    write_steps = [p for p in plan if p.type == "write"]
    assert len(write_steps) == 1
    assert write_steps[0].tool == "bulk_create_program_sections"
    assert write_steps[0].requires_approval is True
    assert write_steps[0].args == {"working_paper_id": 77}
    # the WP read is scoped to the same working paper
    assert plan[0].tool == "get_working_paper"
    assert plan[0].args == {"working_paper_id": 77}
    # nothing before the checkpoint writes
    assert all(p.type != "write" for p in plan[:-1])


def test_prepare_write_payload_shape():
    agent = ProcedureBuildOutAgent()
    ctx = _ctx(working_paper_id=77)
    ctx.run_id = "1c9a2f34-0000-0000-0000-000000000000"
    flat = flatten_plan(_plan_tree(), assertions_enabled=True)
    ctx.results = [StepResult(7, "Draft", "analysis", "propose_procedures", {}, {
        "config_summary": "Default response is a text area; assertions are on.",
        "sections": flat["sections"], "preview": flat["preview"],
        "validation_notes": flat["notes"], "counts": flat["counts"],
    })]
    step = PlannedStep("Create the approved sections", "write",
                       "bulk_create_program_sections", {"working_paper_id": 77}, requires_approval=True)

    payload = agent.prepare_write(step, ctx)
    assert payload["working_paper_id"] == 77
    assert payload["ai_run_id"] == ctx.run_id
    assert payload["sections"] == flat["sections"]
    assert payload["config_summary"].startswith("Default response")
    assert payload["preview"] and payload["counts"]["total_sections"] == 4


def test_prepare_write_fails_loudly_without_a_draft():
    agent = ProcedureBuildOutAgent()
    step = PlannedStep("Create the approved sections", "write",
                       "bulk_create_program_sections", {"working_paper_id": 77}, requires_approval=True)

    with pytest.raises(ValueError):          # no draft at all
        agent.prepare_write(step, _ctx(working_paper_id=77))

    ctx = _ctx(working_paper_id=77)
    ctx.run_id = "abc123"
    ctx.results = [StepResult(7, "Draft", "analysis", "propose_procedures", {}, {"sections": []})]
    with pytest.raises(ValueError):          # empty draft
        agent.prepare_write(step, ctx)


def test_synthesize_reports_created_and_unmatched():
    agent = ProcedureBuildOutAgent()
    ctx = _ctx(working_paper_id=77)
    flat = flatten_plan(_plan_tree(), assertions_enabled=True)
    ctx.results = [
        StepResult(7, "Draft", "analysis", "propose_procedures", {}, {
            "config_summary": "cfg", "sections": flat["sections"], "preview": flat["preview"],
            "validation_notes": ["node 2: dropped picklist with fewer than 2 options"],
            "counts": flat["counts"],
        }),
        StepResult(8, "Write", "write", "bulk_create_program_sections", {}, {
            "count": 4, "unmatched_assertions": ["Occurrence"],
        }),
    ]
    out = agent.synthesize(ctx)
    assert out["sections_created"] == 4
    assert out["sections_proposed"] == 4
    assert out["response_type_breakdown"] == {"picklist": 1, "text_area": 2, "date": 1}
    assert any("Occurrence" in n for n in out["needs_attention"])
    assert any("dropped picklist" in n for n in out["needs_attention"])


def test_procedure_plan_schema_roundtrip():
    """The structured-output schema accepts the exact shape the prompt asks for."""
    plan = ProcedurePlan.model_validate({
        "config_summary": "Signature on; assertions off; default text area.",
        "nodes": [{
            "section_type": "title", "title": "Cut-off",
            "children": [{
                "procedure_html": "<p>Test cut-off.</p>",
                "response_sets": [{"type": "picklist", "options": ["Satisfactory", "N/A"]}],
                "rationale": "fixed verdict",
                "children": [{"procedure_html": "<p>Trace to GRN.</p>"}],
            }],
        }],
    })
    out = flatten_plan(plan.nodes, assertions_enabled=False)
    assert out["counts"]["total_sections"] == 3


# ---------------------------------------------------------------------------
# Grounding: client program adaptation — prior_year / current_year / structure
# (templates are deliberately never scanned)
# ---------------------------------------------------------------------------
from agent.definitions.procedure_buildout import (  # noqa: E402
    compact_prior_program,
    _MAX_PRIOR_ITEMS,
)
from prompts.procedure_plan import build_plan_prompt  # noqa: E402


def _prior_res(kind: str = "prior_year") -> dict:
    return {
        "source": {
            "kind": kind,
            "matched_by": "structure" if kind == "structure" else "reference",
            "audit_file": {
                "id": 41, "name": "ACME FY2025",
                "audit_period_end_date": "2025-12-31", "is_template": False,
                "is_current_file": kind == "structure",
            },
            "working_paper": {"id": 8871, "reference": "A.1", "name": "Inventory"},
            "section_count": 3,
            "truncated": False,
            "items": [
                {"kind": "title", "title": "General", "title_sl": "عام", "children": [
                    {"kind": "procedure",
                     "procedure_html": "<p>Attend the year-end inventory count.</p>",
                     "procedure_html_sl": "<p>حضور الجرد السنوي.</p>",
                     "assertions": ["Existence"],
                     "response_sets": [{"type": "picklist",
                                        "options": ["Completed no exceptions", "Completed with exceptions"]}],
                     "children": []},
                ]},
                {"kind": "comment", "description": "<p>Objective: obtain evidence over inventory.</p>"},
            ],
        },
        "checked": {"prior_files": 2, "current_files": 0},
    }


def test_build_plan_includes_prior_program_read():
    agent = ProcedureBuildOutAgent()
    plan = agent.build_plan(_ctx(working_paper_id=77))
    tools = [p.tool for p in plan]
    idx = tools.index("get_prior_program_sources")
    assert plan[idx].type == "read"
    assert plan[idx].args == {"working_paper_id": 77}
    assert plan[idx].requires_approval is False
    assert idx < tools.index("propose_procedures")


def test_build_plan_template_mode_skips_engagement_reads():
    """A template is a skeleton: no risks/plan/materiality/TB — those steps
    must not appear; the write stays approval-gated."""
    agent = ProcedureBuildOutAgent()
    plan = agent.build_plan(_ctx(working_paper_id=77, is_template=True))
    tools = [p.tool for p in plan]
    assert tools == [
        "get_working_paper", "get_prior_program_sources", "get_audit_file_summary",
        "summarize_config", "propose_procedures", "bulk_create_program_sections",
    ]
    titles = [p.title for p in plan]
    assert "Find reference programs across the firm" in titles
    assert "Draft the firm's standard program (AI)" in titles
    write = plan[-1]
    assert write.requires_approval is True
    assert write.args == {"working_paper_id": 77}


def test_build_plan_real_file_unchanged():
    """Regression: the default (non-template) plan keeps all engagement reads."""
    agent = ProcedureBuildOutAgent()
    plan = agent.build_plan(_ctx(working_paper_id=77))
    tools = [p.tool for p in plan]
    for t in ("get_risks", "get_audit_plan", "get_materiality", "gather_area_context"):
        assert t in tools
    assert len(plan) == 10


def test_compact_prior_program_none_cases():
    assert compact_prior_program(None, bilingual=True) is None
    assert compact_prior_program({"error": "boom"}, bilingual=True) is None
    assert compact_prior_program({"source": None, "checked": {}}, bilingual=True) is None
    empty = _prior_res()
    empty["source"]["items"] = []
    assert compact_prior_program(empty, bilingual=True) is None


def test_compact_prior_program_labels_and_bilingual_strip():
    prior = compact_prior_program(_prior_res(), bilingual=True)
    assert prior["kind"] == "prior_year"
    assert prior["label"] == "FY2025 A.1 Inventory (3 sections)"
    assert prior["matched_by"] == "reference"
    # nesting preserved, secondary language kept when bilingual
    title = prior["items"][0]
    assert title["title_sl"] == "عام"
    assert title["children"][0]["procedure_html_sl"].startswith("<p>حضور")

    mono = compact_prior_program(_prior_res(), bilingual=False)
    assert "title_sl" not in mono["items"][0]
    assert "procedure_html_sl" not in mono["items"][0]["children"][0]

    cur = compact_prior_program(_prior_res("current_year"), bilingual=False)
    assert cur["label"] == "FY2025 A.1 Inventory in this year's file 'ACME FY2025' (3 sections)"

    struct = compact_prior_program(_prior_res("structure"), bilingual=False)
    assert struct["label"] == "the structure of 'Inventory' in this file (3 sections)"

    peer = compact_prior_program(_prior_res("template_peer"), bilingual=False)
    assert peer["label"] == "the firm template 'ACME FY2025' — A.1 Inventory (3 sections)"

    practice = compact_prior_program(_prior_res("org_practice"), bilingual=False)
    assert practice["label"] == "FY2025 A.1 Inventory from engagement file 'ACME FY2025' (3 sections)"


def test_compact_prior_program_caps_items():
    res = _prior_res()
    res["source"]["items"] = [
        {"kind": "procedure", "procedure_html": f"<p>Step {i} of the audit program.</p>"}
        for i in range(_MAX_PRIOR_ITEMS + 20)
    ]
    prior = compact_prior_program(res, bilingual=False)
    assert len(prior["items"]) == _MAX_PRIOR_ITEMS
    assert prior["truncated"] is True


def _prompt(prior=None, style=None, template_mode=False):
    return build_plan_prompt(
        working_paper_name="Inventory", config_facts={"assertions_enabled": True},
        existing_items=[], file_summary={}, risks={}, audit_plan={}, materiality={},
        area_context={}, style_examples=style, language="en", prior_program=prior,
        template_mode=template_mode,
    )


def test_prompt_prior_year_block_and_demoted_examples():
    prior = compact_prior_program(_prior_res(), bilingual=False)
    p = _prompt(prior=prior, style=["<p>House example.</p>"])
    assert "PRIMARY SOURCE" in p
    assert "FY2025 A.1 Inventory (3 sections)" in p
    assert "NEVER copy prior-year amounts" in p
    assert "style reference only" in p            # demoted header
    assert "Attend the year-end inventory count" in p


def test_prompt_current_year_block():
    cur = compact_prior_program(_prior_res("current_year"), bilingual=False)
    p = _prompt(prior=cur)
    assert "PRIMARY SOURCE" in p
    assert "IN ANOTHER OF THIS YEAR'S FILES" in p


def test_prompt_template_mode_blocks():
    """Template mode: standard-program opener, comprehensive-assertion coverage,
    and NO engagement keys in DATA (absent, not null)."""
    p = _prompt(template_mode=True)
    assert "STANDARD audit program TEMPLATE" in p
    assert "copied into future audit files" in p
    assert "FULL standard assertion set" in p
    assert '"assessed_risks"' not in p
    assert '"materiality"' not in p
    assert '"area_accounts_and_testing"' not in p

    real = _prompt(template_mode=False)
    assert '"assessed_risks"' in real
    assert "STANDARD audit program TEMPLATE" not in real


def test_template_system_suffix_is_saudi_aware():
    from prompts.procedure_plan import TEMPLATE_SYSTEM_SUFFIX

    for token in ("SOCPA", "zakat", "VAT", "GOSI", "client-agnostic"):
        assert token in TEMPLATE_SYSTEM_SUFFIX
    assert "Never force local content into unrelated areas" in TEMPLATE_SYSTEM_SUFFIX


def test_prompt_template_peer_and_org_practice_heads():
    peer = compact_prior_program(_prior_res("template_peer"), bilingual=False)
    p = _prompt(prior=peer, template_mode=True)
    assert "THE FIRM'S EXISTING TEMPLATE FOR THE SAME" in p

    practice = compact_prior_program(_prior_res("org_practice"), bilingual=False)
    p2 = _prompt(prior=practice, template_mode=True)
    assert "HOW THIS FIRM ACTUALLY PERFORMS THIS AREA" in p2
    assert "STRIP every client- or period-specific detail" in p2


def test_prompt_structure_block_and_no_source():
    struct = compact_prior_program(_prior_res("structure"), bilingual=False)
    p = _prompt(prior=struct)
    assert "STRUCTURE REFERENCE" in p and "PRIMARY SOURCE" not in p
    assert "Do NOT copy its subject matter" in p

    p2 = _prompt(prior=None, style=["<p>House example.</p>"])
    assert "PRIMARY SOURCE" not in p2 and "STRUCTURE REFERENCE" not in p2
    assert "Examples of how THIS firm writes procedures" in p2   # original header


def test_prepare_write_and_synthesize_carry_grounding():
    agent = ProcedureBuildOutAgent()
    ctx = _ctx(working_paper_id=77)
    ctx.run_id = "1c9a2f34-0000-0000-0000-000000000000"
    flat = flatten_plan(_plan_tree(), assertions_enabled=True)
    grounding = {
        "source": {"kind": "prior_year", "label": "FY2025 A.1 Inventory (3 sections)",
                   "matched_by": "reference", "audit_file_id": 41,
                   "working_paper_id": 8871, "section_count": 3},
        "style_examples": 2,
    }
    draft = {
        "config_summary": "cfg", "sections": flat["sections"], "preview": flat["preview"],
        "validation_notes": [], "counts": flat["counts"], "grounding": grounding,
    }
    ctx.results = [StepResult(7, "Draft", "analysis", "propose_procedures", {}, draft)]
    step = PlannedStep("Create the approved sections", "write",
                       "bulk_create_program_sections", {"working_paper_id": 77}, requires_approval=True)
    payload = agent.prepare_write(step, ctx)
    assert payload["grounding"] == grounding

    ctx.results.append(StepResult(8, "Write", "write", "bulk_create_program_sections", {}, {"count": 4}))
    out = agent.synthesize(ctx)
    assert out["summary"].startswith("Grounded on FY2025 A.1 Inventory")


def test_synthesize_flags_missing_grounding_source():
    agent = ProcedureBuildOutAgent()
    ctx = _ctx(working_paper_id=77)
    flat = flatten_plan(_plan_tree(), assertions_enabled=True)
    ctx.results = [
        StepResult(7, "Draft", "analysis", "propose_procedures", {}, {
            "config_summary": "cfg", "sections": flat["sections"], "preview": flat["preview"],
            "validation_notes": [], "counts": flat["counts"],
            "grounding": {"source": None, "style_examples": 0},
        }),
        StepResult(8, "Write", "write", "bulk_create_program_sections", {}, {"count": 4}),
    ]
    out = agent.synthesize(ctx)
    assert not out["summary"].startswith("Grounded on")
    assert any("no earlier program of this client" in n for n in out["needs_attention"])

    ctx.is_template = True
    out_t = agent.synthesize(ctx)
    assert any("no reference program of the firm" in n for n in out_t["needs_attention"])
