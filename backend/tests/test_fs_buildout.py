"""Unit tests for the Financial-Statement Configurator agent (offline — no
Bedrock, no be).

The LLM call itself is not exercised; we test the pure-Python structure
discipline — criteria building (families, 'in'-only, regenerated patterns),
formula token assembly (FR refs, cycles, cascade drops, no leading minus),
TipTap name wrapping, coverage advisories, the write-checkpoint payload, and
the router's snapshot-swap undo/redo handlers. These are the guarantees that
keep a hallucinated mapping out of the firm's template.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.definitions.fs_buildout import (
    FsBuildOutAgent,
    FsChildRow,
    FsCriterion,
    FsFormulaTerm,
    FsPlan,
    FsRow,
    FsTopRow,
    build_coa_index,
    build_criteria,
    compact_fs_reference,
    coverage_check,
    flatten_fs_plan,
    wrap_name,
)
from agent.types import PlannedStep, RunContext, StepResult
from copilot_tools import CopilotContext


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ctx(**kwargs) -> RunContext:
    return RunContext(copilot=CopilotContext(0, ""), audit_file_id=270, **kwargs)


def _coa_tree() -> dict:
    return {
        "statement": "balance_sheet",
        "accounts": [
            {"id": 1, "code": "1", "name": "Assets", "name_sl": "الأصول", "depth": 1, "parent": None, "group": 1, "type": None},
            {"id": 2, "code": "1.1", "name": "Receivables", "name_sl": "الذمم", "depth": 2, "parent": 1, "group": 1, "type": 2},
            {"id": 69, "code": "1.2", "name": "Cash", "name_sl": "النقد", "depth": 2, "parent": 1, "group": 1, "type": 69},
            {"id": 145, "code": "1.1.1", "name": "Trade receivables", "name_sl": "ذمم تجارية", "depth": 3, "parent": 2, "group": 1, "type": 2},
            {"id": 146, "code": "1.2.1", "name": "Cash at banks", "name_sl": "نقد لدى البنوك", "depth": 3, "parent": 69, "group": 1, "type": 69},
        ],
        "groups": [{"id": 1, "code": "1", "name": "Assets", "name_sl": "الأصول"}],
        "types": [
            {"id": 2, "code": "1.1", "name": "Receivables", "name_sl": "الذمم", "group": 1},
            {"id": 69, "code": "1.2", "name": "Cash", "name_sl": "النقد", "group": 1},
        ],
        "count": 5,
        "truncated": False,
    }


def _index() -> dict:
    return build_coa_index(_coa_tree())


def _flatten(rows, *, bilingual=False):
    return flatten_fs_plan(
        rows, coa_index=_index(), bilingual=bilingual,
        language="en", secondary_language="ar" if bilingual else None,
    )


# ---------------------------------------------------------------------------
# wrap_name
# ---------------------------------------------------------------------------
def test_wrap_name_ltr_rtl_and_escape():
    assert wrap_name("Current assets", "en") == '<p dir="ltr">Current assets</p>'
    assert wrap_name("الأصول المتداولة", "ar") == '<p dir="rtl">الأصول المتداولة</p>'
    assert wrap_name("<b>x</b> & y", "en") == '<p dir="ltr">&lt;b&gt;x&lt;/b&gt; &amp; y</p>'
    assert wrap_name("   ", "en") == ""


# ---------------------------------------------------------------------------
# build_criteria
# ---------------------------------------------------------------------------
def test_build_criteria_single_and_multi_patterns():
    notes: list = []
    single = build_criteria([FsCriterion(criteria_type="account_type", account_ids=[2, 69])], "AND", _index(), notes, "r")
    assert single["account_criteria_pattern"] == "1"
    (field,) = single["account_criteria"]
    assert field["type"] == "field"
    assert field["comparison_operator"] == "in"
    assert field["display_coa_depth"] == 2
    assert field["value"] == [2, 69]
    assert field["position"] == 1

    multi = build_criteria(
        [
            FsCriterion(criteria_type="account_group", account_ids=[1]),
            FsCriterion(criteria_type="account", account_ids=[145]),
        ],
        "OR", _index(), notes, "r",
    )
    assert multi["account_criteria_pattern"] == "( 1 OR 2 )"
    kinds = [c["type"] for c in multi["account_criteria"]]
    assert kinds == ["field", "operator", "field"]
    assert multi["account_criteria"][1]["value"] == "OR"
    assert multi["account_criteria"][2]["display_coa_depth"] == 3


def test_build_criteria_drops_wrong_family_ids_and_empty_rows():
    notes: list = []
    # 145 is an account (depth 3), not a type — dropped with a note
    built = build_criteria([FsCriterion(criteria_type="account_type", account_ids=[145, 2])], "AND", _index(), notes, "r")
    assert built["account_criteria"][0]["value"] == [2]
    assert any("145" in n for n in notes)
    # nothing valid at all -> None (the row is dropped by the flattener)
    assert build_criteria([FsCriterion(criteria_type="account_group", account_ids=[999])], "AND", _index(), notes, "r") is None


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------
def _bs_tree() -> list:
    return [
        FsTopRow(
            temp_id="r1", row_type="text_field", name="ASSETS", name_sl="الأصول",
            children=[
                FsRow(
                    temp_id="r2", row_type="account_group", name="Current assets", name_sl="الأصول المتداولة",
                    criteria=[FsCriterion(criteria_type="account_type", account_ids=[2, 69])],
                    rationale="whole current-asset types",
                ),
                FsRow(
                    temp_id="r3", row_type="formula_row", name="Total current assets",
                    formula_terms=[FsFormulaTerm(sign="+", account_id=145), FsFormulaTerm(sign="+", account_id=146)],
                ),
            ],
        ),
        FsTopRow(
            temp_id="r4", row_type="formula_row", name="TOTAL ASSETS",
            formula_terms=[FsFormulaTerm(sign="+", row_temp_id="r3")],
        ),
    ]


def test_flatten_builds_exact_db_rows():
    flat = _flatten(_bs_tree())
    rows = flat["rows"]
    assert [r["row_type"] for r in rows] == ["text_field", "formula_row"]
    heading = rows[0]
    assert heading["name"] == '<p dir="ltr">ASSETS</p>'
    assert "name_sl" not in heading  # single-language draft strips _sl
    group, total_current = heading["child_row"]
    assert group["account_criteria_pattern"] == "1"
    assert group["account_criteria"][0]["value"] == [2, 69]
    assert group["reverse"] is False
    assert total_current["formula"] == "A145+A146"
    assert total_current["reference"].startswith("FR") and len(total_current["reference"]) == 8
    total_assets = rows[1]
    assert total_assets["formula"] == total_current["reference"]  # temp-id ref resolved
    # every row got a uuid uniqueId
    assert all(len(r["uniqueId"]) == 36 for r in [heading, group, total_current, total_assets])
    assert flat["counts"] == {"total_rows": 4, "headings": 1, "account_groups": 1, "formula_rows": 2}


def test_flatten_bilingual_wraps_both_languages():
    flat = _flatten(_bs_tree(), bilingual=True)
    heading = flat["rows"][0]
    assert heading["name_sl"] == '<p dir="rtl">الأصول</p>'
    preview_heading = flat["preview"][0]
    assert preview_heading["text_sl"] == "الأصول"


def test_flatten_forward_reference_resolves():
    rows = [
        FsTopRow(temp_id="a", row_type="formula_row", name="Grand total",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="b")]),
        FsTopRow(temp_id="b", row_type="formula_row", name="Subtotal",
                 formula_terms=[FsFormulaTerm(sign="+", account_id=145)]),
    ]
    flat = _flatten(rows)
    grand, sub = flat["rows"]
    assert grand["formula"] == sub["reference"]


def test_flatten_drops_dangling_term_but_keeps_row():
    rows = [
        FsTopRow(temp_id="a", row_type="formula_row", name="Total",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="nope"),
                                FsFormulaTerm(sign="+", account_id=145)]),
    ]
    flat = _flatten(rows)
    assert flat["rows"][0]["formula"] == "A145"
    assert any("not a kept total" in n for n in flat["notes"])


def test_flatten_cycle_drops_both_rows():
    rows = [
        FsTopRow(temp_id="a", row_type="formula_row", name="A",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="b")]),
        FsTopRow(temp_id="b", row_type="formula_row", name="B",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="a")]),
    ]
    flat = _flatten(rows)
    assert flat["rows"] == []
    assert sum("loop" in n for n in flat["notes"]) == 2


def test_flatten_empty_formula_cascades_to_referencing_rows():
    rows = [
        FsTopRow(temp_id="a", row_type="formula_row", name="Outer",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="b")]),
        FsTopRow(temp_id="b", row_type="formula_row", name="Inner",
                 formula_terms=[FsFormulaTerm(sign="+", account_id=999)]),  # unknown id
    ]
    flat = _flatten(rows)
    assert flat["rows"] == []  # inner dropped -> outer's only term dangles -> outer dropped


def test_flatten_negative_only_formula_dropped_and_mixed_reordered():
    neg = _flatten([FsTopRow(temp_id="a", row_type="formula_row", name="Neg",
                             formula_terms=[FsFormulaTerm(sign="-", account_id=145)])])
    assert neg["rows"] == []
    assert any("only negative terms" in n for n in neg["notes"])
    mixed = _flatten([FsTopRow(temp_id="a", row_type="formula_row", name="Mixed",
                               formula_terms=[FsFormulaTerm(sign="-", account_id=145),
                                              FsFormulaTerm(sign="+", account_id=146)])])
    assert mixed["rows"][0]["formula"] == "A146-A145"  # positive term leads


def test_flatten_drops_group_without_valid_mapping_with_children():
    rows = [
        FsTopRow(
            temp_id="g", row_type="account_group", name="Ghost",
            criteria=[FsCriterion(criteria_type="account", account_ids=[999])],
            children=[FsRow(temp_id="c", row_type="text_field", name="Child")],
        ),
    ]
    flat = _flatten(rows)
    assert flat["rows"] == []
    assert any("no valid account mapping" in n for n in flat["notes"])
    assert any("child row(s) were dropped with it" in n for n in flat["notes"])


def test_flatten_preview_labels():
    flat = _flatten(_bs_tree())
    heading = flat["preview"][0]
    group, total = heading["children"]
    assert group["kind"] == "group"
    assert group["criteria_labels"] == ["account type in: Receivables, Cash"]
    # structured form: BOTH org languages, the FE shows the one matching the
    # CURRENT UI language and translates the connective itself
    assert group["criteria_view"] == [{
        "type": "account_type",
        "names": ["Receivables", "Cash"],
        "names_sl": ["الذمم", "النقد"],
        "more": False,
    }]
    assert total["kind"] == "total"
    assert total["formula_label"] == "Trade receivables + Cash at banks"
    assert total["formula_label_sl"] == "ذمم تجارية + نقد لدى البنوك"


def test_flatten_helper_labels_emit_both_languages():
    # COA names are English, name_sl Arabic here. Labels are emitted in BOTH
    # languages so the checkpoint stays correct even after a UI language
    # switch on an already-drafted run.
    flat = flatten_fs_plan(
        _bs_tree(), coa_index=_index(), bilingual=True,
        language="en", secondary_language="ar",
    )
    heading = flat["preview"][0]
    group, total = heading["children"]
    assert group["criteria_view"][0]["names"] == ["Receivables", "Cash"]
    assert group["criteria_view"][0]["names_sl"] == ["الذمم", "النقد"]
    assert total["formula_label"] == "Trade receivables + Cash at banks"
    assert total["formula_label_sl"] == "ذمم تجارية + نقد لدى البنوك"
    # a row-reference term carries the referenced row's name in both languages
    ref_tree = [
        FsTopRow(temp_id="sub", row_type="formula_row", name="Subtotal", name_sl="مجموع فرعي",
                 formula_terms=[FsFormulaTerm(sign="+", account_id=145)]),
        FsTopRow(temp_id="grand", row_type="formula_row", name="Grand total", name_sl="المجموع الكلي",
                 formula_terms=[FsFormulaTerm(sign="+", row_temp_id="sub")]),
    ]
    ref_flat = flatten_fs_plan(
        ref_tree, coa_index=_index(), bilingual=True,
        language="en", secondary_language="ar",
    )
    assert ref_flat["preview"][1]["formula_label"] == "Subtotal"
    assert ref_flat["preview"][1]["formula_label_sl"] == "مجموع فرعي"


def test_flatten_notes_language_arabic():
    # validation notes are written once — in the RUNNING user's UI language
    rows = [FsTopRow(
        temp_id="g", row_type="account_group", name="Ghost",
        criteria=[FsCriterion(criteria_type="account", account_ids=[999])],
    )]
    ar = flatten_fs_plan(
        rows, coa_index=_index(), bilingual=False,
        language="en", secondary_language=None, notes_language="ar",
    )
    assert ar["rows"] == []
    assert any("حُذفت المجموعة" in n for n in ar["notes"])
    en = flatten_fs_plan(
        rows, coa_index=_index(), bilingual=False,
        language="en", secondary_language=None, notes_language="en",
    )
    assert any("no valid account mapping" in n for n in en["notes"])


# ---------------------------------------------------------------------------
# coverage_check (advisory)
# ---------------------------------------------------------------------------
def test_coverage_full_via_group_criterion():
    flat = _flatten([FsTopRow(temp_id="g", row_type="account_group", name="All assets",
                              criteria=[FsCriterion(criteria_type="account_group", account_ids=[1])])])
    assert coverage_check(flat["rows"], _coa_tree()) == []


def test_coverage_flags_uncovered_and_doubled():
    flat = _flatten([
        FsTopRow(temp_id="g1", row_type="account_group", name="Receivables",
                 criteria=[FsCriterion(criteria_type="account_type", account_ids=[2])]),
        FsTopRow(temp_id="g2", row_type="account_group", name="Receivables again",
                 criteria=[FsCriterion(criteria_type="account_type", account_ids=[2])]),
    ])
    notes = coverage_check(flat["rows"], _coa_tree())
    assert any("not captured" in n and "Cash" in n for n in notes)
    assert any("MORE THAN ONE" in n and "Receivables" in n for n in notes)


# ---------------------------------------------------------------------------
# compact_fs_reference
# ---------------------------------------------------------------------------
def test_compact_fs_reference_none_cases_and_labels():
    assert compact_fs_reference(None) is None
    assert compact_fs_reference({"error": "x"}) is None
    assert compact_fs_reference({"source": None}) is None
    assert compact_fs_reference({"source": {"kind": "template_peer", "rows": []}}) is None
    peer = compact_fs_reference({"source": {
        "kind": "template_peer", "rows": [{"name": "Assets"}], "row_count": 12,
        "audit_file": {"id": 7, "name": "Master template"},
    }})
    assert peer["label"] == "the firm template 'Master template' (12 rows)"
    practice = compact_fs_reference({"source": {
        "kind": "org_practice", "rows": [{"name": "Assets"}], "row_count": 30,
        "audit_file": {"id": 8, "name": "ACME FY2025"},
    }})
    assert practice["label"] == "engagement file 'ACME FY2025' (30 rows)"


# ---------------------------------------------------------------------------
# the agent: plan / gates / checkpoint / synthesize
# ---------------------------------------------------------------------------
def test_build_plan_requires_wp_and_report_type_and_gates_the_write():
    agent = FsBuildOutAgent()
    with pytest.raises(ValueError):
        agent.build_plan(_ctx())
    with pytest.raises(ValueError):
        agent.build_plan(_ctx(working_paper_id=9, report_type="cash_flow"))
    plan = agent.build_plan(_ctx(working_paper_id=9, report_type="balance_sheet", is_template=True))
    assert [s.tool for s in plan] == [
        "get_audit_file_summary", "get_fs_config", "get_fs_coa_tree",
        "get_fs_reference_configs", "verify_template_and_summarize",
        "propose_fs_structure", "apply_fs_config",
    ]
    assert plan[1].args == {"working_paper_id": 9, "report_type": "balance_sheet"}
    write = plan[-1]
    assert write.type == "write" and write.requires_approval
    assert write.args["report_type"] == "balance_sheet"
    assert all(not s.requires_approval for s in plan[:-1])


def _results(summary=None, fs=None, coa=None) -> list:
    out = []
    if summary is not None:
        out.append(StepResult(0, "s", "read", "get_audit_file_summary", {}, summary))
    if fs is not None:
        out.append(StepResult(1, "f", "read", "get_fs_config", {}, fs))
    if coa is not None:
        out.append(StepResult(2, "c", "read", "get_fs_coa_tree", {}, coa))
    return out


def test_verify_step_gates_non_templates_and_empty_coa():
    agent = FsBuildOutAgent()
    fs = {"report_type": "balance_sheet", "row_count": 0, "working_paper": {"name": "BS"},
          "languages": {"primary": "en", "secondary": "ar", "use_secondary": True}}
    step = PlannedStep("v", "compute", "verify_template_and_summarize")
    with pytest.raises(ValueError, match="TEMPLATES only"):
        agent.execute_step(step, _ctx(results=_results({"is_template": False}, fs, _coa_tree())))
    with pytest.raises(ValueError, match="chart of accounts"):
        agent.execute_step(step, _ctx(results=_results({"is_template": True}, fs, {"accounts": []})))
    facts = agent.execute_step(step, _ctx(results=_results({"is_template": True}, fs, _coa_tree())))
    assert facts["bilingual"] is True
    assert facts["secondary_language"] == "ar"
    assert facts["replaces_existing"] is False


def test_prepare_write_payload_shape_and_loud_failures():
    agent = FsBuildOutAgent()
    draft = {
        "rows": [{"uniqueId": "u", "row_type": "text_field", "name": "<p>x</p>", "child_row": []}],
        "preview": [{"kind": "heading"}], "structure_summary": "s", "validation_notes": ["n"],
        "counts": {"total_rows": 1}, "replaces_existing": True, "previous_row_count": 3,
        "grounding": {"source": {"kind": "template_peer", "label": "L"}},
    }
    ctx = _ctx(working_paper_id=9, results=[StepResult(5, "d", "analysis", "propose_fs_structure", {}, draft)])
    ctx.run_id = "11111111-2222-3333-4444-555555555555"
    step = PlannedStep("w", "write", "apply_fs_config", {"working_paper_id": 9, "report_type": "balance_sheet"}, True)
    payload = agent.prepare_write(step, ctx)
    assert payload["working_paper_id"] == 9
    assert payload["report_type"] == "balance_sheet"
    assert payload["ai_run_id"] == ctx.run_id
    assert payload["rows"] == draft["rows"]
    assert payload["replaces_existing"] is True
    assert payload["previous_row_count"] == 3
    assert payload["grounding"]["source"]["label"] == "L"

    with pytest.raises(ValueError):  # no draft at all
        agent.prepare_write(step, _ctx(working_paper_id=9))
    empty = _ctx(working_paper_id=9, results=[StepResult(5, "d", "analysis", "propose_fs_structure", {}, {"rows": []})])
    empty.run_id = ctx.run_id
    with pytest.raises(ValueError):  # empty draft
        agent.prepare_write(step, empty)


def test_synthesize_created_rejected_and_alias():
    agent = FsBuildOutAgent()
    draft = {
        "structure_summary": "s", "counts": {"total_rows": 4, "headings": 1, "account_groups": 2, "formula_rows": 1},
        "validation_notes": [], "grounding": {"source": {"kind": "template_peer", "label": "the firm template 'M' (12 rows)"}},
    }
    write = {"applied_count": 4, "previous_row_count": 3}
    ctx = _ctx(results=[
        StepResult(5, "d", "analysis", "propose_fs_structure", {}, draft),
        StepResult(6, "w", "write", "apply_fs_config", {}, write),
    ])
    out = agent.synthesize(ctx)
    assert out["rows_created"] == 4
    assert out["sections_created"] == 4  # history-panel alias
    assert out["replaced_previous_rows"] == 3
    assert "Grounded on the firm template 'M'" in out["summary"]
    assert "replaced the previous 3-row structure" in out["summary"]

    rej = _ctx(results=[
        StepResult(5, "d", "analysis", "propose_fs_structure", {}, {**draft, "grounding": {"source": None}}),
        StepResult(6, "w", "write", "apply_fs_config", {}, {"rejected": True}),
    ])
    out = agent.synthesize(rej)
    assert out["rows_created"] == 0
    assert any("rejected" in n for n in out["needs_attention"])
    assert any("no existing statement configuration" in n for n in out["needs_attention"])


def test_synthesize_arabic_report_language():
    # the run's final report is written in the RUNNING user's UI language
    agent = FsBuildOutAgent()
    draft = {
        "structure_summary": "s", "counts": {"total_rows": 4, "headings": 1, "account_groups": 2, "formula_rows": 1},
        "validation_notes": [],
        "grounding": {"source": {"kind": "template_peer", "label": "the firm template 'M' (12 rows)", "file_name": "M"}},
    }
    write = {"applied_count": 4, "previous_row_count": 3}
    ctx = _ctx(language="ar", results=[
        StepResult(5, "d", "analysis", "propose_fs_structure", {}, draft),
        StepResult(6, "w", "write", "apply_fs_config", {}, write),
    ])
    out = agent.synthesize(ctx)
    assert "تم إعداد القائمة بـ 4 صفًا" in out["summary"]
    assert "استنادًا إلى قالب المنشأة «M»" in out["summary"]
    assert "حلّ محل الهيكل السابق" in out["summary"]

    rej = _ctx(language="ar", results=[
        StepResult(5, "d", "analysis", "propose_fs_structure", {}, {**draft, "grounding": {"source": None}}),
        StepResult(6, "w", "write", "apply_fs_config", {}, {"rejected": True}),
    ])
    out = agent.synthesize(rej)
    assert "رُفض الهيكل المقترح" in out["summary"]
    assert any("رُفضت المسودة" in n for n in out["needs_attention"])
    assert any("لم يُعثر على إعداد سابق" in n for n in out["needs_attention"])


def test_language_directive_and_step_action_language():
    import schemas
    from agent.types import language_directive

    assert "Arabic" in language_directive("ar")
    assert "English" in language_directive("en")
    assert "English" in language_directive(None)  # safe default
    # data values must never be translated
    assert "exactly as they appear" in language_directive("ar")

    req = schemas.AgentStepActionRequest(session_token="t")
    assert req.language == "en"
    req = schemas.AgentStepActionRequest(session_token="t", language="ar")
    assert req.language == "ar"


def test_fs_plan_schema_roundtrip():
    plan = FsPlan(structure_summary="ok", rows=[
        FsTopRow(temp_id="r1", row_type="account_group", name="X",
                 criteria=[FsCriterion(criteria_type="account_type", account_ids=[2])],
                 children=[FsRow(temp_id="r2", row_type="formula_row", name="T",
                                 formula_terms=[FsFormulaTerm(sign="+", account_id=145)],
                                 children=[FsChildRow(temp_id="r3", row_type="text_field", name="N")])]),
    ])
    again = FsPlan.model_validate(plan.model_dump())
    assert again.rows[0].children[0].children[0].name == "N"


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
def test_prompt_conventions_reference_and_language_blocks():
    from prompts.fs_plan import SYSTEM_PROMPT, build_fs_plan_prompt

    assert "SECURITY" in SYSTEM_PROMPT
    assert "SOCPA" in SYSTEM_PROMPT
    assert "COVERAGE RULE" in SYSTEM_PROMPT

    base = dict(file_summary={"name": "T"}, coa_tree=_coa_tree())
    bs = build_fs_plan_prompt(statement_type="balance_sheet", language="en", secondary_language="ar", **base)
    assert "BALANCE SHEET CONVENTIONS" in bs
    assert "name_sl in Arabic" in bs
    assert "No existing configuration was found" in bs

    is_ = build_fs_plan_prompt(statement_type="income_statement", language="en", **base)
    assert "INCOME STATEMENT CONVENTIONS" in is_
    assert "leave name_sl empty" in is_

    grounded = build_fs_plan_prompt(
        statement_type="balance_sheet", language="en",
        reference={"label": "the firm template 'M' (12 rows)", "rows": [{"name": "Assets"}]},
        existing_summary=[{"name": "Old row"}], existing_row_count=5, **base,
    )
    assert "PRIMARY SOURCE" in grounded
    assert "the firm template 'M' (12 rows)" in grounded
    assert "never copy an id" in grounded
    assert "ALREADY CONFIGURED with 5 row(s)" in grounded

    # rationale + structure_summary follow the REVIEWER's language, names the
    # org's own convention: Arabic-primary org reviewed in English mode
    rev = build_fs_plan_prompt(
        statement_type="balance_sheet", language="ar", secondary_language="en",
        reviewer_language="en", **base,
    )
    assert "name in Arabic" in rev
    assert "rationale AND the structure_summary in English" in rev
    # no reviewer given -> falls back to the primary language
    rev_default = build_fs_plan_prompt(statement_type="balance_sheet", language="ar", secondary_language="en", **base)
    assert "rationale AND the structure_summary in Arabic" in rev_default


# ---------------------------------------------------------------------------
# router: request schema, history derivation, snapshot undo/redo handlers
# ---------------------------------------------------------------------------
def test_agent_run_request_accepts_report_type():
    import schemas

    req = schemas.AgentRunRequest(
        session_token="t", audit_file_id=1, copilot_grant="g",
        agent_type="fs_buildout", is_template=True, report_type="balance_sheet",
    )
    assert req.report_type == "balance_sheet"


def test_history_row_can_redo_for_fs_snapshot():
    from routers.agent import _history_row

    def run(**over):
        base = dict(
            id="r", status="done", agent_type="fs_buildout", working_paper_id=9,
            is_template=True, goal="g", created_at=None, created_by="u",
            result_summary={"sections_created": 4, "summary": "s", **over.pop("rs", {})},
            dismissed=False,
        )
        base.update(over)
        return SimpleNamespace(**base)

    live = _history_row(run())
    assert live["can_undo"] and not live["can_redo"]
    undone = _history_row(run(rs={"undone": True, "undone_count": 4, "can_redo_fs": True}))
    assert not undone["can_undo"] and undone["can_redo"]
    # procedure derivation unchanged: ids present -> redo, absent -> no redo
    proc = _history_row(run(agent_type="procedure_buildout", rs={"undone": True, "undone_section_ids": [1]}))
    assert proc["can_redo"]
    proc_no_ids = _history_row(run(agent_type="procedure_buildout", rs={"undone": True}))
    assert not proc_no_ids["can_redo"]


def _fake_db():
    return SimpleNamespace(commit=_async_noop)


async def _async_noop(*a, **k):
    return None


def _fs_run(rs=None):
    return SimpleNamespace(
        id="11111111-2222-3333-4444-555555555555",
        agent_type="fs_buildout",
        working_paper_id=9,
        result_summary=dict(rs or {"sections_created": 4}),
    )


def test_undo_fs_swaps_snapshot_and_records_rows_at_undo(monkeypatch):
    import routers.agent as agent_router

    step = SimpleNamespace(
        output={"working_paper_id": 9, "report_type": "balance_sheet",
                "previous_rows": [{"old": 1}], "previous_row_count": 1},
        approved_payload={"rows": [{"ai": 1}, {"ai": 2}]},
        input={"working_paper_id": 9, "report_type": "balance_sheet"},
    )

    async def fake_find(db, run, tool):
        assert tool == "apply_fs_config"
        return step

    calls = []

    def fake_tool(payload):
        calls.append(payload)
        # the restore's response returns what it just replaced (AI rows + tweaks)
        return {"applied_count": 1, "previous_row_count": 2,
                "previous_rows": [{"ai": 1}, {"ai": 2, "tweaked": True}]}

    monkeypatch.setattr(agent_router, "_find_write_step", fake_find)
    monkeypatch.setattr(agent_router.REGISTRY, "impls_for", lambda ctx, names: {"apply_fs_config": fake_tool})

    run = _fs_run()
    out = asyncio.run(agent_router._undo_fs(_fake_db(), run, object()))
    assert out["undone"] and out["deleted_count"] == 2
    assert calls[0]["rows"] == [{"old": 1}]
    assert calls[0]["restore"] is True
    assert run.result_summary["undone"] is True
    assert run.result_summary["rows_at_undo"] == [{"ai": 1}, {"ai": 2, "tweaked": True}]
    assert run.result_summary["can_redo_fs"] is True

    # a second undo must 409 (another swap would corrupt the redo snapshot)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        asyncio.run(agent_router._undo_fs(_fake_db(), run, object()))
    assert e.value.status_code == 409


def test_redo_fs_reposts_rows_at_undo(monkeypatch):
    import routers.agent as agent_router

    step = SimpleNamespace(
        output={"working_paper_id": 9, "report_type": "balance_sheet", "previous_rows": []},
        approved_payload={"rows": [{"ai": 1}]},
        input={},
    )

    async def fake_find(db, run, tool):
        return step

    calls = []

    def fake_tool(payload):
        calls.append(payload)
        return {"applied_count": 2, "previous_row_count": 1, "previous_rows": [{"old": 1}]}

    monkeypatch.setattr(agent_router, "_find_write_step", fake_find)
    monkeypatch.setattr(agent_router.REGISTRY, "impls_for", lambda ctx, names: {"apply_fs_config": fake_tool})

    run = _fs_run(rs={"sections_created": 4, "undone": True, "can_redo_fs": True,
                      "rows_at_undo": [{"ai": 1}, {"ai": 2, "tweaked": True}]})
    out = asyncio.run(agent_router._redo_fs(_fake_db(), run, object()))
    assert out["restored"] and out["restored_count"] == 2
    assert calls[0]["rows"] == [{"ai": 1}, {"ai": 2, "tweaked": True}]  # tweaks come back
    assert calls[0]["restore"] is True
    assert run.result_summary["undone"] is False

    # redo without an undo must 409
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        asyncio.run(agent_router._redo_fs(_fake_db(), _fs_run(), object()))
    assert e.value.status_code == 409
