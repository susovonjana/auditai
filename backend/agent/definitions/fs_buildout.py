"""
Financial-Statement Configurator agent (agent_type = "fs_buildout").

Goal: "Configure ONE statement (balance sheet / income statement) inside an
AUDIT FILE TEMPLATE." It reads the statement's current configuration, the
template's chart of accounts (the only legal id pool) and a reference config
from another of the firm's templates/files, makes ONE structured LLM call that
drafts the full presentation structure — headings, account groups mapped by
criteria, formula subtotals, bilingual captions — and pauses at an APPROVAL
checkpoint. Only after the auditor approves does 1audit-be REPLACE the
statement's financial_reports_config rows in one transaction. TEMPLATES ONLY:
files created from the template inherit the rows via the existing copy flow.

Safe write policy: REPLACE-WITH-SNAPSHOT. The be apply endpoint returns the
rows as they stood the instant before the write (previous_rows, read inside the
same transaction); the router stores them so undo/redo is a snapshot swap — no
stamping, nothing is ever irrecoverably lost.

Structure discipline (mirror of procedure_buildout): the LLM proposes captions,
grouping and mapping INTENT only. Everything enforceable is validated or built
in PURE PYTHON — criteria families checked against the COA read, only 'in'
comparisons, patterns/positions regenerated, formula token strings assembled
from structured terms (never model-written), cycles rejected, size caps, TipTap
name wrapping. The be endpoint re-validates all of it against the DB.
"""
from __future__ import annotations

import html as _html
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from prompts.fs_plan import SYSTEM_PROMPT, build_fs_plan_prompt
from structured import generate_structured


REPORT_TYPES = ("balance_sheet", "income_statement")


# ---------------------------------------------------------------------------
# LLM output schema. Depth is bounded STRUCTURALLY (top -> row -> child) like
# procedure_buildout's section/step/sub-step, so depth<=3 holds by construction.
# Formulas are STRUCTURED TERMS (sign + account_id | row_temp_id) — the token
# string the product stores is assembled in Python, never written by the model.
# ---------------------------------------------------------------------------
class FsCriterion(BaseModel):
    criteria_type: Literal["account_group", "account_type", "account"]
    account_ids: List[int] = Field(
        default_factory=list,
        description="coa_original_ids FROM THE CHART OF ACCOUNTS DATA at the family's depth (group=1, type=2, account=3+)",
    )


class FsFormulaTerm(BaseModel):
    sign: Literal["+", "-"] = "+"
    account_id: Optional[int] = Field(default=None, description="an account's id from the CHART OF ACCOUNTS DATA")
    row_temp_id: Optional[str] = Field(default=None, description="the temp_id of ANOTHER formula_row in this draft")


class _FsRowBase(BaseModel):
    temp_id: str = Field(description="unique short id (r1, r2, …) — formula terms reference rows by it")
    row_type: Literal["text_field", "account_group", "formula_row"]
    name: str = Field(description="the caption, PLAIN TEXT, in the PRIMARY language")
    name_sl: str = Field(default="", description="the caption in the SECONDARY language; empty on single-language firms")
    reverse: bool = Field(default=False, description="true for credit-natural rows so they display positive")
    criteria: List[FsCriterion] = Field(default_factory=list, description="account_group rows only (1-3 criteria)")
    criteria_join: Literal["AND", "OR"] = Field(default="AND", description="how multiple criteria combine")
    formula_terms: List[FsFormulaTerm] = Field(default_factory=list, description="formula_row rows only")
    rationale: str = Field(default="", description="one line: why this row / this mapping")


class FsChildRow(_FsRowBase):
    pass


class FsRow(_FsRowBase):
    children: List[FsChildRow] = Field(default_factory=list)


class FsTopRow(_FsRowBase):
    children: List[FsRow] = Field(default_factory=list)


class FsPlan(BaseModel):
    structure_summary: str = Field(description="2-3 plain-language sentences on the structure, shown to the auditor first")
    rows: List[FsTopRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure-Python validation + flattening (no LLM) — module level for tests
# ---------------------------------------------------------------------------
_MAX_ROWS_TOTAL = 100
_MAX_CRITERIA_PER_ROW = 3
_MAX_NAME_CHARS = 300
_FR_RE = re.compile(r"^FR[0-9a-f]{6}$")


def wrap_name(text: Optional[str], lang: Optional[str]) -> str:
    """Plain caption -> the TipTap HTML the manual editor stores
    (<p dir="ltr|rtl">…</p>; rtl for Arabic). Escapes any markup."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()[:_MAX_NAME_CHARS]
    if not clean:
        return ""
    direction = "rtl" if (lang or "").lower() == "ar" else "ltr"
    return f'<p dir="{direction}">{_html.escape(clean)}</p>'


def build_coa_index(coa_tree: Any) -> Dict[int, Dict[str, Any]]:
    """The fs_coa_tree read -> {coa_original_id: node} for validation/labels."""
    index: Dict[int, Dict[str, Any]] = {}
    if isinstance(coa_tree, dict):
        for a in coa_tree.get("accounts") or []:
            if isinstance(a, dict) and a.get("id") is not None:
                index[int(a["id"])] = a
    return index


_FAMILY_DEPTH = {"account_group": 1, "account_type": 2, "account": 3}


def build_criteria(
    criterions: List[Any],
    join: str,
    coa_index: Dict[int, Dict[str, Any]],
    notes: List[str],
    label: str,
    notes_language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The model's criteria -> the exact account_criteria array + pattern the
    manual editor saves. Ids are checked against the COA read at the family's
    depth (group=1, type=2, account>=3); wrong-family/unknown ids are dropped
    with a note; a row with no valid criterion left returns None (drop row).
    Only 'in' comparisons exist — 'not_in' is broken on the income statement
    (the be evaluator complements against the BS groups), so it is never built.
    """
    join = "OR" if str(join).upper() == "OR" else "AND"
    fields: List[Dict[str, Any]] = []
    for c in (criterions or [])[:_MAX_CRITERIA_PER_ROW]:
        family = getattr(c, "criteria_type", None) or (c.get("criteria_type") if isinstance(c, dict) else None)
        ids = getattr(c, "account_ids", None) or (c.get("account_ids") if isinstance(c, dict) else None) or []
        min_depth = _FAMILY_DEPTH.get(family)
        if min_depth is None:
            notes.append(_bi(
                notes_language,
                f"{label}: dropped criterion with unknown type '{family}'",
                f"{label}: حُذف معيار بنوع غير معروف '{family}'",
            ))
            continue
        valid_ids: List[int] = []
        for raw in ids:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                continue
            node = coa_index.get(i)
            depth = int(node.get("depth") or 0) if node else 0
            ok = depth == min_depth if family != "account" else depth >= 3
            if node and ok:
                if i not in valid_ids:
                    valid_ids.append(i)
            else:
                notes.append(_bi(
                    notes_language,
                    f"{label}: dropped id {i} — not a {family} in this template's chart of accounts",
                    f"{label}: حُذف المعرّف {i} — ليس من النوع '{family}' في دليل حسابات هذا القالب",
                ))
        if valid_ids:
            fields.append({"criteria_type": family, "value": valid_ids})
    if not fields:
        return None
    account_criteria: List[Dict[str, Any]] = []
    parts: List[str] = []
    for i, f in enumerate(fields):
        if i > 0:
            account_criteria.append({"type": "operator", "value": join})
            parts.append(join)
        account_criteria.append({
            "position": i + 1,
            "type": "field",
            "criteria_type": f["criteria_type"],
            "comparison_operator": "in",
            "value": f["value"],
            "display_coa_depth": _FAMILY_DEPTH[f["criteria_type"]],
        })
        parts.append(str(i + 1))
    pattern = "1" if len(fields) == 1 else f"( {' '.join(parts)} )"
    return {"account_criteria": account_criteria, "account_criteria_pattern": pattern}


def _bi(lang: Optional[str], en: str, ar: str) -> str:
    """Pick the note/summary wording for the RUNNING user's UI language. The
    checkpoint/report prose is generated once per run, so it is written in the
    language of the auditor who runs (and approves) it."""
    return ar if str(lang or "").lower().startswith("ar") else en


def _coa_names(coa_index: Dict[int, Dict[str, Any]], i: int) -> tuple:
    """The account's display names for the checkpoint's HELPER labels (criteria /
    formula breakdown) in BOTH org languages: (primary, secondary). The COA
    stores its primary language in ``name`` (Arabic for an Arabic-primary org);
    emitting both lets the FE show whichever matches the CURRENT UI language —
    even after the user switches languages on an already-drafted run."""
    node = coa_index.get(int(i)) or {}
    primary = str(node.get("name") or node.get("name_sl") or f"#{i}")
    secondary = str(node.get("name_sl") or node.get("name") or f"#{i}")
    return primary, secondary


def flatten_fs_plan(
    rows: List[Any],
    *,
    coa_index: Dict[int, Dict[str, Any]],
    bilingual: bool,
    language: str,
    secondary_language: Optional[str],
    notes_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn the LLM's tree into the exact financial_reports_config rows the be
    apply endpoint saves + a render-ready preview tree + validation notes.

    Rules (mirror what the manual editor stores):
      - names wrapped as TipTap <p dir=…> HTML; name_sl only when bilingual;
      - account_group criteria built/validated via build_criteria (drop row when
        nothing valid remains);
      - formula token strings assembled from structured terms: A<id> for valid
        accounts, FR<ref> for other formula rows (references generated here,
        forward refs fine); dangling/invalid terms drop with a note; a formula
        left with no terms drops the row, iterated to a fixpoint so nothing ever
        references a dropped row; cycles drop every row on the cycle;
      - size caps; invalid nodes are dropped with a note, never invented.

    The human helper labels (criteria / formula breakdown) are emitted in BOTH
    org languages (primary + *_sl) so the FE can render whichever matches the
    CURRENT UI language — the draft stays readable even after a language switch.
    ``notes_language`` is the RUNNING user's UI language; validation notes are
    prose written once, so they are written in that language.
    """
    notes: List[str] = []
    counter = {"n": 0}
    nlang = notes_language

    # ---- pass A: normalize every node, assign FR references up front ---------
    flat: List[Dict[str, Any]] = []  # normalized nodes in tree order (with parent link)

    def norm(node: Any, depth: int, parent: Optional[Dict[str, Any]], label: str) -> None:
        if counter["n"] >= _MAX_ROWS_TOTAL:
            notes.append(_bi(
                nlang,
                f"kept the first {_MAX_ROWS_TOTAL} rows — the draft was larger",
                f"تم الإبقاء على أول {_MAX_ROWS_TOTAL} صفًا — كانت المسودة أكبر من ذلك",
            ))
            return
        row_type = getattr(node, "row_type", None)
        name = re.sub(r"\s+", " ", str(getattr(node, "name", "") or "")).strip()
        if not name:
            notes.append(_bi(
                nlang,
                f"{label}: dropped {row_type or 'row'} with an empty name",
                f"{label}: حُذف صف ({row_type or 'row'}) بدون اسم",
            ))
            return
        counter["n"] += 1
        entry: Dict[str, Any] = {
            "node": node,
            "row_type": row_type,
            "name": name,
            "name_sl": re.sub(r"\s+", " ", str(getattr(node, "name_sl", "") or "")).strip() if bilingual else "",
            "temp_id": str(getattr(node, "temp_id", "") or f"r{counter['n']}"),
            "depth": depth,
            "parent": parent,
            "label": label,
            "children": [],
            "dropped": False,
        }
        if row_type == "formula_row":
            entry["reference"] = f"FR{uuid.uuid4().hex[:6]}"
        if parent is not None:
            parent["children"].append(entry)
        flat.append(entry)
        for j, child in enumerate(list(getattr(node, "children", []) or [])):
            norm(child, depth + 1, entry, f"{label}.{j + 1}")

    for i, node in enumerate(rows or []):
        norm(node, 1, None, f"row {i + 1}")

    ref_by_temp = {e["temp_id"]: e["reference"] for e in flat if e["row_type"] == "formula_row"}
    entry_by_temp = {e["temp_id"]: e for e in flat}

    # ---- pass B: criteria for groups ------------------------------------------
    for e in flat:
        if e["row_type"] == "account_group":
            built = build_criteria(
                list(getattr(e["node"], "criteria", []) or []),
                str(getattr(e["node"], "criteria_join", "AND") or "AND"),
                coa_index, notes, e["label"], notes_language=nlang,
            )
            if built is None:
                notes.append(_bi(
                    nlang,
                    f"{e['label']}: dropped group '{e['name']}' — no valid account mapping remained",
                    f"{e['label']}: حُذفت المجموعة «{e['name']}» — لم يتبقَّ أي ربط حسابات صالح",
                ))
                e["dropped"] = True
            else:
                e["criteria"] = built

    # ---- pass C: formulas — cycle check, then token build to a fixpoint -------
    formula_entries = [e for e in flat if e["row_type"] == "formula_row"]
    edges = {
        e["temp_id"]: [
            t.row_temp_id for t in (getattr(e["node"], "formula_terms", []) or [])
            if getattr(t, "row_temp_id", None) in ref_by_temp
        ]
        for e in formula_entries
    }
    # Kahn peel: whatever cannot be topologically ordered sits on a cycle.
    remaining = set(edges)
    peeled = True
    while peeled:
        peeled = False
        for tid in list(remaining):
            if all(dep not in remaining for dep in edges[tid]):
                remaining.discard(tid)
                peeled = True
    for tid in remaining:
        e = entry_by_temp[tid]
        notes.append(_bi(
            nlang,
            f"{e['label']}: dropped total '{e['name']}' — its formula references itself in a loop",
            f"{e['label']}: حُذف الإجمالي «{e['name']}» — معادلته تشير إلى نفسها في حلقة",
        ))
        e["dropped"] = True

    changed = True
    while changed:
        changed = False
        for e in formula_entries:
            if e["dropped"] or e.get("formula"):
                continue
            terms: List[tuple] = []  # (sign, token, primary label, secondary label)
            for t in (getattr(e["node"], "formula_terms", []) or []):
                sign = "-" if getattr(t, "sign", "+") == "-" else "+"
                ref_tid = getattr(t, "row_temp_id", None)
                acc_id = getattr(t, "account_id", None)
                if ref_tid is not None:
                    target = entry_by_temp.get(str(ref_tid))
                    if target is None or target["row_type"] != "formula_row" or target["dropped"]:
                        notes.append(_bi(
                            nlang,
                            f"{e['label']}: dropped a term of '{e['name']}' — it references a row that is not a kept total",
                            f"{e['label']}: حُذف حدّ من «{e['name']}» — يشير إلى صف ليس إجماليًا مُبقى",
                        ))
                        continue
                    terms.append((sign, target["reference"], target["name"], target.get("name_sl") or target["name"]))
                elif acc_id is not None and int(acc_id) in coa_index:
                    pl_name, sl_name = _coa_names(coa_index, int(acc_id))
                    terms.append((sign, f"A{int(acc_id)}", pl_name, sl_name))
                else:
                    notes.append(_bi(
                        nlang,
                        f"{e['label']}: dropped a term of '{e['name']}' — unknown account id {acc_id}",
                        f"{e['label']}: حُذف حدّ من «{e['name']}» — معرّف حساب غير معروف {acc_id}",
                    ))
            # the stored token string must START with a token (no leading minus)
            # — the be validator enforces it — so a positive term leads.
            terms.sort(key=lambda t: t[0] == "-")
            if not terms or terms[0][0] == "-":
                notes.append(_bi(
                    nlang,
                    f"{e['label']}: dropped total '{e['name']}' — "
                    + ("no valid term remained" if not terms else "a total of only negative terms is not supported (use reverse instead)"),
                    f"{e['label']}: حُذف الإجمالي «{e['name']}» — "
                    + ("لم يتبقَّ أي حدّ صالح" if not terms else "إجمالي من حدود سالبة فقط غير مدعوم (استخدم عكس الإشارة بدلاً من ذلك)"),
                ))
                e["dropped"] = True
                changed = True  # rows referencing it must re-resolve
                # invalidate already-built formulas that reference this row
                for other in formula_entries:
                    if other.get("formula") and e["reference"] in other["formula"]:
                        other.pop("formula", None)
                        other.pop("formula_label", None)
                        other.pop("formula_label_sl", None)
                continue
            formula = terms[0][1]
            for sign, tok, _lbl, _lbl_sl in terms[1:]:
                formula += sign + tok
            e["formula"] = formula
            # human breakdown in BOTH org languages — the FE shows the one
            # matching the current UI language
            e["formula_label"] = " ".join(
                f"{'−' if s == '-' else '+'} {lbl}" for s, tok, lbl, lbl_sl in terms
            ).lstrip("+ ").strip()
            e["formula_label_sl"] = " ".join(
                f"{'−' if s == '-' else '+'} {lbl_sl}" for s, tok, lbl, lbl_sl in terms
            ).lstrip("+ ").strip()

    # ---- pass D: emit DB rows + preview (skipping dropped subtrees) -----------
    def emit(entries: List[Dict[str, Any]]) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        db_rows: List[Dict[str, Any]] = []
        preview: List[Dict[str, Any]] = []
        for e in entries:
            if e["dropped"]:
                kept_children = [c for c in e["children"] if not c["dropped"]]
                if kept_children:
                    notes.append(_bi(
                        nlang,
                        f"{e['label']}: its {len(kept_children)} child row(s) were dropped with it",
                        f"{e['label']}: حُذف معه {len(kept_children)} من الصفوف الفرعية",
                    ))
                continue
            row: Dict[str, Any] = {
                "uniqueId": str(uuid.uuid4()),
                "row_type": e["row_type"],
                "name": wrap_name(e["name"], language),
            }
            if bilingual and e["name_sl"]:
                row["name_sl"] = wrap_name(e["name_sl"], secondary_language)
            item: Dict[str, Any] = {
                "temp_id": e["temp_id"],
                "kind": {"text_field": "heading", "account_group": "group", "formula_row": "total"}[e["row_type"]],
                "text": e["name"],
                "text_sl": e["name_sl"],
                "rationale": re.sub(r"\s+", " ", str(getattr(e["node"], "rationale", "") or "")).strip()[:300],
                "depth": e["depth"],
                "children": [],
            }
            if e["row_type"] == "account_group":
                row["reverse"] = bool(getattr(e["node"], "reverse", False))
                row.update(e["criteria"])
                item["reverse"] = row["reverse"]
                fields = [f for f in e["criteria"]["account_criteria"] if f.get("type") == "field"]
                # structured criteria in BOTH org languages so the FE can pick
                # the names matching the CURRENT UI language and translate the
                # connective ("account type in: …") itself
                item["criteria_view"] = [
                    {
                        "type": f["criteria_type"],
                        "names": [_coa_names(coa_index, v)[0] for v in f["value"][:8]],
                        "names_sl": [_coa_names(coa_index, v)[1] for v in f["value"][:8]],
                        "more": len(f["value"]) > 8,
                    }
                    for f in fields
                ]
                # kept for backward-compat (older drafts render this raw string)
                item["criteria_labels"] = [
                    f"{f['criteria_type'].replace('_', ' ')} in: "
                    + ", ".join(_coa_names(coa_index, v)[0] for v in f["value"][:8])
                    + ("…" if len(f["value"]) > 8 else "")
                    for f in fields
                ]
            elif e["row_type"] == "formula_row":
                row["reverse"] = bool(getattr(e["node"], "reverse", False))
                row["reference"] = e["reference"]
                row["formula"] = e["formula"]
                item["reverse"] = row["reverse"]
                item["formula_label"] = e.get("formula_label", "")
                item["formula_label_sl"] = e.get("formula_label_sl", "")
            child_rows, child_items = emit(e["children"])
            row["child_row"] = child_rows
            item["children"] = child_items
            db_rows.append(row)
            preview.append(item)
        return db_rows, preview

    top = [e for e in flat if e["parent"] is None]
    db_rows, preview = emit(top)

    counts = {
        "total_rows": 0,
        "headings": 0,
        "account_groups": 0,
        "formula_rows": 0,
    }

    def count(rows_: List[Dict[str, Any]]) -> None:
        for r in rows_:
            counts["total_rows"] += 1
            key = {"text_field": "headings", "account_group": "account_groups", "formula_row": "formula_rows"}[r["row_type"]]
            counts[key] += 1
            count(r.get("child_row") or [])

    count(db_rows)
    return {"rows": db_rows, "preview": preview, "notes": notes, "counts": counts}


def coverage_check(db_rows: List[Dict[str, Any]], coa_tree: Any, notes_language: Optional[str] = None) -> List[str]:
    """ADVISORY coverage notes for the checkpoint (approximate on purpose — the
    real evaluator runs in the be): which depth-2 account types no group row
    captures, and which are matched by more than one group (double counting).
    Criteria are unioned per row (an AND may narrow further — upper bound, so
    'uncovered' findings are reliable, 'covered' is optimistic). Notes are prose
    written once per run — in the RUNNING user's UI language."""
    types = [t for t in (coa_tree.get("types") or [] if isinstance(coa_tree, dict) else []) if isinstance(t, dict)]
    accounts = [a for a in (coa_tree.get("accounts") or [] if isinstance(coa_tree, dict) else []) if isinstance(a, dict)]
    if not types:
        return []
    hits: Dict[int, int] = {int(t["id"]): 0 for t in types if t.get("id") is not None}
    type_by_account: Dict[int, Optional[int]] = {
        int(a["id"]): (int(a["type"]) if a.get("type") is not None else None)
        for a in accounts if a.get("id") is not None
    }
    group_of_type: Dict[int, Optional[int]] = {
        int(t["id"]): (int(t["group"]) if t.get("group") is not None else None) for t in types if t.get("id") is not None
    }

    def walk(rows_: List[Dict[str, Any]]) -> None:
        for r in rows_ or []:
            captured: set = set()
            for f in r.get("account_criteria") or []:
                if f.get("type") != "field":
                    continue
                values = {int(v) for v in f.get("value") or []}
                if f.get("criteria_type") == "account_type":
                    captured |= {t for t in hits if t in values}
                elif f.get("criteria_type") == "account_group":
                    captured |= {t for t, g in group_of_type.items() if g in values}
                else:
                    captured |= {type_by_account.get(a) for a in values if type_by_account.get(a) in hits}
            for t in captured:
                hits[t] += 1
            walk(r.get("child_row") or [])

    walk(db_rows)
    # type names embedded in the note stay as stored (primary language) — only
    # the note's PROSE follows the user's language
    label_by_id = {int(t["id"]): str(t.get("name") or t.get("name_sl") or t["id"]) for t in types if t.get("id") is not None}
    notes: List[str] = []
    uncovered = [label_by_id[t] for t, n in hits.items() if n == 0]
    doubled = [label_by_id[t] for t, n in hits.items() if n > 1]
    if uncovered:
        notes.append(_bi(
            notes_language,
            "accounts under these types are not captured by any row: " + ", ".join(sorted(uncovered)[:10]),
            "الحسابات ضمن هذه الأنواع لا يلتقطها أي صف: " + "، ".join(sorted(uncovered)[:10]),
        ))
    if doubled:
        notes.append(_bi(
            notes_language,
            "these types are matched by MORE THAN ONE row (possible double counting): " + ", ".join(sorted(doubled)[:10]),
            "هذه الأنواع مطابقة بأكثر من صف واحد (احتمال احتساب مزدوج): " + "، ".join(sorted(doubled)[:10]),
        ))
    return notes


def compact_fs_reference(res: Any) -> Optional[Dict[str, Any]]:
    """The fs_reference_configs response -> the grounding dict for the prompt +
    provenance for the checkpoint; None when there is nothing usable. Pure."""
    if not isinstance(res, dict) or res.get("error"):
        return None
    source = res.get("source")
    if not isinstance(source, dict) or not source.get("rows"):
        return None
    kind = source.get("kind") or "org_practice"
    audit_file = source.get("audit_file") or {}
    file_name = str(audit_file.get("name") or "").strip()
    row_count = int(source.get("row_count") or 0)
    if kind == "template_peer":
        label = f"the firm template '{file_name or 'another template'}' ({row_count} rows)"
    else:
        label = f"engagement file '{file_name or 'a recent engagement'}' ({row_count} rows)"
    return {
        "kind": kind,
        "label": label,
        # raw pieces so the FE / synthesize can compose the provenance line in
        # the viewer's language (label is the English fallback)
        "file_name": file_name,
        "audit_file_id": audit_file.get("id"),
        "row_count": row_count,
        "truncated": bool(source.get("truncated")),
        "rows": source["rows"],
    }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class FsBuildOutAgent:
    agent_type = "fs_buildout"
    # scoped to ONE statement working paper — the router enforces both params
    requires_working_paper = True
    allowed_tools = [
        "get_audit_file_summary",
        "get_fs_config",
        "get_fs_coa_tree",
        "get_fs_reference_configs",
        "apply_fs_config",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Configure a financial statement in audit file template {audit_file_id}."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        wp_id = ctx.working_paper_id
        if not wp_id:
            raise ValueError("fs_buildout requires working_paper_id")
        report_type = ctx.report_type
        if report_type not in REPORT_TYPES:
            raise ValueError("fs_buildout requires report_type=balance_sheet|income_statement")
        statement = "balance sheet" if report_type == "balance_sheet" else "income statement"
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep(
                f"Read the current {statement} configuration", "read", "get_fs_config",
                {"working_paper_id": int(wp_id), "report_type": report_type},
            ),
            PlannedStep("Read the chart of accounts", "read", "get_fs_coa_tree", {"statement": report_type}),
            PlannedStep(
                "Find a reference configuration across the firm", "read", "get_fs_reference_configs",
                {"report_type": report_type},
            ),
            PlannedStep("Verify template & summarize setup", "compute", "verify_template_and_summarize"),
            PlannedStep(f"Draft the {statement} structure (AI)", "analysis", "propose_fs_structure"),
            PlannedStep(
                "Apply the approved structure", "write", "apply_fs_config",
                {"working_paper_id": int(wp_id), "report_type": report_type},
                requires_approval=True,
            ),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "verify_template_and_summarize":
            return self._verify_and_summarize(ctx)
        if step.tool == "propose_fs_structure":
            return self._propose_fs_structure(ctx)
        raise ValueError(f"fs_buildout: unknown compute step '{step.tool}'")

    # ---- write checkpoint: the exact payload the auditor approves -------------
    def prepare_write(self, step: PlannedStep, ctx: RunContext) -> Optional[dict]:
        if step.tool != "apply_fs_config":
            return None
        draft = ctx.find("propose_fs_structure")
        if not isinstance(draft, dict) or draft.get("error"):
            raise ValueError("drafting failed — nothing to apply")
        rows = draft.get("rows") or []
        if not rows:
            raise ValueError("the draft contains no valid rows to apply")
        if not ctx.run_id:
            raise ValueError("run id missing — cannot trace the write")
        return {
            # popped into the URL path by the registry tool (never model-chosen)
            "working_paper_id": int(step.args.get("working_paper_id") or ctx.working_paper_id),
            "report_type": step.args.get("report_type"),
            "ai_run_id": str(ctx.run_id),
            "rows": rows,
            # human-readable half of the checkpoint; be ignores these keys
            "preview": draft.get("preview", []),
            "structure_summary": draft.get("structure_summary", ""),
            "validation_notes": draft.get("validation_notes", []),
            "counts": draft.get("counts", {}),
            "replaces_existing": bool(draft.get("replaces_existing")),
            "previous_row_count": int(draft.get("previous_row_count") or 0),
            "languages": draft.get("languages"),
            "grounding": draft.get("grounding"),
        }

    # ---- compute: template gate + code-derived facts ---------------------------
    def _verify_and_summarize(self, ctx: RunContext) -> Dict[str, Any]:
        summary = ctx.find("get_audit_file_summary") or {}
        if not (isinstance(summary, dict) and summary.get("is_template")):
            # authoritative be value — raising here means the write step is never reached
            raise ValueError("fs_buildout runs on audit file TEMPLATES only — this file is not a template")
        fs = ctx.find("get_fs_config") or {}
        if not isinstance(fs, dict) or fs.get("error"):
            raise ValueError(f"could not read the statement configuration: {fs.get('error') if isinstance(fs, dict) else 'unavailable'}")
        coa = ctx.find("get_fs_coa_tree") or {}
        if not isinstance(coa, dict) or coa.get("error") or not coa.get("accounts"):
            raise ValueError(
                "this template's chart of accounts has no accounts for this statement — "
                "assign/complete the COA template first"
            )
        langs = fs.get("languages") or {}
        primary = (langs.get("primary") or "en").lower()
        secondary = (langs.get("secondary") or "").lower() or None
        bilingual = bool(langs.get("use_secondary") and secondary and secondary != primary)
        row_count = int(fs.get("row_count") or 0)
        return {
            "working_paper_name": (fs.get("working_paper") or {}).get("name"),
            "report_type": fs.get("report_type"),
            "existing_row_count": row_count,
            "replaces_existing": row_count > 0,
            "coa_accounts": int(coa.get("count") or 0),
            "coa_truncated": bool(coa.get("truncated")),
            "primary_language": primary,
            "secondary_language": secondary if bilingual else None,
            "bilingual": bilingual,
        }

    # ---- analysis: the ONE structured LLM call ---------------------------------
    def _propose_fs_structure(self, ctx: RunContext) -> Dict[str, Any]:
        facts = ctx.find("verify_template_and_summarize") or {}
        fs = ctx.find("get_fs_config") or {}
        coa = ctx.find("get_fs_coa_tree") or {}
        report_type = facts.get("report_type") or ctx.report_type
        bilingual = bool(facts.get("bilingual"))
        reference = compact_fs_reference(ctx.find("get_fs_reference_configs"))

        prompt = build_fs_plan_prompt(
            statement_type=report_type,
            file_summary=ctx.find("get_audit_file_summary"),
            coa_tree=coa,
            existing_summary=fs.get("existing_summary") if isinstance(fs, dict) else [],
            existing_row_count=int(facts.get("existing_row_count") or 0),
            reference=reference,
            language=facts.get("primary_language") or ctx.language,
            secondary_language=facts.get("secondary_language"),
            # rationale + structure_summary are read by the auditor RUNNING the
            # agent — written in their UI language, not the org's primary
            reviewer_language=ctx.language,
        )
        plan = generate_structured(
            prompt, FsPlan,
            system=SYSTEM_PROMPT,
            # a bilingual draft carries every caption twice — give it head-room
            max_output_tokens=16000 if bilingual else 10000,
            usage_out=ctx.usage_out,
        )
        coa_index = build_coa_index(coa)
        flat = flatten_fs_plan(
            plan.rows,
            coa_index=coa_index,
            bilingual=bilingual,
            language=facts.get("primary_language") or "en",
            secondary_language=facts.get("secondary_language"),
            # validation notes are written once — in the runner's UI language
            notes_language=ctx.language,
        )
        notes = flat["notes"] + coverage_check(flat["rows"], coa, notes_language=ctx.language)
        return {
            "structure_summary": plan.structure_summary,
            "rows": flat["rows"],
            "preview": flat["preview"],
            "validation_notes": notes,
            "counts": flat["counts"],
            "replaces_existing": bool(facts.get("replaces_existing")),
            "previous_row_count": int(facts.get("existing_row_count") or 0),
            # which org language is primary vs secondary — the FE uses this to
            # show the name matching the CURRENT UI language first
            "languages": {
                "primary": facts.get("primary_language"),
                "secondary": facts.get("secondary_language"),
            },
            "grounding": {
                "source": (
                    {k: reference[k] for k in ("kind", "label", "file_name", "audit_file_id", "row_count")}
                    if reference else None
                ),
            },
        }

    # ---- final structured result (no extra LLM call) ---------------------------
    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        draft = ctx.find("propose_fs_structure") or {}
        write_res = ctx.find("apply_fs_config") or {}
        counts = draft.get("counts", {}) if isinstance(draft, dict) else {}
        created = int(write_res.get("applied_count") or 0) if isinstance(write_res, dict) else 0
        replaced = int(write_res.get("previous_row_count") or 0) if isinstance(write_res, dict) else 0
        rejected = isinstance(write_res, dict) and write_res.get("rejected") is True

        lang = ctx.language
        needs_attention: List[str] = list(draft.get("validation_notes", []) if isinstance(draft, dict) else [])
        if rejected:
            needs_attention.append(_bi(
                lang,
                "the draft was rejected at the checkpoint — the statement is unchanged",
                "رُفضت المسودة عند نقطة الموافقة — القائمة المالية لم تتغير",
            ))

        grounding = draft.get("grounding") if isinstance(draft, dict) else None
        source = (grounding or {}).get("source") if isinstance(grounding, dict) else None
        if isinstance(grounding, dict) and not source:
            needs_attention.append(_bi(
                lang,
                "no existing statement configuration of the firm was found to learn from — "
                "the structure was built from the chart of accounts and professional standards only",
                "لم يُعثر على إعداد سابق لهذه القائمة لدى المنشأة للاستناد إليه — "
                "تم بناء الهيكل من دليل الحسابات والمعايير المهنية فقط",
            ))

        if created:
            if source and source.get("label"):
                src_name = str(source.get("file_name") or "").strip()
                if str(lang or "").lower().startswith("ar"):
                    grounded_on = (
                        f"استنادًا إلى قالب المنشأة «{src_name}». " if source.get("kind") == "template_peer" and src_name
                        else f"استنادًا إلى ملف الارتباط «{src_name}». " if src_name
                        else f"استنادًا إلى {source['label']}. "
                    )
                else:
                    grounded_on = f"Grounded on {source['label']}. "
            else:
                grounded_on = ""
            replaced_note = _bi(
                lang,
                f" It replaced the previous {replaced}-row structure (restorable in one click).",
                f" وقد حلّ محل الهيكل السابق المكوّن من {replaced} صفًا (يمكن استعادته بنقرة واحدة).",
            ) if replaced else ""
            summary = grounded_on + _bi(
                lang,
                f"Configured the statement with {created} row(s): "
                f"{counts.get('headings', 0)} heading(s), {counts.get('account_groups', 0)} mapped account group(s) "
                f"and {counts.get('formula_rows', 0)} total(s).",
                f"تم إعداد القائمة بـ {created} صفًا: "
                f"{counts.get('headings', 0)} عناوين، و{counts.get('account_groups', 0)} مجموعة حسابات مرتبطة، "
                f"و{counts.get('formula_rows', 0)} إجماليات.",
            ) + replaced_note
        elif rejected:
            summary = _bi(
                lang,
                "The proposed structure was rejected at the approval checkpoint; the statement is unchanged.",
                "رُفض الهيكل المقترح عند نقطة الموافقة؛ القائمة المالية لم تتغير.",
            )
        else:
            summary = _bi(
                lang,
                "The run finished without changing the statement.",
                "انتهت العملية دون تغيير القائمة المالية.",
            )

        return {
            "summary": summary,
            "structure_summary": draft.get("structure_summary", "") if isinstance(draft, dict) else "",
            "rows_proposed": counts.get("total_rows", 0),
            "rows_created": created,
            # alias so the shared run-history panel (sections_created) works unchanged
            "sections_created": created,
            "replaced_previous_rows": replaced,
            "needs_attention": needs_attention,
            "data_gaps": [],
        }


register_definition(FsBuildOutAgent())
