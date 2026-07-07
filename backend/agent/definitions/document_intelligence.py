"""
Document Intelligence agent (agent_type = "document_intelligence").

Goal: "Classify the uploaded documents on this file and suggest where each
belongs." It lists the file's supporting documents, fetches each one's content
(via a new grant-scoped 1audit-be endpoint that returns a short-lived presigned
S3 URL), OCR/parses the bytes locally with ``parser.py``, and classifies each
document — type, key fields, and the account/working-paper it likely supports.

This is the one agent that reads document CONTENT, so it needs the new be
endpoint ``GET …/documents/:reference/content``. It is still READ-ONLY toward the
audit file — it suggests routing; it does not attach anything (the write/route
step is a deferred refinement).

Numbers/▶facts discipline: extraction is deterministic (download + parse in
``_extract_documents``); the LLM only classifies from the extracted text and is
told to use 'other' / low confidence rather than invent. Counts in the report are
computed in code.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field

from agent.types import PlannedStep, RunContext, register_definition
from parser import extract_text
from structured import generate_structured


_MAX_DOCS = 6          # cap downloads/parses per run (OCR is slow)
_TEXT_TO_LLM = 2000    # chars of each doc's text sent to the classifier
_DOWNLOAD_TIMEOUT = (5, 60)


# ---------------------------------------------------------------------------
# LLM output schema — classification only (counts are computed in code)
# ---------------------------------------------------------------------------
class DocClass(BaseModel):
    index: int = Field(description="the `index` of the document this classifies")
    doc_type: str = Field(description="invoice / bank statement / contract / confirmation / ledger / receipt / financial statement / form / correspondence / other")
    summary: str = Field(default="", description="one line: what this document is")
    key_fields: List[str] = Field(default_factory=list, description="key facts visible in the text: dates, amounts, parties, ids")
    suggested_routing: str = Field(default="", description="the account / working paper / audit area it likely supports")
    confidence: float = Field(default=0.0, description="0..1 confidence in the classification")


class DocBatch(BaseModel):
    classifications: List[DocClass] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic fetch + parse (no LLM) — download presigned URL, OCR/parse bytes
# ---------------------------------------------------------------------------
def _ext_for(name: str, mime: str) -> str:
    if name and "." in os.path.basename(name):
        return os.path.splitext(name)[1].lower()
    m = (mime or "").lower()
    if "pdf" in m:
        return ".pdf"
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "tiff" in m:
        return ".tiff"
    return ".bin"


def _fetch_and_parse(ctx: RunContext, reference: str, name: str, mime: str) -> Dict[str, Any]:
    """Resolve a document to a presigned URL (be), download it, and extract text.
    Any failure returns {"error": …} — the run records it as a gap and continues."""
    meta = ctx.copilot.get(f"documents/{quote(str(reference), safe='')}/content")
    if not isinstance(meta, dict) or meta.get("error"):
        return {"text": "", "error": (meta.get("error") if isinstance(meta, dict) else "content endpoint failed")}
    url = meta.get("download_url")
    if not url:
        return {"text": "", "error": "no download_url returned"}
    try:
        resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        data = resp.content
    except requests.RequestException as exc:
        return {"text": "", "error": f"download failed: {exc}"}

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=_ext_for(name, mime), delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        text, _kind = extract_text(tmp)
        text = (text or "").strip()
        return {"text": text[:_TEXT_TO_LLM], "chars": len(text), "error": None}
    except Exception as exc:  # parser raises a few custom errors; never fatal here
        return {"text": "", "error": f"parse failed: {type(exc).__name__}: {exc}"}
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class DocumentIntelligenceAgent:
    agent_type = "document_intelligence"
    allowed_tools = [
        "get_audit_file_summary",
        "list_documents",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Classify the uploaded documents on audit file {audit_file_id} and suggest where each belongs."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("List documents", "read", "list_documents"),
            PlannedStep("Fetch & extract document text", "compute", "extract_documents"),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "extract_documents":
            return self._extract_documents(ctx)
        raise ValueError(f"document_intelligence: unknown compute step '{step.tool}'")

    def _extract_documents(self, ctx: RunContext) -> Dict[str, Any]:
        listing = ctx.find("list_documents") or {}
        docs = listing.get("documents", []) if isinstance(listing, dict) else []
        total = len(docs)
        out: List[Dict[str, Any]] = []
        gaps: List[str] = []
        with_text = 0
        for d in docs[:_MAX_DOCS]:
            if not isinstance(d, dict):
                continue
            ref, name, mime = d.get("reference"), d.get("name"), d.get("mime_type")
            linked = bool(d.get("working_papers"))
            if not ref:
                gaps.append(f"{name or 'document'}: no reference to fetch")
                continue
            res = _fetch_and_parse(ctx, ref, name, mime)
            entry = {"reference": ref, "name": name or str(ref), "mime_type": mime,
                     "linked": linked, "text": res.get("text") or "", "chars": res.get("chars", 0)}
            if res.get("error") or not entry["text"]:
                gaps.append(f"{name or ref}: {res.get('error') or 'no extractable text'}")
            else:
                with_text += 1
            out.append(entry)
        if total > _MAX_DOCS:
            gaps.append(f"only the first {_MAX_DOCS} of {total} documents were processed")
        return {"docs": out, "total": total, "considered": len(out), "with_text": with_text, "gaps": gaps}

    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        ext = ctx.find("extract_documents") or {}
        docs = ext.get("docs", []) if isinstance(ext, dict) else []
        gaps = list(ext.get("gaps", []) if isinstance(ext, dict) else [])

        textful = [(i, d) for i, d in enumerate(docs) if d.get("text")]
        by_index: Dict[int, DocClass] = {}
        if textful:
            payload = {"documents": [
                {"index": i, "name": d["name"], "mime_type": d["mime_type"], "linked": d["linked"], "text": d["text"]}
                for i, d in textful
            ]}
            prompt = (
                "You are triaging uploaded audit documents. For EACH document below (referenced by its `index`), "
                "use its extracted text to classify it.\n\n"
                f"DOCUMENTS (JSON):\n{json.dumps(payload, default=str)[:90000]}\n\n"
                "For each, return: doc_type (invoice / bank statement / contract / confirmation / ledger / receipt / "
                "financial statement / form / correspondence / other), a one-line `summary`, `key_fields` (dates, "
                "amounts, parties, ids actually visible in the text), `suggested_routing` (which account, working "
                "paper, or audit area it likely supports), and `confidence` 0..1.\n"
                "Classify ONLY from the provided text. If the text is too sparse to tell, use 'other' with low "
                "confidence. Never invent a fact that is not in the text."
            )
            system = (
                "You are a senior auditor triaging engagement documents. Classify and extract only from the provided "
                "text; never invent a fact. Be concise."
            )
            batch = generate_structured(prompt, DocBatch, system=system, usage_out=ctx.usage_out)
            by_index = {c.index: c for c in batch.classifications}

        items: List[Dict[str, Any]] = []
        unlinked = 0
        type_counts: Dict[str, int] = {}
        classified = 0
        for i, d in enumerate(docs):
            if not d.get("linked"):
                unlinked += 1
            link_txt = "linked to a working paper" if d.get("linked") else "not linked to any working paper"
            c = by_index.get(i)
            if c:
                classified += 1
                type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1
                metrics = f"{link_txt} · {round((c.confidence or 0) * 100)}% confidence"
                if c.key_fields:
                    metrics += " · " + " · ".join(c.key_fields[:4])
                items.append({
                    "title": d["name"], "status": c.doc_type, "metrics": metrics,
                    "reason": c.summary,
                    "action": (f"Likely supports: {c.suggested_routing}" if c.suggested_routing else ""),
                })
            else:
                items.append({
                    "title": d["name"], "status": "unreadable", "metrics": link_txt,
                    "reason": "Could not extract text to classify (scanned/empty or unsupported format).",
                    "action": "",
                })

        summary = (
            f"Reviewed {ext.get('considered', 0)} of {ext.get('total', 0)} document(s); extracted text from "
            f"{ext.get('with_text', 0)}. {unlinked} not linked to any working paper."
        )
        if type_counts:
            summary += " Types: " + ", ".join(f"{k} ({v})" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
        return {
            "summary": summary,
            "documents_total": ext.get("total", 0),
            "documents_classified": classified,
            "unlinked_documents": unlinked,
            "classified_documents": items,
            "data_gaps": gaps,
        }


register_definition(DocumentIntelligenceAgent())
