"""
Document Extraction agent (agent_type = "document_extraction").

Goal: "Read and summarise the SELECTED documents." The multi-document
counterpart of the per-document AI summary: it runs the document reader on the
documents the auditor selected (``document_ids`` — REQUIRED; an audit file can
hold hundreds of documents, so nothing is ever read wholesale), skips ones whose
current version is already read, stores every insight in
``aura_document_insights`` (so the per-document panel and the chat tool find
them instantly), and reports what the documents are, which working papers they
support and what needs a second look.

Read-only toward the audit file (ISA 220): it never attaches, moves or edits a
document — suggested routing is advice. Numbers discipline: every figure in the
report is extracted from a document or counted in code; the model never adds up.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import config
import document_insight_store as store
import document_reader
from agent.types import PlannedStep, RunContext, register_definition

logger = logging.getLogger(__name__)


def _file_context(ctx: RunContext) -> Dict[str, Any]:
    summary = ctx.find("get_audit_file_summary") or {}
    fc: Dict[str, Any] = {}
    if isinstance(summary, dict) and not summary.get("error"):
        for k in ("name", "client", "sector", "currency", "period_start", "period_end"):
            if summary.get(k):
                fc[k] = summary[k]
    return fc


def _working_papers(ctx: RunContext) -> List[Dict[str, Any]]:
    wps = ctx.find("list_working_papers") or {}
    if isinstance(wps, dict) and isinstance(wps.get("working_papers"), list):
        return [w for w in wps["working_papers"] if isinstance(w, dict)][:150]
    return []


class DocumentExtractionAgent:
    agent_type = "document_extraction"
    # the router refuses a start without document_ids (422) — selection only
    requires_document_ids = True
    allowed_tools = [
        "get_audit_file_summary",
        "list_working_papers",
        "list_documents",
    ]

    def default_goal(self, audit_file_id: int) -> str:
        return f"Read and summarise the selected documents on audit file {audit_file_id}."

    def build_plan(self, ctx: RunContext) -> List[PlannedStep]:
        ids = [int(d) for d in (ctx.document_ids or [])]
        if not ids:
            raise ValueError("document_extraction requires document_ids (the selected documents)")
        return [
            PlannedStep("Read file summary", "read", "get_audit_file_summary"),
            PlannedStep("List working papers", "read", "list_working_papers"),
            PlannedStep("List documents", "read", "list_documents"),
            # the selection is baked into the step args so it survives pauses/restarts
            PlannedStep("Read & summarise the selected documents", "compute", "read_documents", {"document_ids": ids}),
        ]

    def execute_step(self, step: PlannedStep, ctx: RunContext) -> Any:
        if step.tool == "read_documents":
            return self._read_documents(ctx, step.args.get("document_ids") or ctx.document_ids)
        raise ValueError(f"document_extraction: unknown compute step '{step.tool}'")

    # ------------------------------------------------------------------ compute
    def _read_documents(self, ctx: RunContext, document_ids: List[int]) -> Dict[str, Any]:
        """Loop the document reader over the SELECTED documents only, skipping ones
        whose current version is already read, within a per-run document cap AND a
        wall-clock budget so the step never outlives the runtime's deadline. Token
        usage is summed into ctx.usage_out."""
        wanted = {int(d) for d in (document_ids or [])}
        listing = ctx.find("list_documents") or {}
        all_docs = listing.get("documents", []) if isinstance(listing, dict) else []
        docs = [d for d in all_docs if isinstance(d, dict) and d.get("document_id") is not None and int(d["document_id"]) in wanted]
        total = len(docs)
        fc = _file_context(ctx)
        wps = _working_papers(ctx)
        max_docs = int(config.DOC_READER_MAX_BATCH_DOCS)
        budget = float(config.DOC_READER_BATCH_TIME_BUDGET_SEC)
        t0 = time.monotonic()

        results: List[Dict[str, Any]] = []
        gaps: List[str] = []
        missing = wanted - {int(d["document_id"]) for d in docs}
        if missing:
            gaps.append(f"{len(missing)} selected document(s) were not found on this file: {sorted(missing)[:10]}")
        read_now = 0
        reused = 0
        tot_in = tot_out = 0
        model = ""
        considered = 0
        for d in docs:
            if not isinstance(d, dict):
                continue
            if considered >= max_docs:
                gaps.append(f"only the first {max_docs} of {total} selected documents were read (per-run cap) — run again for the rest")
                break
            if time.monotonic() - t0 > budget:
                gaps.append("time budget reached — remaining documents were not read; run the agent again to continue")
                break
            considered += 1
            name = d.get("name") or d.get("reference") or "document"
            doc_id = d.get("document_id")
            ref = d.get("reference")
            linked = bool(d.get("working_papers"))
            if doc_id is None and not ref:
                gaps.append(f"{name}: no id/reference to fetch")
                continue
            try:
                meta = document_reader.fetch_document_meta(ctx.copilot, document_id=doc_id, reference=ref)
            except document_reader.DocumentReaderError as exc:
                gaps.append(f"{name}: {exc}")
                continue
            key = document_reader.content_key_for(meta)
            # reuse the stored reading if the source is unchanged (any language —
            # the structured fields are what the report lists)
            cached = store.lookup_cached_sync(
                audit_file_id=ctx.audit_file_id, document_id=meta.get("document_id") or doc_id,
                content_key=key, language=ctx.language,
            )
            if cached is not None:
                reused += 1
                results.append(self._entry(meta, cached.get("insight") or {}, linked, cached=True))
                continue
            usage: dict = {}
            try:
                out = document_reader.build_insight(
                    ctx.copilot, document_id=doc_id, reference=ref, language=ctx.language,
                    file_context=fc, working_papers=wps, usage_out=usage, meta=meta,
                )
            except document_reader.DocumentReaderError as exc:
                gaps.append(f"{name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 — one bad document never ends the run
                logger.exception("document_extraction: %s failed", name)
                gaps.append(f"{name}: reading failed ({type(exc).__name__})")
                continue
            tot_in += int(usage.get("input", 0) or 0)
            tot_out += int(usage.get("output", 0) or 0)
            model = usage.get("model") or model
            read_now += 1
            store.persist_from_thread(
                organization_id=ctx.organization_id,
                audit_file_id=ctx.audit_file_id,
                document_id=int(meta.get("document_id") or doc_id or 0),
                document_reference=meta.get("reference"),
                document_name=meta.get("name"),
                mime_type=meta.get("mime_type"),
                language=ctx.language,
                content_key=out["content_key"],
                insight=out["insight"],
                extracted_text=out.get("extracted_text"),
                read_method=out.get("read_method"),
                pages=out.get("pages"),
                model=str(model or ""),
                created_by=None,
            )
            results.append(self._entry(meta, out["insight"], linked, cached=False))

        if tot_in or tot_out:
            ctx.usage_out.update(input=tot_in, output=tot_out, model=model)
        return {
            "documents": results,
            "total": total,
            "considered": considered,
            "read_now": read_now,
            "reused": reused,
            "gaps": gaps,
        }

    @staticmethod
    def _entry(meta: Dict[str, Any], insight: Dict[str, Any], linked: bool, *, cached: bool) -> Dict[str, Any]:
        rel = insight.get("audit_relevance") or {}
        warn = [c.get("detail") for c in (insight.get("checks") or []) if isinstance(c, dict) and c.get("status") == "warning"]
        return {
            "document_id": meta.get("document_id"),
            "reference": meta.get("reference"),
            "name": meta.get("name"),
            "linked": linked,
            "cached": cached,
            "doc_type": insight.get("doc_type"),
            "title": insight.get("title"),
            "summary_short": insight.get("summary_short"),
            "confidence": insight.get("confidence"),
            "suggested_working_papers": rel.get("suggested_working_papers") or [],
            "audit_areas": rel.get("audit_areas") or [],
            "red_flags": list(insight.get("red_flags") or []) + [w for w in warn if w],
            "amounts": insight.get("amounts") or [],
            "currency": insight.get("currency"),
        }

    # --------------------------------------------------------------- synthesize
    def synthesize(self, ctx: RunContext) -> Dict[str, Any]:
        out = ctx.find("read_documents") or {}
        docs = out.get("documents", []) if isinstance(out, dict) else []
        gaps = list(out.get("gaps", []) if isinstance(out, dict) else [])

        type_counts: Dict[str, int] = {}
        items: List[Dict[str, Any]] = []
        flagged: List[Dict[str, Any]] = []
        unlinked = 0
        for d in docs:
            t = str(d.get("doc_type") or "other")
            type_counts[t] = type_counts.get(t, 0) + 1
            if not d.get("linked"):
                unlinked += 1
            conf = d.get("confidence")
            metrics = ("linked to a working paper" if d.get("linked") else "not linked to any working paper")
            if isinstance(conf, (int, float)):
                metrics += f" · {round(float(conf) * 100)}% confidence"
            main_amount = next((a for a in d.get("amounts") or [] if isinstance(a, dict) and a.get("value") is not None), None)
            if main_amount:
                try:
                    metrics += f" · {main_amount.get('label')}: {float(main_amount['value']):,.2f} {main_amount.get('currency') or d.get('currency') or ''}".rstrip()
                except (TypeError, ValueError):
                    pass
            routing = ", ".join(str(w) for w in (d.get("suggested_working_papers") or [])[:3])
            items.append({
                "title": f"{d.get('reference') + ' · ' if d.get('reference') else ''}{d.get('name') or 'document'}",
                "status": t,
                "metrics": metrics,
                "reason": d.get("summary_short") or d.get("title") or "",
                "action": (f"Likely supports: {routing}" if routing else ""),
            })
            for rf in d.get("red_flags") or []:
                flagged.append({"title": d.get("name") or d.get("reference") or "document", "status": "warning", "reason": str(rf)})

        summary = (
            f"Read {out.get('considered', 0)} of {out.get('total', 0)} document(s): "
            f"{out.get('read_now', 0)} newly read, {out.get('reused', 0)} already read. "
            f"{unlinked} not linked to any working paper; {len(flagged)} item(s) need a second look."
        )
        if type_counts:
            summary += " Types: " + ", ".join(f"{k} ({v})" for k, v in sorted(type_counts.items(), key=lambda x: -x[1]))
        return {
            "summary": summary,
            "documents_total": out.get("total", 0),
            "documents_read": len(docs),
            "documents_newly_read": out.get("read_now", 0),
            "unlinked_documents": unlinked,
            "red_flags_found": len(flagged),
            "document_summaries": items,
            "needs_attention": flagged,
            "data_gaps": gaps,
        }


register_definition(DocumentExtractionAgent())
