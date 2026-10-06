# 1audit AI Document Reader — Complete Guide

*How the "AI summary" on uploaded documents and the "Read & summarise the documents" agent work — for auditors, product, and engineers. Includes the plan for feeding document content into AI response generation.*

Last updated: 2026-10-05

---

## 1. What it is, in one paragraph

The Document Reader lets an auditor open any uploaded document in an audit file — a scanned invoice photo, a bank statement PDF, a 40-page lease contract, an Excel ageing report — and get, in a few seconds, what they would get from dropping the file into Claude or ChatGPT: **what the document is, who it involves, the key amounts (subtotal / VAT / total), dates and reference numbers, line items, a short and a full summary, which audit area and working paper it supports, and what looks wrong**. It reads scans and photos **visually** (Claude vision on Bedrock — Arabic and English, handwriting, stamps), so no Tesseract quality problems. Everything is **read-only** toward the audit file: nothing is attached, moved or edited; the reading is cached in AURA's own database so re-opening is instant and other AI features can cite it.

---

## 2. Where you find it

| Surface | What happens |
|---|---|
| **Audit file → All documents → click a file (preview)** | Header gets an **✨ AI summary** button (pink). The panel opens beside the viewer. On first open it only *checks* for a saved reading; **Summarise with AI** spends credits. |
| **All documents list** | Each read document shows a pink chip with its type (e.g. *Invoice*, *Lease agreement*) and an amber count of items to check. Row menu: **Summarise with AI** / **AI summary**. |
| **Working paper → supporting documents → preview** | Same modal, same panel (shared component). |
| **All documents → tick documents → "Summarise selected with AI"** | Reads the SELECTED documents one at a time (cached ones return instantly, cancel between documents, per-row status). A file is never scanned wholesale — hundreds of documents would be slow and costly. |
| **AI agent panel** | The `document_extraction` agent is kept but HIDDEN; its backend now refuses to run without an explicit `document_ids` selection (422). Enable it only if a consolidated multi-document report is wanted. |
| **AURA chat (inside a file)** | New tool `get_document_insight`: "What does invoice D-12 say?" / "Is the VAT on the Al Noor invoice 15%?" answers from the stored reading. |
| **✨ Draft findings / response (grounded writer)** | Same tool is available to the writer, so a draft can cite what the attached evidence says (see §8). |

Access: its own entry in the prime-admin **Features** module — feature key `document_ai` ("AI document reader"), seeded by `1audit-be/src/db_migrations/2026_10_06_document_ai_feature.sql` (or created from the Features page). It starts in *testing* (deny-closed) until organisations/users are allowlisted or the status is set to *live* for plans that include it. Three sub-features, each a prime-admin switch: `ask_document` (follow-up questions), `batch_read` ("Summarise selected with AI"), `read_on_upload` (the upload option).

How the gate is enforced, end to end:
1. FE: `isFeatureEnabled(userInfo, FEATURE_KEYS.DOCUMENT_AI)` shows/hides the ✨ button, chips and actions; `isSubFeatureEnabled(...)` gates the Ask box, the multi-select UI and the upload checkbox.
2. 1audit-be: the copilot grant is minted for `document_ai` (`POST …/copilot/grant` → `FeatureAccessService.ensureAccess`), and for `/ask` additionally with `sub_feature: "ask_document"` (`ensureSubFeatureAccess`). No access → no grant (403).
3. auditai: every `/copilot/document/*` route checks the grant's `feature` claim (`CopilotContext.require_feature("document_ai")`), and `/ask` also the `sub_feature` claim → 403 for a grant minted for chat/agent/other features.

Environment switch: `DOC_READER_ENABLED` in auditai (ships dark per environment).

---

## 3. What the auditor sees in the panel

1. **Type · confidence · how it was read** (text PDF / scanned — read visually / Word / spreadsheet) · language · "Saved summary" when served from cache.
2. **Title + one-line summary**, then the **full summary** as bullets (purpose, parties, period, key figures, terms, approvals, anything unusual).
3. **Key figures** table (subtotal, VAT, total, paid, balance… exactly as printed, with currency).
4. **Checks** — computed in Python, never by the model:
   - subtotal + VAT = total;
   - VAT rate vs the KSA 15% standard rate (zero → "check exemption basis");
   - line items add up to the subtotal/total;
   - every dated item vs the audit period (from the file's summary);
   - tax invoices: a VAT registration number is present (ZATCA).
5. **Red flags** the model observed (unsigned, undated, altered figures, related-party names, missing approval…).
6. **Parties** with identifiers (VAT/CR/IBAN as printed), **dates & references** chips, collapsible **line items** and **other key facts**.
7. **Audit relevance** — audit areas, FS assertions supported, **likely working paper(s)** chosen *only* from the file's own WP list, evidence quality (original / copy / scan / unsigned).
8. **Reading notes** (pages skipped, truncation, unreadable parts).
9. **Ask about this document** — follow-up Q&A grounded in the document's stored text ("Who signed it?", "What is the penalty clause?").
10. Copy summary · Regenerate · the standard "Generated by AI — verify against the source" note.

Prose follows the auditor's UI language (Arabic UI → Arabic summary) while identifiers, names and amounts stay exactly as printed.

---

## 4. How a reading is produced (engine: `backend/document_reader.py`)

```
1audit-be  GET …/documents/by_id/:id/content   → presigned URL + name, mime, size, updated_at
           (grant-scoped: a document of another file can never be read)
content_key = sha1(document_id:size:updated_at)   → cache check (no download on a hit)
download (size-capped, DOC_READER_MAX_FILE_MB)
extract:
   PDF   → PyMuPDF text per page; a page with no text layer is RENDERED to an image
           (scan / photo) → vision.  Mixed PDFs send text pages as text and scanned
           pages as images.  Caps: DOC_READER_MAX_PAGES, DOC_READER_MAX_IMAGE_EDGE.
   image → downscaled JPEG/PNG → vision
   DOCX / XLSX → python-docx / openpyxl (tables as Markdown)
   CSV / TXT → text
   long text → head + tail sample (DOC_READER_MAX_TEXT_CHARS), truncation flagged
analyse  ONE multimodal structured call (Claude Sonnet via Bedrock, forced tool-use on the
         DocumentInsight schema) with the engagement context (client, period, currency)
         and the working-paper list for routing.
checks   run_checks(): arithmetic / VAT rate / period / completeness — pure Python.
store    aura_document_insights (insight JSON + extracted text + content_key), per
         (audit file, document, UI language).
```

`DOC_READER_VISION_ENABLED=false` switches scans to the legacy Tesseract OCR path (English only) — keep vision on.

### Grounding discipline
- The model may only **extract** what is printed; every computed number comes from `run_checks`.
- Unreadable or missing → `data_gaps` + lower `confidence`, never a guess.
- Working-paper suggestions are restricted to the file's real WP list.

---

## 5. Cost & limits

| Guard | Default | Why |
|---|---|---|
| Pages rendered for vision | 20 | a 200-page scan would be ~200 images |
| Image longest edge | 1568 px | Claude's quality/token sweet spot |
| Native text sent | 60 000 chars (head+tail) | long contracts |
| Download size | 40 MB | presigned S3 fetch |
| Output tokens | 6 000 | line items + summary |
| Selected-documents loop (list) | one document per request, sequential | per-row progress, cancel any time; only what the auditor ticked |
| Batch agent (hidden) | selection required; 8 docs / 100 s per run | stays inside the agent runtime's 120 s step budget; re-run continues |

Every call is metered to the org ledger under feature `document_reader` and blocked at 402 when the org is over its monthly AI credits. Cached readings cost nothing.

---

## 6. Endpoints (auditai, `backend/routers/document.py`)

| Route | Body | Returns |
|---|---|---|
| `POST /copilot/document/insight` | session_token, audit_file_id, copilot_grant, document_id (or document_reference), language, refresh, cached_only | NDJSON: `progress{stage}` × n → `result{insight, cached, stale, document…}` → `done` |
| `POST /copilot/document/ask` | …, document_id, question, history[] | NDJSON text deltas (Markdown) |
| `POST /copilot/document/insights` | …, audit_file_id | `{insights:[{document_id, doc_type, summary_short, confidence, red_flags_count, suggested_working_papers}]}` |

1audit-be additions (`CopilotData.Route/Controller/Service`): `GET …/documents/by_id/:document_id/content` (returns `download_url` + `preview_url`; falls back to the S3-presign path when the CloudFront signer is not configured), `GET …/documents/by_id/:document_id/bytes` (the raw object streamed from S3 with be's own credentials — the reader's fallback when the CDN URLs are rejected, grant-scoped, size-capped by auditai) and `document_id / size / updated_at` on `GET …/documents`.

Download order in auditai (`document_reader.download_document`): attachment URL → inline/preview URL → be bytes endpoint. On a machine whose CDN rejects presigned URLs the third path is what actually serves the file (seen locally: S3 answered *"Only one auth mechanism allowed"* to the CloudFront URLs, for the FE preview too).

### Local testing without S3/CloudFront
On a developer machine the CDN may reject the presigned URLs (we saw S3 answer *"Only one auth mechanism allowed"* — the same URLs fail in the FE preview too). To exercise the whole UI flow anyway, start auditai with a static file that stands in for every download:

```
APP_ENV=local DOC_READER_DEV_STATIC_FILE=/path/to/sample_invoice.pdf \
  .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Every "Summarise with AI" then reads that file (a warning is logged per request). Restart without the variable for real documents. Never set it on dev/beta/prod. `ONEAUDIT_BASE_URL` pointing at `host.docker.internal` is auto-corrected to `localhost` when that name does not resolve (host-run server with the Docker `.env`).

New copilot tool (`copilot_tools.py`): `get_document_insight(document)` — bridged to the store in `/copilot/chat` and `/copilot/write`; auto-registered as a read tool for agents.

New agent (`agent/definitions/document_extraction.py`, type `document_extraction`): plan = file summary → working papers → documents → *read & summarise the selected documents* (compute; `document_ids` baked into the step args). Read-only, no approval checkpoint; **requires `document_ids`** — the route returns 422 without a selection so a whole-file scan is impossible.

Migration: `alembic/versions/016_document_insights.py` (also created by `init_db()` locally).

---

## 7. Other places this agent adds value (opportunities)

1. **Read on upload (opt-in)** — a "Read with AI after upload" option on the supporting-document upload: only the files the auditor just uploaded are read, right away, so the chip and summary are ready before anyone opens them. Never automatic for the whole file.
2. **Auto-attach suggestion** — "This looks like a bank confirmation; attach to C-2 Cash & bank?" (approval-gated write via the agent runtime — the write tool already pattern exists).
3. **Sampling & vouching** — in substantive testing, match a sample's amount/date/vendor against the readings of attached invoices ("3 of 25 samples have supporting invoices whose totals differ").
4. **Confirmations** — bank/receivable confirmation letters: extract confirmed balance and compare with the TB account (number-safe: both figures are extracted/read, the comparison is Python).
5. **File Review / EQR agents** — add "evidence check": working papers signed off with no readable supporting document, or whose documents are outside the period.
6. **Client portal / file requests** — read client uploads on arrival and tell the requester whether the uploaded file matches what was asked for ("you asked for the Q4 VAT return; this is a Q3 return").
7. **Chat** — already live: "what documents do we have for payroll?" → `list_documents` + `get_document_insight`.
8. **Document templates / knowledge base** — the same engine can ingest firm manuals into the help KB (parser already shared).

---

## 8. Plan — document context in AI **response generation** (procedure section)

Today the grounded writer (`write_with_file`, used by *Draft findings* and the *response* field) sees the procedure text plus the file's structured data (TB, risks, sampling, procedure results). The goal: the response **also considers what the uploaded evidence says**, e.g. *"Inspected invoice INV-0142 (SAR 11,500 incl. VAT 1,500, dated 12 Mar 2026) attached to this working paper; amount agrees to the receivable listing."*

### Phase A — available now (shipped in this change)
- `get_document_insight` is in the writer's tool set. If the auditor's instruction names a document ("use the lease agreement"), the model can pull its reading.
- Readings exist only for documents someone has summarised (or the batch agent has read).

### Phase B — evidence-aware response (next sprint, ~3–4 dev days)
1. **be**: `GET …/working_papers/:id/documents` → the documents attached to THIS working paper / section (ids, references, names). (The link table `WPSupportingDocuments` already exists.)
2. **auditai**: read tool `list_working_paper_evidence(working_paper_id)` returning attached docs **with their stored readings** (type, title, key figures, red flags, suggested assertions) — one call, no per-doc tool hopping. Documents without a reading are read on the fly (bounded: ≤3 docs, ≤10 pages each) so a response never waits on a 40-page scan.
3. **Prompt**: a `RESPONSE_EVIDENCE_ADDENDUM` for the write prompt — "When the procedure asks to inspect/vouch/agree/recalculate, cite the attached evidence by reference and figure; if no evidence is attached, say the procedure is not yet supported rather than inventing"; keep the existing `<procedure>` grounding and the HTML-only output rules.
4. **FE** (`ProgramChecklistProcedure.js`): the ✨ response/findings call passes `working_paper_id` + `section_id`; show a small "Evidence used: D-12, D-15" line under the generated text (from the `done` event's `sources`).
5. **Memory**: when the auditor approves/saves the response, store (procedure → evidence refs used) in `proc_memory`-style memory so next year's draft knows which evidence types normally support this procedure.

### Phase C — evidence-first drafting (later)
- Auto-read on upload (§7.1) so Phase B never waits.
- Cross-document reconciliation steps inside the response (invoice ↔ GRN ↔ payment) computed in Python from readings.
- Citations with page numbers (`[Page n]` markers are already in the extracted text) and a click-through that opens the preview at that page.

### Guardrails that stay
- Numbers only from documents or Python; the model never computes.
- Grant-scoped reads; readings cached in auditai Postgres only; nothing written to 1audit without approval.
- Per-org credit cap and per-run caps apply to every path.

---

## 9. Files touched (this change)

**auditai/backend**: `config.py`, `llm.py` (+`complete_structured_blocks`), `structured.py` (+`generate_structured_from_blocks`), `prompts/document_reader.py`, `document_reader.py`, `document_insight_store.py`, `models.py` (+`DocumentInsightRow`), `alembic/versions/016_document_insights.py`, `schemas.py`, `routers/document.py`, `routers/copilot.py` (evidence bridge), `copilot_tools.py` (+tool), `agent/definitions/document_extraction.py`, `agent/definitions/__init__.py`, `main.py`, `.env.example`, tests `tests/test_document_reader.py`, `tests/test_document_extraction_agent.py`, `tests/test_document_router.py`.

**1audit-be-v3**: `src/routes/internal/CopilotData.Route.js`, `src/controllers/audit/copilot/CopilotData.Controller.js`, `src/services/audit/copilot/CopilotData.Service.js`.

**1audit-fe-v3**: `src/component/auditai/DocumentInsightPanel.js` (new), `src/component/files/AuditFileDocuments/AuditFileDocumentModalPreview.js`, `src/component/files/AuditFileDocuments/AuditFileDocumentList.js`, `src/services/auditai.service.js`, `src/component/auditai/AgentPanel.js`, `src/translations/languages/ar.js`.
