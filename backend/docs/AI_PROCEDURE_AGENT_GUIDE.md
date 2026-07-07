# 1audit AI Procedure Agent — Complete Guide

*How the "Build this audit program" AI agent works — for auditors, product, and engineers.*

Last updated: 2026-07-07

---

## 1. What it is, in one paragraph

The Procedure Agent is a **supervised AI assistant** that drafts a complete audit
program (the procedures, titles, comments and response options) for one **Program &
Checklist working paper**. It reads real data from the audit file, drafts the program
in your firm's style, and **pauses for your approval before saving anything**. You can
edit the draft at the checkpoint, approve it, and undo the whole thing with one click.
It works both inside real audit files and inside audit file **templates**, and it writes
in **English and Arabic** together. It is tuned for the **Saudi market** (ISAs as adopted
by SOCPA, plus zakat/ZATCA, VAT/FATOORA, GOSI where relevant).

"Supervised" is the key word: the AI proposes, **you dispose**. Nothing reaches the
working paper without your explicit approval, and everything it creates can be undone.

---

## 2. Where you find it

The agent lives on a **Program & Checklist** working paper's configuration screen.

- **Real audit file:** open the file → the Program & Checklist working paper → config /
  edit mode → the **AI Agent** button.
- **Audit file template:** same path, under *Audit file templates*. The button appears
  there too (added in the template-mode release).

It is gated per-organization and per-user (the same access control as the AuditAI chat),
so only enabled firms/users see it.

**On templates, only the program builder is offered.** The read-only agents (File
Review, Substantive Testing, etc.) analyse engagement data — risks, balances, sampling —
which a template skeleton doesn't have, so they are hidden there.

---

## 3. The two modes

The same agent behaves differently depending on **where** it runs.

### Real audit file mode

The file has a client, a period, a trial balance, assessed risks, materiality. The agent
uses all of it, so the program is **tailored to this engagement's risks**.

Plan (10 steps): read working paper config → **find this client's earlier programs** →
read file summary → read risks → read audit plan → read materiality → read area accounts
& testing → summarise the setup → **draft the program** → **create the approved sections**.

### Template mode

A template is a **skeleton** — no client, no period, no risks, no numbers. Here the agent
drafts the firm's **standard program** for the area, which will later be copied into every
audit file created from that template. It **skips the engagement-data steps** (they'd be
empty and misleading) and covers the **full standard assertion set** for the area.

Plan (6 steps): read working paper config → **find reference programs across the firm** →
read file summary → summarise the setup → **draft the firm's standard program** → **create
the approved sections**.

When you later create a real audit file from that template, the template's working papers
**and their program sections are copied into the new file** by the existing template-copy
flow — so the standard program you authored once is reused automatically.

---

## 4. How a draft is produced

The agent **never writes from a blank page**. It writes by AI (Claude, via AWS Bedrock),
but grounded in real data. Two things feed every draft:

### 4a. The grounding source — a real program to adapt

Before drafting, the agent looks for the **best real program to adapt**, in trust order.
The source is shown to you at the checkpoint as an "Adapted from…" banner.

**In a real audit file** (this client's own work only — templates are never used here):

| Priority | Source (`kind`) | What it is |
|---|---|---|
| 1 | `prior_year` | The **same** working paper in this client's **previous-year** file |
| 2 | `current_year` | The **same** working paper in another of this client's **current-year** files |
| 3 | `structure` | Another of this client's programs, used for **structure & house style only** (not its content) |
| — | *(none)* | Nothing matched — drafted from risk data + firm style |

**In a template** (the firm's own programs — never this or other clients' private data
beyond the org):

| Priority | Source (`kind`) | What it is |
|---|---|---|
| 1 | `template_peer` | The **same** working paper in another of the firm's **templates** |
| 2 | `org_practice` | The **same** working paper from a recent **real engagement** — adapted and **stripped of all client-specific detail** |
| 3 | `structure` | A sibling program of the firm, for structure only |
| — | *(none)* | Drafted from professional standards + firm style |

Matching is by the working paper's stable **reference** first, then by name. The agent is
instructed to **adapt** the source (keep order, grouping, wording, assertions, response
sets where they still fit) and to **never copy figures, dates, sample sizes, findings or
conclusions** — a procedure describes work to *perform*, not results.

### 4b. Firm style memory (house style)

Alongside the source, the agent retrieves the **3 closest past procedures** your firm has
approved, as style examples. This is what makes drafts read like *your* firm's papers. It's
a private, per-organisation store that **grows every time you approve a draft** (see §7).

### 4c. Saudi-market awareness (templates)

Template drafts are written under **ISAs as adopted by SOCPA**. Where genuinely relevant to
the area, they include standard KSA considerations: zakat & income tax (**ZATCA**) for tax
and provisions; **VAT and e-invoicing (FATOORA)** for revenue, receivables and purchases;
**GOSI** and the Wage Protection System for payroll; **SAMA** only for regulated financial
entities. It never forces local content into unrelated areas.

### 4d. The working paper title matters

The title tells the agent **which area** this is. "Trade receivables" pulls the receivables
accounts, retrieves receivables-style examples, and produces confirmation/ageing/cut-off/ECL
procedures; "Inventory" produces count-attendance/costing/NRV testing. **If the title is
meaningless** (e.g. a test name like "qq"), the agent can't tell the area and falls back to
generic procedures — so meaningful working-paper names give far better drafts.

---

## 5. The approval checkpoint

When drafting finishes, the run **pauses** and shows you the proposed program:

- A human-readable preview (titles, procedures, sub-steps, suggested response options).
- The **"Adapted from…"** grounding banner and a plain-language config summary.
- A **structured editor** — you can change any title or procedure text (both languages),
  and remove nodes, before approving.

**Nothing has been written to the working paper yet.** Only when you press **Approve** are
the sections created. If you reject, nothing is written.

Bilingual note: the draft carries every field twice — `title`/`title_sl`,
`procedure_html`/`procedure_html_sl`, response options and their `_sl` pairs — so English
and Arabic land together, matching the file's language settings.

---

## 6. Undo, Redo, and Draft History

### Where the data lives

When you approve, the sections become **real rows in the working paper's own tables** — the
same place your manual sections live. Each AI-created section carries an invisible **stamp**
(`config.ai_run_id`) identifying which run made it. Manual sections have no stamp.

### Undo = hide, not erase

Undo flips that run's stamped sections from `active` to `deleted` — a **soft delete**, the
same mechanism as deleting a section by hand. They disappear from the working paper view,
but **the full data (both languages, response options, positions) stays stored in the same
working paper**, just hidden. Undo touches **only that run's sections** (plus their
sub-steps) — your manual sections and other runs' sections are never affected.

### Redo = show again

Redo takes the exact list of rows that undo hid and flips them back to `active`. Because the
rows never left the database, everything returns instantly and identically — no AI, no
re-generation. If you manually deleted one of the AI's sections before undoing, redo will
**not** resurrect it (it restores only what *that undo* hid).

### Draft history (the panel)

Every run on a working paper is remembered. When you open the AI panel on a working paper
that already has drafts, it shows **"Previous AI drafts on this working paper"** instead of
auto-starting — so you can see and act on earlier drafts. Each row shows when it ran, how
many sections it created, and its state:

- **in the paper** (green) → **Undo** button.
- **removed** (amber) → **Redo** button.
- **needs approval / 0 sections / error** → nothing to undo.

You can also start a **fresh** draft from the card above the list.

> History only shows runs made from the history release onward — older runs weren't tagged
> with their working paper, so they don't appear.

### Remove from history (trash)

The **trash icon** removes an entry from the history list — for abandoned/awaiting drafts
you don't want cluttering the list, or removed drafts you no longer need a redo handle for.

- It is **only offered when nothing that run created is still live** in the paper.
- A draft that is **still in the paper must be Undone first** (the server refuses otherwise
  — this prevents orphaned sections).
- Removing an **unfinished/awaiting** draft also **cancels** it (which frees the
  "one active run per file" slot, so you can start fresh).
- Removing a **removed** draft is permanent (you lose the redo handle); the already-hidden
  rows simply stay hidden.

### What undo does NOT reverse

The firm **style memory** (§7). Once approved, a draft's wording is learned as a house-style
example, and that stays even if you later remove the sections. This is deliberate — the firm
keeps learning from what its auditors accepted.

---

## 7. The learning loop (firm style memory)

Every time you **approve** a draft, the approved procedures (including **your edits**) are
saved into the firm's private **procedure memory** in the background. Future drafts retrieve
the closest examples from it, so the agent **converges on your house style** with use. On an
organisation's first use, it also **seeds** the memory from your existing procedures (your
templates first, then recent files) so it isn't empty on day one.

The memory is:
- **Per-organisation** and private (never shared across firms).
- **De-duplicated** by content hash (approving the same thing twice doesn't bloat it).
- **Retrieval, not training** — it's a search index the prompt reads from, so it's instant,
  auditable, and reversible by deleting rows; the underlying model is never fine-tuned.

---

## 8. Safety and guarantees

- **Nothing is written without your approval.** The draft is a proposal until you approve.
- **One-click, precise undo.** Only the run's own sections; manual work is never touched.
- **No orphans.** A live draft must be undone before its history entry can be deleted.
- **No copied client data.** Procedures describe work to perform; the prompt forbids copying
  figures, dates, samples, findings — and template `org_practice` drafts are explicitly
  stripped of client specifics.
- **Privacy scoping.** Real-file grounding uses only *this client's* files; template and
  memory grounding are *organisation-scoped*; the grant pins the run to one audit file.
- **Credit-metered.** Runs count against the org's AI credit allowance; the run has a
  per-run ceiling guardrail.

---

## 9. Architecture (for engineers)

Three services cooperate. Content lives in the product DB; run bookkeeping lives in the AI DB.

```
┌────────────────────────┐     copilot grant (15-min, pins one file)     ┌─────────────────────────┐
│  Frontend (React)      │ ───────────────────────────────────────────▶ │  auditai (FastAPI, Py)  │
│  AgentPanel /          │      /agent/run, /approve, /undo, /redo,      │  agent runtime +        │
│  AuditAIWidget         │ ◀──  /dismiss, /runs (history)                │  procedure_buildout def │
└────────────────────────┘         run state, plan, checkpoint           └───────────┬─────────────┘
                                                                                      │ internal copilot API
                                                                                      │ (grant-authenticated)
                                                                                      ▼
                                                                         ┌─────────────────────────┐
                                                                         │  1audit-be (Node/Express)│
                                                                         │  CopilotData.Service     │
                                                                         │  reads file data +       │
                                                                         │  writes program sections │
                                                                         └───────────┬─────────────┘
                                                                                      ▼
                                                                         MySQL (product DB): the
                                                                         working-paper sections
```

### The runtime (auditai)

A run is **plan → execute steps → pause at a write checkpoint → approve → synthesize**. The
plan is **static** (built at start from the request), so the template flag and working-paper
id must arrive with the start call. Read steps that fail or return empty are tolerated; only
a failed **write** kills a run. Each definition (`agent/definitions/procedure_buildout.py`)
supplies: `build_plan`, `execute_step`, `prepare_write` (the approval payload), and
`synthesize` (the final summary).

### Grounding source resolution (1audit-be)

`CopilotData.Service.js::getPriorProgramSources` decides the source server-side based on
whether the granted file is a template, and returns a compact program tree + provenance. The
auditai side compacts it (`compact_prior_program`) and injects it into the prompt
(`prompts/procedure_plan.py`), with per-`kind` heading text.

### Writes and undo/redo (1audit-be)

- `bulkCreateProgramSections` — creates the approved tree in one transaction, every section
  **stamped** with the run id, batched to avoid RDS timeouts.
- `undoProgramSectionsRun` — soft-deletes the stamped rows (+ descendants), returns the exact
  `section_ids`.
- `restoreProgramSectionsRun` — the redo: reactivates **only** the recorded ids that are
  stamped and currently deleted.

### Data stores

| Store | What it holds |
|---|---|
| **Product DB (MySQL)** | The program **sections** themselves — active or soft-deleted. The only place procedure text lives. |
| **AI DB (Postgres, `agent_runs` / `agent_steps`)** | Run bookkeeping: plan, status, `working_paper_id`, `is_template`, `dismissed`, and `result_summary` (sections created, undone flag, `undone_section_ids`). Powers history + undo/redo/delete. |
| **AI DB (Postgres, `proc_memory`, pgvector)** | Per-org firm **style memory** (embeddings + source provenance). |

---

## 10. Endpoint reference (auditai `/agent`)

| Method & path | Purpose |
|---|---|
| `POST /agent/run` | Start a run. Body includes `working_paper_id`, `is_template`, `copilot_grant`. |
| `GET  /agent/run/{id}` | Poll one run's live state (plan, steps, checkpoint). |
| `POST /agent/run/{id}/step/{idx}/approve` | Approve the checkpoint (optional `edited_payload`). |
| `POST /agent/run/{id}/step/{idx}/reject` | Reject the checkpoint. |
| `POST /agent/run/{id}/abort` | Stop a running run. |
| `POST /agent/run/{id}/undo` | Soft-delete everything the run created. |
| `POST /agent/run/{id}/redo` | Restore exactly what that undo removed. |
| `POST /agent/run/{id}/dismiss` | Remove the run from history (refuses if its draft is still live). |
| `GET  /agent/runs` | Draft history for a file/working paper (with `can_undo` / `can_redo` / `can_delete`). |

Corresponding write endpoints on 1audit-be live under
`/api/v1/internal/copilot/audit_files/:id/working_papers/:wpid/program_sections/{bulk_create,undo_run,restore_run}`,
all behind the copilot-grant middleware.

---

## 11. Operations / deployment

- **auditai container:** rebuild with `docker build --no-cache-filter runtime -t auditai-api:latest .`,
  then restart. Any code change needs a rebuild.
- **Database migrations (Postgres):** run `alembic upgrade heads`. Relevant to this feature:
  `010` (approved payload), `011` (memory provenance), `012` (run `working_paper_id` +
  `is_template`), `013` (run `dismissed`).
- **Seeding** of firm memory auto-fires per org on first agent run (a `count < 10` guard).
- **1audit-be** runs under nodemon in dev (hot-reloads); the DB is remote RDS.
- Gating: `AGENT_FEATURE_ENABLED` + `AGENT_ALLOWED_ORG_IDS` (auditai); the FE access list
  gates which orgs/users see the button.

---

## 12. FAQ / troubleshooting

**Q: I don't see any history.**
Opening the AI button used to auto-start a run immediately, skipping the history screen — it
now waits and shows history if any exists. Also, only runs made from the history release
onward appear (older runs weren't tagged with their working paper).

**Q: Two working papers got the same procedures.**
Likely their titles were meaningless (test names), so the agent couldn't tell the areas apart
and produced generic programs. Give working papers real area names. Identical *structure*
(objective comment first, same response options) is intentional house style; it's the
*content* that should differ by area.

**Q: After undo, is the data still in the working paper?**
Yes — it's stored in the same working paper, just marked hidden (`deleted`). That's why redo
is instant and lossless.

**Q: Can I start a new draft while one is "Needs your approval"?**
Not until that pending run is resolved — there's a one-active-run-per-file guard. Approve,
reject, or **delete** the pending entry (delete cancels it) to free the slot.

**Q: Does undo remove what the firm learned?**
No. Approved drafts stay in the firm style memory even after you remove their sections.

**Q: Zero hallucinations?**
Nothing can make it literally zero, but adapting *real approved programs* + code validation +
your approval gate is the strongest practical combination — and the "Adapted from…" banner
shows you exactly what each draft was built on.

---

*Questions or changes: this agent's core lives in `auditai/backend/agent/definitions/procedure_buildout.py`
(logic), `prompts/procedure_plan.py` (drafting prompt), `routers/agent.py` (API), and
`1audit-be-v3/src/services/audit/copilot/CopilotData.Service.js` (data + writes); the UI is
`1audit-fe-v3/src/component/auditai/AgentPanel.js`.*
