"""
Backfill proc_memory (ticket A-1b) from a JSON export of the firm's past accepted
procedures, so the house-style few-shot examples are useful on day one. (Going
forward, /copilot/procedure/confirm appends new ones organically.)

Usage:
    python backfill_proc_memory.py path/to/procedures.json

Input: a JSON array of objects (extra keys ignored):
    [
      {
        "organization_id": "10",
        "client_sector": "Retail",
        "audit_area": "Receivables",
        "risk_summary": "Existence/valuation of trade receivables",
        // or instead of risk_summary: "risks": [{"title": "...", "description": "..."}]
        "assertions": ["Existence", "Valuation"],
        "procedure_html": "<ol><li>...</li></ol>",
        "confirmed_by": "42"
      }
    ]

Additive only: appends rows to auditai's own Postgres; never reads or writes
1audit. Each run appends, so run once per export (de-duplicate the export first).
"""
import asyncio
import json
import logging
import sys

import proc_memory
from database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_proc_memory")


async def run(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON array of procedure records.")

    inserted = skipped = 0
    async with AsyncSessionLocal() as db:
        for rec in records:
            html = (rec.get("procedure_html") or "").strip()
            org = rec.get("organization_id")
            if not html or not org:
                skipped += 1
                continue
            risk_summary = rec.get("risk_summary") or proc_memory.summarize_risks(
                rec.get("risks") or []
            )
            mem_id = await proc_memory.add_memory(
                db,
                organization_id=str(org),
                client_sector=rec.get("client_sector"),
                audit_area=rec.get("audit_area"),
                risk_summary=risk_summary,
                assertions=rec.get("assertions") or [],
                procedure_html=html,
                confirmed_by=rec.get("confirmed_by"),
            )
            inserted += 1 if mem_id else 0
            skipped += 0 if mem_id else 1
    log.info("proc_memory backfill done: inserted=%d skipped=%d", inserted, skipped)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python backfill_proc_memory.py <procedures.json>")
    asyncio.run(run(sys.argv[1]))
