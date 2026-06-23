"""
Backfill tb_mapping_memory (ticket C-2) from a JSON export of the firm's
CONFIRMED trial-balance account -> chart-of-account mappings, so "AI auto map"
is useful on day one. (Going forward, /copilot/tb-mapping/feedback appends new
confirmed mappings organically.)

This single backfill captures BOTH learning signals:
  * "previous data" / "organization trend" — pass each org's real confirmed
    mappings with that org's id.
  * cold-start "prime" seed — pass a curated golden set (e.g. the Saudi-Arabia
    reference client's confirmed mappings) stamped with the reserved
    ``--prime-org-id`` so brand-new orgs still get suggestions.

Usage:
    python backfill_tb_mappings.py path/to/mappings.json
    # stamp a curated golden set under the reserved prime org id:
    python backfill_tb_mappings.py path/to/golden.json --prime-org-id 1

Input: a JSON array of objects (extra keys ignored):
    [
      {
        "organization_id": "10",
        "client_sector": "Retail",
        "account_name": "Trade receivables",
        "account_name_sl": "العملاء",        // optional second-language name
        "account_code": "1200",
        "coa_original_id": 845,
        "coa_label": "Accounts receivable",
        "confirmed_by": "42"
      }
    ]

Additive only: appends rows to auditai's own Postgres; never reads or writes
1audit. Each run appends, so run once per export (de-duplicate the export first).
"""
import argparse
import asyncio
import json
import logging

import tb_mapping_memory
from database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill_tb_mappings")


async def run(path: str, prime_org_id: str | None = None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON array of mapping records.")

    inserted = skipped = 0
    async with AsyncSessionLocal() as db:
        for rec in records:
            org = prime_org_id or rec.get("organization_id")
            mem_id = await tb_mapping_memory.add_mapping(
                db,
                organization_id=str(org) if org not in (None, "") else None,
                client_sector=rec.get("client_sector"),
                account_name=rec.get("account_name"),
                account_name_sl=rec.get("account_name_sl"),
                account_code=rec.get("account_code"),
                coa_original_id=rec.get("coa_original_id"),
                coa_label=rec.get("coa_label"),
                confirmed_by=rec.get("confirmed_by"),
            )
            inserted += 1 if mem_id else 0
            skipped += 0 if mem_id else 1
    log.info(
        "tb_mapping_memory backfill done: inserted=%d skipped=%d prime_org_id=%s",
        inserted, skipped, prime_org_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill tb_mapping_memory from a JSON export.")
    parser.add_argument("path", help="Path to the mappings JSON export.")
    parser.add_argument(
        "--prime-org-id",
        default=None,
        help="Stamp every record under this reserved org id (curated cold-start seed).",
    )
    args = parser.parse_args()
    asyncio.run(run(args.path, args.prime_org_id))
