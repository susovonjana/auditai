"""
Trial-balance mapping memory (ticket C-2) — the learning store behind
"AI auto map".

Every CONFIRMED ``trial-balance account -> chart-of-account`` decision is
normalised, embedded by its account text, and appended here. A future auto-map
run retrieves the firm's closest past mappings to suggest a COA:

  * "previous data"      — this client's / org's own prior confirmed mappings.
  * "organization trend" — the whole firm's confirmed mappings (same org),
    preferring the same client sector.
  * cold start           — a reserved "prime" organization id may hold a curated
    golden seed, surfaced by the SAME search path when the real org has none.

Org-scoped and additive: this module only ever reads/appends rows in auditai's
own Postgres. It never touches 1audit's database, and contains no update/delete.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from embeddings import embed_query, embed_text
from models import TbMappingMemory

logger = logging.getLogger(__name__)

# Keep the embedded text bounded so a pathological account name can't bloat a row.
_MAX_ACCOUNT_TEXT_CHARS = 300

# Legal / entity suffixes that carry no mapping signal.
_LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "llp", "inc", "incorporated", "plc", "co",
    "corp", "corporation", "company", "pvt", "private", "gmbh", "sa", "sarl",
    "wll", "psc", "jsc", "est", "establishment",
}

# Common accounting abbreviations -> their expanded form, so "a/c recv" and
# "accounts receivable" land near each other in embedding space.
_ABBREVIATIONS = {
    "a/c": "account",
    "ac": "account",
    "acc": "account",
    "acct": "account",
    "accts": "accounts",
    "recv": "receivable",
    "rec": "receivable",
    "recvbl": "receivable",
    "rcv": "receivable",
    "pay": "payable",
    "pybl": "payable",
    "payb": "payable",
    "dr": "debit",
    "cr": "credit",
    "exp": "expense",
    "exps": "expenses",
    "rev": "revenue",
    "dep": "depreciation",
    "depr": "depreciation",
    "amort": "amortisation",
    "accr": "accrued",
    "prepd": "prepaid",
    "prov": "provision",
    "inv": "inventory",
    "invt": "inventory",
    "ppe": "property plant equipment",
    "o/s": "outstanding",
    "w/o": "written off",
    "wip": "work in progress",
    "ar": "accounts receivable",
    "ap": "accounts payable",
    "gl": "general ledger",
    "fx": "foreign exchange",
    "vat": "value added tax",
    "wht": "withholding tax",
    "cogs": "cost of goods sold",
    "p&l": "profit and loss",
    "bs": "balance sheet",
}

_PUNCT_RE = re.compile(r"[^\w\s/&]+")
_WS_RE = re.compile(r"\s+")


def normalise_account(
    name: Optional[str],
    name_sl: Optional[str] = None,
    code: Optional[str] = None,
) -> str:
    """Normalise an account name (+ second-language name) into a stable text for
    embedding and fuzzy matching: lowercase, strip punctuation and legal
    suffixes, and expand common accounting abbreviations.

    ``code`` is accepted for signature symmetry but intentionally NOT embedded —
    numeric codes carry no semantic meaning and are matched exactly via the
    ``(organization_id, account_code)`` index instead.
    """
    raw = " ".join(p for p in [(name or ""), (name_sl or "")] if p).strip().lower()
    if not raw:
        return ""
    # Normalise common separators to spaces before tokenising, but keep "a/c"
    # style slashes so the abbreviation map can catch them.
    raw = raw.replace(" ", " ")
    tokens = _WS_RE.sub(" ", raw).split(" ")
    out: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Expand a known abbreviation (check the slash form before stripping).
        if tok in _ABBREVIATIONS:
            out.append(_ABBREVIATIONS[tok])
            continue
        # Strip surrounding punctuation, then re-test.
        cleaned = _PUNCT_RE.sub("", tok.replace("/", " ")).strip()
        for piece in cleaned.split(" "):
            piece = piece.strip()
            if not piece or piece in _LEGAL_SUFFIXES:
                continue
            out.append(_ABBREVIATIONS.get(piece, piece))
    text = _WS_RE.sub(" ", " ".join(out)).strip()
    return text[:_MAX_ACCOUNT_TEXT_CHARS]


async def add_mapping(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    client_sector: Optional[str],
    account_name: Optional[str],
    account_name_sl: Optional[str],
    account_code: Optional[str],
    coa_original_id: Optional[int],
    coa_label: Optional[str] = None,
    confirmed_by: Optional[str] = None,
) -> Optional[uuid.UUID]:
    """Embed and append one confirmed mapping. Returns the new row id, or None
    when there is nothing worth storing (no org, no target COA, or empty
    account text)."""
    if not organization_id or coa_original_id in (None, ""):
        return None
    account_text = normalise_account(account_name, account_name_sl, account_code)
    if not account_text:
        return None
    try:
        coa_id = int(coa_original_id)
    except (TypeError, ValueError):
        return None
    embedding = await embed_text(account_text)
    row = TbMappingMemory(
        organization_id=str(organization_id),
        client_sector=(str(client_sector) if client_sector not in (None, "") else None),
        account_text=account_text,
        account_code=(str(account_code) if account_code not in (None, "") else None),
        coa_original_id=coa_id,
        coa_label=(coa_label or None),
        embedding=embedding,
        confirmed_by=(str(confirmed_by) if confirmed_by not in (None, "") else None),
    )
    db.add(row)
    await db.commit()
    return row.id


def _scope_orgs(organization_id: str, prime_org_id: Optional[str]) -> List[str]:
    orgs = [str(organization_id)]
    if prime_org_id and str(prime_org_id) != str(organization_id):
        orgs.append(str(prime_org_id))
    return orgs


async def org_has_mappings(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    prime_org_id: Optional[str] = None,
) -> bool:
    """Cheap one-shot check: does the caller's scope (own org + optional prime
    seed) hold ANY confirmed mappings at all? Lets the engine skip the per-account
    memory lookups entirely for a brand-new org (the large-dataset worst case),
    collapsing thousands of futile queries into one."""
    if not organization_id:
        return False
    orgs = _scope_orgs(organization_id, prime_org_id)
    stmt = select(TbMappingMemory.id).where(TbMappingMemory.organization_id.in_(orgs)).limit(1)
    res = await db.execute(stmt)
    return res.scalar() is not None


async def search_mappings(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    client_sector: Optional[str],
    account_name: Optional[str],
    account_name_sl: Optional[str] = None,
    account_code: Optional[str] = None,
    k: int = 8,
    prime_org_id: Optional[str] = None,
    embedding: Optional[List[float]] = None,
) -> List[TbMappingMemory]:
    """Up to ``k`` nearest past mappings by cosine distance on the normalised
    account text. Scope is the caller's org (and optionally a reserved
    ``prime_org_id`` golden seed for cold start). Results are ordered so the
    caller's OWN org ranks above the prime seed, then the SAME client sector,
    then nearest by meaning. Org scoping is mandatory — returns [] without one.

    ``embedding`` may be a precomputed QUERY embedding of the SAME normalised
    account text (the engine already batch-embeds every account) — pass it to
    skip a redundant per-account re-embed."""
    if not organization_id:
        return []
    if embedding is None:
        account_text = normalise_account(account_name, account_name_sl, account_code)
        if not account_text:
            return []
        try:
            embedding = await embed_query(account_text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tb_mapping_memory embed failed: %s", exc)
            return []

    orgs = _scope_orgs(organization_id, prime_org_id)
    sector = (str(client_sector).strip() if client_sector not in (None, "") else None)

    order_by = []
    if len(orgs) > 1:
        # The caller's own confirmed history outranks the shared prime seed.
        order_by.append((TbMappingMemory.organization_id == str(organization_id)).desc())
    if sector is not None:
        order_by.append((TbMappingMemory.client_sector == sector).desc())
    order_by.append(TbMappingMemory.embedding.cosine_distance(embedding))

    stmt = (
        select(TbMappingMemory)
        .where(TbMappingMemory.organization_id.in_(orgs))
        .order_by(*order_by)
        .limit(max(1, int(k)))
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def find_by_code(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    account_code: Optional[str],
    prime_org_id: Optional[str] = None,
    limit: int = 5,
) -> List[TbMappingMemory]:
    """Exact-code lookups within the org scope (Tier-1 "previous data" fast path),
    most recent first, own-org before the prime seed. Returns [] when org or code
    is missing."""
    if not organization_id or account_code in (None, ""):
        return []
    orgs = _scope_orgs(organization_id, prime_org_id)
    order_by = []
    if len(orgs) > 1:
        order_by.append((TbMappingMemory.organization_id == str(organization_id)).desc())
    order_by.append(TbMappingMemory.confirmed_at.desc())
    stmt = (
        select(TbMappingMemory)
        .where(
            TbMappingMemory.organization_id.in_(orgs),
            TbMappingMemory.account_code == str(account_code),
        )
        .order_by(*order_by)
        .limit(max(1, int(limit)))
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())
