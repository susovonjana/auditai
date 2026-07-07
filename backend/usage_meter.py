"""
Per-organization AI usage metering + monthly credit cap.

Bedrock is pay-per-token with no free-tier ceiling, so each org gets a monthly
"AI credit" budget. Credits are a normalised, model-weighted unit decoupled from
raw $ (1 credit ≈ $0.001), so price changes don't touch balances and credits
stay sellable later (see config AI_CREDIT_* / the plan's phase-2 1audit-be
billing hand-off).

Public surface
--------------
  - credits_for(tier, in_tokens, out_tokens)  -> int        (the cost function)
  - record(db, ...) / record_usage(db, ...)    -> None        (meter one call)
  - remaining_credits(db, org_id)              -> int | None  (None = unmetered)
  - has_credits(db, org_id)                    -> bool         (TB-tail soft gate)
  - ensure_credits(db, org_id)                 -> None         (raises 402 when over)

Design
------
  * The ledger (ai_usage_ledger) is the append-only source of truth; metering is
    idempotent on request_id.
  * A denormalised per-org counter (ai_org_quota) makes the pre-flight check a
    single cheap read, cached briefly via TTLCache. The monthly reset is handled
    in SQL (date_trunc('month', now())) on both the read and the debit, so no
    cron is needed.
  * Enforcement is intentionally NOT exact: output length is unknown up front, so
    we gate on "remaining > floor" before the call and true-up the actual tokens
    after. A burst can marginally overshoot — acceptable for a cost ceiling.
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    AI_CREDIT_BALANCE_CACHE_TTL_SEC,
    AI_CREDIT_CAP_ENABLED,
    AI_CREDIT_FLOOR,
    AI_CREDIT_TIER_RATES,
    AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT,
    AI_USAGE_METERING_ENABLED,
)
from copilot_cache import TTLCache

logger = logging.getLogger(__name__)

# Cache of org_id -> remaining credits (int). Short TTL so a newly exhausted org
# is blocked within the window; refreshed precisely after every debit.
_balance_cache = TTLCache(ttl_seconds=AI_CREDIT_BALANCE_CACHE_TTL_SEC, max_entries=2048)


# ---------------------------------------------------------------------------
# Credit cost
# ---------------------------------------------------------------------------
def credits_for(tier: str, input_tokens: int, output_tokens: int) -> int:
    """Model-weighted credit cost for one call. Rounds up so even tiny calls
    cost at least 1 credit when they used any tokens."""
    in_rate, out_rate = AI_CREDIT_TIER_RATES.get(tier, AI_CREDIT_TIER_RATES["smart"])
    cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
    return max(0, math.ceil(cost))


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
# Insert one ledger row. Idempotent: a duplicate non-null request_id is ignored
# via the partial unique index, so retries don't double-charge.
_INSERT_LEDGER = text(
    """
    INSERT INTO ai_usage_ledger
        (id, organization_id, user_id, feature, tier, model,
         input_tokens, output_tokens, credits, request_id, created_at)
    VALUES
        (:id, :org, :user, :feature, :tier, :model,
         :in_tok, :out_tok, :credits, :rid, now())
    ON CONFLICT (request_id) WHERE request_id IS NOT NULL DO NOTHING
    RETURNING id
    """
)

# Bump the org's period counter, creating the row lazily and resetting it when
# the stored period is a prior month. Returns the post-debit balance so we can
# refresh the cache precisely.
_UPSERT_QUOTA = text(
    """
    INSERT INTO ai_org_quota
        (organization_id, monthly_credit_allowance, period_start,
         credits_used_this_period, status, updated_at)
    VALUES
        (:org, :default_allowance, date_trunc('month', now())::date, :credits, 'active', now())
    ON CONFLICT (organization_id) DO UPDATE SET
        credits_used_this_period = CASE
            WHEN ai_org_quota.period_start < date_trunc('month', now())::date
                THEN EXCLUDED.credits_used_this_period
            ELSE ai_org_quota.credits_used_this_period + EXCLUDED.credits_used_this_period
        END,
        period_start = CASE
            WHEN ai_org_quota.period_start < date_trunc('month', now())::date
                THEN date_trunc('month', now())::date
            ELSE ai_org_quota.period_start
        END,
        updated_at = now()
    RETURNING monthly_credit_allowance, credits_used_this_period
    """
)

# Read the org's allowance and used-this-period, with the monthly reset applied
# in-query (a prior-month period reads as zero used).
_SELECT_QUOTA = text(
    """
    SELECT monthly_credit_allowance,
           CASE WHEN period_start < date_trunc('month', now())::date
                THEN 0 ELSE credits_used_this_period END AS used
    FROM ai_org_quota
    WHERE organization_id = :org
    """
)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
async def record(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    user_id: Optional[str],
    feature: str,
    tier: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    request_id: Optional[str] = None,
) -> None:
    """Append a ledger row and bump the org's period counter. Best-effort:
    metering must never break a user request, so all errors are swallowed
    (logged). No-op when metering is disabled."""
    if not AI_USAGE_METERING_ENABLED:
        return
    credits = credits_for(tier, int(input_tokens or 0), int(output_tokens or 0))
    try:
        res = await db.execute(
            _INSERT_LEDGER,
            {
                "id": uuid.uuid4(),
                "org": organization_id,
                "user": user_id,
                "feature": feature,
                "tier": tier,
                "model": model or "",
                "in_tok": int(input_tokens or 0),
                "out_tok": int(output_tokens or 0),
                "credits": credits,
                "rid": request_id,
            },
        )
        inserted = res.first() is not None
        # Only bump the counter for a fresh (non-duplicate) row that has an org.
        if inserted and organization_id:
            row = (
                await db.execute(
                    _UPSERT_QUOTA,
                    {
                        "org": organization_id,
                        "default_allowance": AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT,
                        "credits": credits,
                    },
                )
            ).first()
            if row is not None:
                remaining = max(0, int(row[0]) - int(row[1]))
                _balance_cache.set(organization_id, remaining)
        await db.commit()
    except IntegrityError:
        # Concurrent duplicate request_id — already recorded by the other writer.
        await db.rollback()
    except Exception as exc:  # never let metering break the request
        logger.warning("usage_meter.record failed (%s) — skipping.", exc)
        try:
            await db.rollback()
        except Exception:
            pass


async def record_usage(
    db: AsyncSession,
    *,
    organization_id: Optional[str],
    user_id: Optional[str],
    feature: str,
    tier: str,
    usage: dict,
    request_id: Optional[str] = None,
) -> None:
    """Convenience wrapper for the {"input", "output", "model"} usage dicts that
    structured.py / qa.py fill via ``usage_out``."""
    if not usage:
        return
    await record(
        db,
        organization_id=organization_id,
        user_id=user_id,
        feature=feature,
        tier=tier,
        model=str(usage.get("model", "")),
        input_tokens=int(usage.get("input", 0) or 0),
        output_tokens=int(usage.get("output", 0) or 0),
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Balance reads + enforcement
# ---------------------------------------------------------------------------
async def remaining_credits(db: AsyncSession, org_id: Optional[str]) -> Optional[int]:
    """Credits the org has left this period. None means "not metered" (metering
    disabled, or no org id) and is treated as unlimited by callers."""
    if not AI_USAGE_METERING_ENABLED or not org_id:
        return None
    cached = _balance_cache.get(org_id)
    if cached is not None:
        return cached
    try:
        row = (await db.execute(_SELECT_QUOTA, {"org": org_id})).first()
    except Exception as exc:
        # If the read fails, don't block the user — fail open.
        logger.warning("usage_meter.remaining_credits read failed (%s) — failing open.", exc)
        return None
    if row is None:
        remaining = AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT
    else:
        remaining = max(0, int(row[0]) - int(row[1]))
    _balance_cache.set(org_id, remaining)
    return remaining


async def has_credits(db: AsyncSession, org_id: Optional[str]) -> bool:
    """Soft gate for the TB Tier-3 tail: True if the org may spend on the LLM.
    Always True when the cap is disabled or the org is unmetered."""
    if not AI_CREDIT_CAP_ENABLED:
        return True
    remaining = await remaining_credits(db, org_id)
    if remaining is None:
        return True
    return remaining > AI_CREDIT_FLOOR


async def ensure_credits(db: AsyncSession, org_id: Optional[str]) -> None:
    """Hard gate for the user-facing features: raise HTTP 402 when the org is
    over its cap. No-op when the cap is disabled or the org is unmetered."""
    if not AI_CREDIT_CAP_ENABLED:
        return
    remaining = await remaining_credits(db, org_id)
    if remaining is None:
        return
    if remaining <= AI_CREDIT_FLOOR:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "code": "quota_exceeded",
                "message": (
                    "Your organization's AI usage limit has been reached. "
                    "Please contact your administrator to add more credits."
                ),
                "remaining_credits": remaining,
            },
        )


# ---------------------------------------------------------------------------
# Read summary + admin management (powers the UI meter + the admin "AI Caps" tab)
# ---------------------------------------------------------------------------
# Allowance + used + period boundaries in one shot. The single-row derived table
# LEFT JOINed to ai_org_quota guarantees exactly one result even when the org has
# no row yet (COALESCE then fills the constant default / zero used). The monthly
# reset is applied in-query, matching enforcement's date_trunc('month', now()).
_SELECT_QUOTA_SUMMARY = text(
    """
    SELECT
        COALESCE(q.monthly_credit_allowance, :default_allowance) AS allowance,
        COALESCE(
            CASE WHEN q.period_start < date_trunc('month', now())::date
                 THEN 0 ELSE q.credits_used_this_period END,
            0
        ) AS used,
        date_trunc('month', now())::date AS period_start,
        (date_trunc('month', now()) + interval '1 month')::date AS resets_on
    FROM (SELECT CAST(:org AS varchar) AS org) s
    LEFT JOIN ai_org_quota q ON q.organization_id = s.org
    """
)

# Set one org's allowance (per-org override), lazily creating the row. Usage +
# period are preserved; the RETURNing CASE zeroes a stale prior-month period.
_SET_ALLOWANCE = text(
    """
    INSERT INTO ai_org_quota
        (organization_id, monthly_credit_allowance, period_start,
         credits_used_this_period, status, updated_at)
    VALUES
        (:org, :allowance, date_trunc('month', now())::date, 0, 'active', now())
    ON CONFLICT (organization_id) DO UPDATE SET
        monthly_credit_allowance = EXCLUDED.monthly_credit_allowance,
        updated_at = now()
    RETURNING monthly_credit_allowance,
        CASE WHEN period_start < date_trunc('month', now())::date
             THEN 0 ELSE credits_used_this_period END AS used
    """
)


async def get_org_credit_summary(db: AsyncSession, org_id: Optional[str]) -> dict:
    """Everything the UI needs to render "X% used · resets <date>" for one org.
    Returns {"metered": False} when metering is off or there is no org id (the UI
    hides the meter); otherwise the full breakdown."""
    if not AI_USAGE_METERING_ENABLED or not org_id:
        return {"metered": False}
    try:
        row = (
            await db.execute(
                _SELECT_QUOTA_SUMMARY,
                {"org": str(org_id), "default_allowance": AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT},
            )
        ).first()
    except Exception as exc:  # never break the caller over a display read
        logger.warning("usage_meter.get_org_credit_summary read failed (%s).", exc)
        return {"metered": False}
    allowance = int(row[0])
    used = int(row[1])
    remaining = max(0, allowance - used)
    pct_used = round(min(100.0, used / allowance * 100.0), 1) if allowance > 0 else 100.0
    return {
        "metered": True,
        "cap_enforced": AI_CREDIT_CAP_ENABLED,
        "allowance": allowance,
        "used": used,
        "remaining": remaining,
        "pct_used": pct_used,
        "period_start": row[2],
        "resets_on": row[3],
    }


async def set_org_allowance(
    db: AsyncSession, org_id: str, monthly_credit_allowance: int
) -> dict:
    """Admin: set one org's monthly credit allowance (per-org override of the
    constant default). Preserves the current period's usage, and refreshes the
    balance cache so new headroom is honoured immediately by the pre-flight gate."""
    allowance = max(0, int(monthly_credit_allowance))
    row = (
        await db.execute(_SET_ALLOWANCE, {"org": str(org_id), "allowance": allowance})
    ).first()
    await db.commit()
    used = int(row[1]) if row else 0
    remaining = max(0, allowance - used)
    _balance_cache.set(str(org_id), remaining)  # keep ensure_credits/has_credits in sync
    return {
        "organization_id": str(org_id),
        "allowance": allowance,
        "used": used,
        "remaining": remaining,
    }


async def list_org_quotas(
    db: AsyncSession,
    organization_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """Admin table rows: every org that has a quota row (created lazily on first
    AI use, or via an explicit override), with the monthly reset applied. Orgs
    with no row simply run on the constant default and aren't listed here."""
    where = "WHERE organization_id = :org" if organization_id else ""
    filt = {"org": str(organization_id)} if organization_id else {}
    total = (
        await db.execute(text(f"SELECT count(*) FROM ai_org_quota {where}"), filt)
    ).scalar() or 0
    rows = (
        await db.execute(
            text(
                f"""
                SELECT organization_id,
                       monthly_credit_allowance AS allowance,
                       CASE WHEN period_start < date_trunc('month', now())::date
                            THEN 0 ELSE credits_used_this_period END AS used
                FROM ai_org_quota {where}
                ORDER BY organization_id
                LIMIT :limit OFFSET :offset
                """
            ),
            {**filt, "limit": page_size, "offset": (page - 1) * page_size},
        )
    ).all()
    items = [
        {
            "organization_id": r[0],
            "allowance": int(r[1]),
            "used": int(r[2]),
            "remaining": max(0, int(r[1]) - int(r[2])),
        }
        for r in rows
    ]
    return {"items": items, "total": int(total), "page": page, "page_size": page_size}
