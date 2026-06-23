"""
Trial-balance auto-mapping engine (ticket C-3) — the 3-tier cascade.

For each unmapped TB account, suggest a chart-of-account from the supplied
candidate set, stopping at the first confident tier:

  Tier 1  "previous data"      — exact code+name / exact code / fuzzy name match
                                 against this client's prior-year confirmed
                                 mappings and the org's memory (NO LLM).
  Tier 2  "organization trend" — semantic similarity (local embeddings) to the
                                 candidate COA labels, blended with a frequency
                                 prior from org/sector memory neighbours
                                 (NO LLM).
  Tier 3  "LLM tail"           — the low-confidence residue is batched into ONE
                                 structured Gemini call. Gated by ``use_llm_tail``
                                 (default OFF) so the engine is fully functional
                                 and testable without any Gemini quota.

A suggestion only ever references a ``coa_original_id`` from the supplied
candidate set. The engine reads auditai's own Postgres memory; it never writes
1audit and contains no update/delete.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

import config
import structured
import tb_mapping_memory as memory
from copilot_cache import TTLCache
from embeddings import embed_queries, embed_texts

logger = logging.getLogger(__name__)

# Confidence assigned by each deterministic (Tier-1) outcome.
_CONF_EXACT = 0.97        # same code AND same normalised name
_CONF_CODE = 0.90         # same account code, different name
_CONF_FUZZY_HI = 0.88     # very close name
_CONF_FUZZY_LO = 0.74     # close name
_FUZZY_HI = 0.92
_FUZZY_LO = 0.85
# Tier-2 routing.
_TIER2_MAX = 0.85         # semantic confidence is capped below a code/exact hit
_TIER3_FLOOR = 0.55       # below this, defer to the LLM tail (when enabled)
_AMBIGUOUS_MARGIN = 0.05  # top vs 2nd blended gap that still counts as "clear"
_MAX_SHORTLIST = 5        # candidates handed to the LLM per account
_LLM_TAIL_BATCH = 15      # accounts per Gemini call — bounds output so JSON can't truncate
# Large-dataset guards for the LLM tail (the deterministic Tiers 1/2 are already
# fast; Gemini is the slow part). A big low-confidence tail would otherwise fire
# dozens of sequential calls and/or hang on one slow response.
_LLM_TAIL_MAX_ACCOUNTS = int(config.COPILOT_TB_LLM_TAIL_MAX)  # cap accounts sent to Gemini per request (0 = no cap)
_LLM_TAIL_TIMEOUT_SEC = float(config.COPILOT_TB_LLM_TIMEOUT_SEC)  # per-sub-batch wall-clock cap → degrade, never hang
_LLM_TAIL_CONCURRENCY = max(1, int(config.COPILOT_TB_LLM_CONCURRENCY))  # sub-batches in flight (1=sequential/free-tier-safe)
# Tier-3 blend: the LLM pick is cross-checked against the provisional Tier-2 guess
# rather than blindly overriding it.
_LLM_AGREE_MAX = 0.90     # cap when LLM and semantic agree (still < a code/exact hit)
_LLM_AGREE_BONUS = 0.10   # confidence bump when both signals agree
_LLM_CONFLICT_CAP = 0.60  # cap when they disagree → stays below bulk-accept, for review

# Caches ONLY the expensive Gemini picks (deterministic Tiers 1/2 recompute every
# call). Keyed by account content + candidate shortlist + language, so re-running
# "AI auto map" on the same accounts doesn't re-spend quota. See config note.
_LLM_TAIL_CACHE = TTLCache(ttl_seconds=config.COPILOT_TB_LLM_CACHE_TTL_SEC, max_entries=512)


def _blend_llm_pick(prov: dict, cid: int, llm_conf, rationale: str, label) -> tuple:
    """Blend a (non-null) LLM pick with the provisional Tier-2 suggestion, instead
    of blindly overriding it. Returns ``(confidence, rationale)``:

      - agreement (LLM picks the same COA as Tier-2): two independent signals
        concur → raise confidence (bonus, capped at ``_LLM_AGREE_MAX``);
      - conflict (LLM picks a different COA): trust the LLM's shortlist reasoning
        but cap confidence at ``_LLM_CONFLICT_CAP`` so it stays below the
        bulk-accept threshold and a human reviews it.
    """
    t2_cid = prov.get("coa_original_id")
    t2_conf = float(prov.get("confidence") or 0.0)
    if t2_cid is not None and cid == t2_cid:
        conf = min(_LLM_AGREE_MAX, max(t2_conf, float(llm_conf or 0.0)) + _LLM_AGREE_BONUS)
        r = f"Semantic + LLM agree: {rationale or label or 'shortlisted candidate'}"
    else:
        conf = min(float(llm_conf or 0.0), _LLM_CONFLICT_CAP)
        pick = label if label else cid
        r = f"LLM chose “{pick}” over the semantic guess — please review."
        if rationale:
            r += f" {rationale}"
    return conf, r


def _llm_cache_key(account, shortlist, language: str) -> str:
    """Content key for a tail account's LLM pick: normalised name + the exact
    candidate shortlist it was offered + language. Independent of tb_account_id
    so the same account in a later request reuses the cached pick."""
    norm = memory.normalise_account(
        getattr(account, "account_name", None), getattr(account, "account_name_sl", None)
    )
    ids = ",".join(str(c) for c in sorted(shortlist))
    return f"{language}|{norm}|{ids}"


def _to_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _fuzzy(a: str, b: str) -> float:
    """Order-insensitive lexical similarity in [0,1], dependency-free:
    max(token-set Jaccard, char-level ratio). Catches 'trade receivables' vs
    'receivables - trade' that strict equality misses."""
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    jacc = len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0
    seq = difflib.SequenceMatcher(None, a, b).ratio()
    return max(jacc, seq)


def _cos_matrix(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of a query vector against every row of ``mat``."""
    if mat.size == 0 or vec is None:
        return np.zeros((0,))
    mn = np.linalg.norm(mat, axis=1) + 1e-9
    vn = float(np.linalg.norm(vec)) + 1e-9
    return (mat @ vec) / (mn * vn)


def _mk(account, coa_id: Optional[int], confidence: float, tier: str, rationale: str) -> dict:
    return {
        "trial_balance_account_id": _to_int(getattr(account, "tb_account_id", None)),
        "coa_original_id": (_to_int(coa_id) if coa_id is not None else None),
        "confidence": round(float(confidence), 3),
        "tier": tier,
        "rationale": rationale,
    }


def _tier2_rationale(sem: float, votes: float, sector: Optional[str], label: Optional[str]) -> str:
    bits = [f"Closest match to “{label}”" if label else "Closest semantic match"]
    bits.append(f"similarity {sem:.0%}")
    if votes > 0:
        where = f"same-sector ({sector})" if sector else "firm"
        bits.append(f"and {where} history maps similar accounts here")
    return "; ".join(bits) + "."


# --- Tier-3 structured-output schema (only used when use_llm_tail=True) ---
# NOTE: NO field defaults here. Gemini's response_schema is a Vertex Schema proto
# that has no `default` keyword — a Pydantic default leaks a `default` key into the
# generated schema and the SDK rejects it ("Unknown field for Schema: default"),
# which silently degraded the whole tail to Tier-2. Keep every field required.
class _LlmMapping(BaseModel):
    tb_account_id: int
    coa_original_id: Optional[int]
    confidence: float
    rationale: str


class _LlmResult(BaseModel):
    mappings: List[_LlmMapping]


_LLM_SYSTEM = (
    "You are an audit assistant mapping trial-balance accounts to a chart of "
    "accounts. For each account, choose the SINGLE best coa_original_id ONLY "
    "from that account's provided candidate shortlist. If none fit, return null. "
    "Never invent an id outside the shortlist. Give a one-line rationale and a "
    "0..1 confidence. Respond as JSON."
)


def _build_llm_prompt(tail: list, coa_index: dict, language: str) -> str:
    lines = [f"Language: {language}", "Map each account to one candidate id (or null):", ""]
    for (_idx, a, shortlist) in tail:
        amt = getattr(a, "cy_amount", None)
        sign = ""
        if amt is not None:
            try:
                sign = " (credit/negative balance)" if float(amt) < 0 else " (debit/positive balance)"
            except (TypeError, ValueError):
                sign = ""
        nm = a.account_name or ""
        if getattr(a, "account_name_sl", None):
            nm = f"{nm} / {a.account_name_sl}"
        lines.append(f"- tb_account_id={a.tb_account_id} | code={a.account_code or ''} | name=\"{nm}\"{sign}")
        for cid in shortlist:
            meta = coa_index.get(cid, {})
            lines.append(
                f"    candidate coa_original_id={cid} | label=\"{meta.get('label','')}\""
                f" | group={meta.get('group') or ''} | type={meta.get('type') or ''}"
            )
        lines.append("")
    return "\n".join(lines)


async def _run_llm_tail(tail: list, coa_index: dict, valid_ids: set, language: str) -> dict:
    """Resolve the deferred accounts with structured Gemini calls, in sub-batches
    so the JSON response never overruns the output-token budget (a large tail in
    ONE call truncates → invalid JSON → the whole tail is lost). Returns a map
    tb_account_id -> _LlmMapping. Best-effort: a failed sub-batch is skipped (those
    accounts keep their Tier-2 guess), the rest still resolve."""
    if not tail:
        return {}
    # Cap the number of accounts that reach Gemini so a huge tail can't balloon into
    # dozens of calls. The overflow keeps its (already-computed) Tier-2 guess.
    full = len(tail)
    if _LLM_TAIL_MAX_ACCOUNTS > 0 and full > _LLM_TAIL_MAX_ACCOUNTS:
        logger.info(
            "tb-map Tier-3: capping LLM tail to %d of %d accounts (rest keep Tier-2)",
            _LLM_TAIL_MAX_ACCOUNTS, full,
        )
        tail = tail[:_LLM_TAIL_MAX_ACCOUNTS]

    subs = [tail[i:i + _LLM_TAIL_BATCH] for i in range(0, len(tail), _LLM_TAIL_BATCH)]
    # Bounded concurrency: a Semaphore(1) keeps this sequential (free-tier-safe);
    # raise COPILOT_TB_LLM_CONCURRENCY on a paid key to run sub-batches in parallel
    # so a large tail resolves in ~one call's time instead of N.
    sem = asyncio.Semaphore(_LLM_TAIL_CONCURRENCY)

    async def _run_sub(n: int, sub: list) -> dict:
        async with sem:
            prompt = _build_llm_prompt(sub, coa_index, language)
            try:
                # Per-sub-batch wall-clock cap so a slow/unresponsive Gemini degrades
                # to Tier-2 instead of hanging the whole mapping request.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        structured.generate_structured, prompt, _LlmResult,
                        system=_LLM_SYSTEM, max_output_tokens=8192,
                    ),
                    timeout=_LLM_TAIL_TIMEOUT_SEC,
                )
            except Exception as exc:  # timeout / quota / network / truncation — degrade
                logger.warning(
                    "tb-map Tier-3 LLM sub-batch %d/%d unavailable (keeping Tier-2): %s",
                    n + 1, len(subs), exc,
                )
                return {}
            res = {}
            for m in result.mappings or []:
                cid = _to_int(m.coa_original_id)
                if cid is not None and cid not in valid_ids:
                    cid = None  # never trust an id outside the shortlist/candidate set
                res[_to_int(m.tb_account_id)] = (cid, m.confidence, m.rationale)
            return res

    parts = await asyncio.gather(*[_run_sub(n, sub) for n, sub in enumerate(subs)])
    out = {}
    for p in parts:
        out.update(p)
    return out


async def map_accounts(
    db,
    *,
    organization_id: Optional[str],
    client_sector: Optional[str],
    accounts: list,
    coa: list,
    prior_mappings: Optional[list] = None,
    prime_org_id: Optional[str] = None,
    use_llm_tail: bool = False,
    language: str = "en",
) -> List[dict]:
    """Return one suggestion dict per account (same order as ``accounts``):
    ``{trial_balance_account_id, coa_original_id, confidence, tier, rationale}``."""
    prior_mappings = prior_mappings or []
    if not accounts:
        return []

    # --- candidate COA index (the only ids we may suggest) ---
    valid_ids: set = set()
    coa_index: dict = {}
    cand_ids: List[int] = []
    cand_norm_labels: List[str] = []
    for c in coa or []:
        cid = _to_int(getattr(c, "coa_original_id", None))
        if cid is None or cid in valid_ids:
            continue
        valid_ids.add(cid)
        coa_index[cid] = {
            "label": getattr(c, "label", None),
            "group": getattr(c, "group", None),
            "type": getattr(c, "type", None),
        }
        cand_ids.append(cid)
        cand_norm_labels.append(memory.normalise_account(getattr(c, "label", None)))
    label_embs = (
        np.asarray(await embed_texts(cand_norm_labels), dtype=float)
        if cand_norm_labels else np.zeros((0, 0))
    )

    # --- prior-year confirmed mappings (this client's "previous data") ---
    prior_list: List[tuple] = []          # (norm_name, code, coa_id)
    prior_by_code: dict = defaultdict(list)
    for pm in prior_mappings:
        cid = _to_int(getattr(pm, "coa_original_id", None))
        if cid is None or cid not in valid_ids:
            continue  # only suggest ids that exist in the current template
        nm = memory.normalise_account(
            getattr(pm, "account_name", None),
            getattr(pm, "account_name_sl", None),
            getattr(pm, "account_code", None),
        )
        code = str(pm.account_code) if getattr(pm, "account_code", None) not in (None, "") else ""
        prior_list.append((nm, code, cid))
        if code:
            prior_by_code[code].append(cid)

    # --- batch-embed account queries once ---
    acc_norms = [
        memory.normalise_account(a.account_name, getattr(a, "account_name_sl", None), a.account_code)
        for a in accounts
    ]
    acc_embs = await embed_queries(acc_norms) if acc_norms else []

    # One cheap probe instead of thousands of futile lookups: a brand-new org (the
    # large-dataset worst case) has no memory, so skip every per-account memory
    # query (Tier-1c code lookup + Tier-2 neighbour search). Behaviour is identical
    # to those queries returning empty — just without N round-trips.
    mem_enabled = await memory.org_has_mappings(
        db, organization_id=organization_id, prime_org_id=prime_org_id
    )

    results_by_idx: dict = {}
    tail: list = []  # accounts deferred to Tier-3: (idx, account, shortlist_ids)

    for idx, a in enumerate(accounts):
        nm = acc_norms[idx]
        code = str(a.account_code) if getattr(a, "account_code", None) not in (None, "") else ""
        sugg = None

        # ===== TIER 1 — previous data =====
        # 1a) exact code + name in this client's prior mappings
        for (pnm, pcode, pcoa) in prior_list:
            if pcode and pcode == code and pnm == nm:
                sugg = _mk(a, pcoa, _CONF_EXACT, "memory",
                           f"Matches this client's prior-year mapping (code {code}, same name).")
                break
        # 1b) exact code (unambiguous) in prior mappings
        if sugg is None and code and code in prior_by_code:
            distinct = set(prior_by_code[code])
            if len(distinct) == 1:
                sugg = _mk(a, next(iter(distinct)), _CONF_CODE, "memory",
                           f"Account code {code} was mapped here in the prior year.")
        # 1c) exact code in org/prime memory
        if sugg is None and code and mem_enabled:
            try:
                mem_code = await memory.find_by_code(
                    db, organization_id=organization_id, account_code=code, prime_org_id=prime_org_id
                )
            except Exception as exc:
                logger.warning("tb-map Tier-1 code lookup failed: %s", exc)
                mem_code = []
            mem_code = [m for m in mem_code if m.coa_original_id in valid_ids]
            if mem_code:
                cid, _n = Counter(m.coa_original_id for m in mem_code).most_common(1)[0]
                sugg = _mk(a, cid, _CONF_CODE, "memory",
                           f"Account code {code} has been mapped to this account before.")
        # 1d) fuzzy name over prior mappings
        if sugg is None and prior_list:
            best_score, best_cid = 0.0, None
            for (pnm, _pc, pcoa) in prior_list:
                s = _fuzzy(nm, pnm)
                if s > best_score:
                    best_score, best_cid = s, pcoa
            if best_cid is not None and best_score >= _FUZZY_HI:
                sugg = _mk(a, best_cid, _CONF_FUZZY_HI, "memory",
                           f"Closely matches a prior-year account name ({best_score:.0%}).")
            elif best_cid is not None and best_score >= _FUZZY_LO:
                sugg = _mk(a, best_cid, _CONF_FUZZY_LO, "memory",
                           f"Resembles a prior-year account name ({best_score:.0%}).")

        # ===== TIER 2 — organization trend + semantics =====
        if sugg is None:
            qv = np.asarray(acc_embs[idx], dtype=float) if acc_embs else None
            sims = _cos_matrix(label_embs, qv) if label_embs.size else np.zeros((0,))
            votes: dict = defaultdict(float)
            neighbors = []
            if mem_enabled:
                try:
                    neighbors = await memory.search_mappings(
                        db, organization_id=organization_id, client_sector=client_sector,
                        account_name=a.account_name, account_name_sl=getattr(a, "account_name_sl", None),
                        account_code=a.account_code, k=8, prime_org_id=prime_org_id,
                        embedding=(acc_embs[idx] if acc_embs else None),  # reuse the batch embed
                    )
                except Exception as exc:
                    logger.warning("tb-map Tier-2 search failed: %s", exc)
                    neighbors = []
            for rank, n in enumerate(neighbors):
                if n.coa_original_id in valid_ids:
                    votes[n.coa_original_id] += 1.0 / (1.0 + rank)

            if sims.size:
                vote_max = max(votes.values()) if votes else 0.0
                blended = []
                for i, cid in enumerate(cand_ids):
                    sem = max(0.0, float(sims[i]))
                    if vote_max > 0:
                        # Firm has history for similar accounts → blend semantic
                        # similarity with the frequency prior.
                        vote = votes.get(cid, 0.0) / vote_max
                        score = 0.70 * sem + 0.30 * vote
                    else:
                        # No memory signal (new firm / novel account) → semantic
                        # similarity carries full weight, so a strong label match
                        # isn't penalised for the absent prior.
                        score = sem
                    blended.append((score, sem, cid))
                blended.sort(reverse=True)
                top_score, top_sem, top_cid = blended[0]
                second = blended[1][0] if len(blended) > 1 else 0.0
                conf = min(_TIER2_MAX, top_score)
                if (top_score - second) < _AMBIGUOUS_MARGIN:
                    conf *= 0.85  # ambiguous between two COAs → soften
                rationale = _tier2_rationale(
                    top_sem, votes.get(top_cid, 0.0), client_sector, coa_index.get(top_cid, {}).get("label")
                )
                sugg = _mk(a, top_cid, conf, "semantic", rationale)
                if use_llm_tail and conf < _TIER3_FLOOR:
                    shortlist = [cid for _s, _se, cid in blended[:_MAX_SHORTLIST]]
                    tail.append((idx, a, shortlist))  # provisional sugg stays if LLM unavailable

        if sugg is None:
            sugg = _mk(a, None, 0.0, "none", "No confident match found; please map manually.")
        results_by_idx[idx] = sugg

    # ===== TIER 3 — LLM tail (gated; BLENDS with the provisional Tier-2 guess) =====
    if use_llm_tail and tail:
        id_to_idx = {a.tb_account_id: idx for (idx, a, _s) in tail}

        # Split the tail into cache hits (zero quota) and misses; only misses go to
        # Gemini, still as one batched call. A cached `None` pick (LLM abstained) is
        # honoured too, so we don't re-ask about genuinely-hard accounts.
        llm: dict = {}
        misses: list = []
        key_by_tbid: dict = {}
        for (idx, a, shortlist) in tail:
            key = _llm_cache_key(a, shortlist, language)
            key_by_tbid[a.tb_account_id] = key
            cached = _LLM_TAIL_CACHE.get(key)
            if cached is not None:
                llm[a.tb_account_id] = cached
            else:
                misses.append((idx, a, shortlist))
        if misses:
            fresh = await _run_llm_tail(misses, coa_index, valid_ids, language)
            for tb_id, val in fresh.items():
                llm[tb_id] = val
                k = key_by_tbid.get(tb_id)
                if k is not None:
                    _LLM_TAIL_CACHE.set(k, val)

        # Blend each LLM pick with the provisional Tier-2 suggestion instead of a
        # blind override: agreement raises confidence, conflict caps it for review.
        for tb_id, (cid, llm_conf, rationale) in llm.items():
            idx = id_to_idx.get(tb_id)
            if idx is None:
                continue
            if cid is None:
                continue  # LLM abstained → keep the provisional Tier-2 guess
            label = coa_index.get(cid, {}).get("label")
            conf, r = _blend_llm_pick(results_by_idx[idx], cid, llm_conf, rationale, label)
            results_by_idx[idx] = _mk(accounts[idx], cid, conf, "llm", r)

    return [results_by_idx[i] for i in range(len(accounts))]
