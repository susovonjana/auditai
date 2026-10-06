"""
Central configuration loader.
Reads environment variables from .env and exposes them as constants.

STACK:
  - LLM:        AWS Bedrock (Claude) — tiered: a Haiku-class "fast" model for
                cheap high-frequency calls, a Sonnet-class "smart" model for
                generation/chat/TB-tail. Auth via the standard AWS credential
                chain (access keys locally, the ECS task role in prod).
  - Embeddings: sentence-transformers locally (multilingual-e5-small, 384-dim)
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load env file based on APP_ENV.
#   APP_ENV=local      -> .env.local
#   APP_ENV=production -> .env.production
#   unset              -> .env  (backwards-compat default)
BASE_DIR = Path(__file__).resolve().parent
_env_name = os.getenv("APP_ENV", "").strip().lower()
_env_file = BASE_DIR / (f".env.{_env_name}" if _env_name else ".env")
# override=True: the deployed .env is the runtime source of truth, so it wins
# over any stale value already in the process environment (e.g. a key baked in
# via `docker run -e GEMINI_API_KEY=…`). Without this, editing .env and
# restarting would silently keep using the old baked value.
load_dotenv(_env_file, override=True)

# --- AWS Bedrock (LLM provider) ---
# Region the Bedrock runtime lives in. Defaults to AWS_REGION (the standard
# variable boto3 reads) and falls back to eu-west-1 (same region as the RDS).
BEDROCK_REGION: str = os.getenv("AWS_REGION") or os.getenv("BEDROCK_REGION") or "eu-west-1"
# Credentials are NOT read here — the AnthropicBedrock client resolves them via
# the standard AWS chain: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (set in
# .env.local for dev) or the ECS task role in production. We only surface a
# boolean for a clean "creds missing" diagnostic at call time.
HAS_AWS_STATIC_KEYS: bool = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))

# --- Database ---
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/auditai_db",
)
# SSL for RDS in production. Leave DB_SSL_MODE empty for local dev (no SSL).
# For Aurora/RDS set DB_SSL_MODE=verify-full and ship the AWS RDS CA bundle at
# DB_SSL_ROOT_CERT (the Dockerfile downloads it to /etc/ssl/certs/rds-global-bundle.pem).
DB_SSL_MODE: str = os.getenv("DB_SSL_MODE", "")
DB_SSL_ROOT_CERT: str = os.getenv(
    "DB_SSL_ROOT_CERT", "/etc/ssl/certs/rds-global-bundle.pem"
)

# --- JWT ---
JWT_SECRET_KEY: str = os.getenv(
    "JWT_SECRET_KEY",
    "change-me-in-production-this-is-not-secure-at-all-please-replace",
)
JWT_EXPIRE_HOURS: int = int(os.getenv("JWT_EXPIRE_HOURS", "8"))
JWT_ALGORITHM: str = "HS256"

# --- Initial Admin ---
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")

# --- CORS ---
FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# --- 1audit integration (copilot data callbacks) ---
# Base URL of 1audit-be's internal copilot API. The copilot tools HTTP-GET
# {ONEAUDIT_BASE_URL}/copilot/audit_files/{id}/... presenting an X-Copilot-Grant
# header. In Docker, reach the host's 1audit-be via host.docker.internal.
def _resolve_oneaudit_base_url(raw: str) -> str:
    """The same .env is used in Docker (where 1audit-be is reached via
    host.docker.internal) and when uvicorn runs directly on the developer's
    machine (where that name does not resolve and EVERY be call would fail with
    'nodename nor servname provided'). If the configured host can't be resolved
    here, fall back to localhost so a host-run server still reaches be."""
    import socket
    from urllib.parse import urlparse, urlunparse

    try:
        parts = urlparse(raw)
        host = parts.hostname or ""
        if host and host != "localhost":
            socket.gethostbyname(host)
        return raw
    except (socket.gaierror, UnicodeError):
        pass
    except Exception:  # pragma: no cover - never let URL sniffing break startup
        return raw
    netloc = f"localhost:{parts.port}" if parts.port else "localhost"
    fixed = urlunparse(parts._replace(netloc=netloc))
    import logging
    logging.getLogger("config").warning(
        "ONEAUDIT_BASE_URL host %r does not resolve on this machine; using %s", parts.hostname, fixed
    )
    return fixed


ONEAUDIT_BASE_URL: str = _resolve_oneaudit_base_url(
    os.getenv("ONEAUDIT_BASE_URL", "http://localhost:5030/api/v1/internal")
)
# Timeout (seconds) for copilot data callbacks to 1audit-be. Some read services
# (e.g. a lead-sheet-heavy working paper's content) take ~20s, so allow headroom.
# Read timeout for ONE internal be data call. Kept tight so a stalled be call
# fails fast and the tool loop proceeds (or says it couldn't fetch) instead of
# freezing the whole answer — a single hung call used to block ~45s. Connect
# timeout is separate (short) so a down be is detected immediately.
ONEAUDIT_HTTP_TIMEOUT: int = int(os.getenv("ONEAUDIT_HTTP_TIMEOUT", "15"))
ONEAUDIT_HTTP_CONNECT_TIMEOUT: int = int(os.getenv("ONEAUDIT_HTTP_CONNECT_TIMEOUT", "5"))

# --- Help-center "Learn more" links ---
# The knowledge base stores only a help-page SLUG (e.g. "settings-general-
# information") per chunk — language- and domain-agnostic. The full clickable
# URL is assembled at ANSWER time from the answer's language, because the live
# help route is language-prefixed (FE: /{lang}/app/knowledge_base/helps/{slug}).
# {lang} → "en"|"ar", {slug} → the stored slug. Override per environment (prod
# domain) via env. Set to "" to disable Learn-more links entirely.
HELP_CENTER_URL_TEMPLATE: str = os.getenv(
    "HELP_CENTER_URL_TEMPLATE",
    "https://1audit.com/{lang}/app/knowledge_base/helps/{slug}",
)

# Shared secret used by 1audit-be to SIGN copilot grants (HS256). When set here
# (must MATCH be's COPILOT_GRANT_SECRET / JWT_LOGIN_SECRET), auditai verifies the
# grant's signature locally — no per-question /summary round-trip to be. When
# unset, auditai still checks the grant's expiry + scope + file locally (be
# remains the hard enforcer on every tool call).
COPILOT_GRANT_SECRET: str = os.getenv("COPILOT_GRANT_SECRET", "") or ""

# Shared secret protecting auditai's INTERNAL endpoints (e.g. the "file changed"
# cache-invalidation ping from 1audit-be). 1audit-be must send the SAME value in
# the X-Internal-Secret header (its AUDITAI_INTERNAL_SECRET). When unset, the
# internal endpoints are disabled (fail closed).
INTERNAL_SHARED_SECRET: str = os.getenv("INTERNAL_SHARED_SECRET", "") or ""

# Shared secret protecting the SUPPORT-DESK endpoints (/support/*), which the
# support system's backend calls server-to-server (its AUDITAI_SUPPORT_SECRET,
# sent in the X-Support-Secret header). Deliberately distinct from
# INTERNAL_SHARED_SECRET so the support desk cannot reach /internal/*. When
# unset, /support/* answers 503 (fail closed).
SUPPORT_SHARED_SECRET: str = os.getenv("SUPPORT_SHARED_SECRET", "") or ""

# Short-TTL cache for the file-mode chat's data fetches. Within this window a
# repeated tool call (same file + endpoint + args) is served from memory instead
# of re-querying 1audit-be, so a burst of questions doesn't re-fetch the same
# trial balance / summary each time. The grant is still re-validated once per
# chat request, so caching never bypasses authorization. Set to 0 to disable.
# File-data cache lifetime. Now a SAFETY BACKSTOP: real freshness comes from the
# per-file "changed" flag (1audit-be pings auditai on any edit → that file's cache
# is cleared on the next question). So idle files can cache long without going
# stale, and edits show immediately. This TTL only catches a missed/dropped ping.
COPILOT_DATA_CACHE_TTL_SEC: int = int(os.getenv("COPILOT_DATA_CACHE_TTL_SEC", "1800"))

# Output-token cap for the in-file chat tool loop. File-data answers are short
# (## Answer + ≤3 follow-ups), so a tight cap lowers worst-case generation time
# without truncating real answers — distinct from the 4096 default used elsewhere.
COPILOT_CHAT_MAX_TOKENS: int = int(os.getenv("COPILOT_CHAT_MAX_TOKENS", "1500"))

# Phase-2 per-file RAG (search_file). The index is rebuilt lazily when the file's
# change-signature differs; this TTL is a BACKSTOP that forces a re-check even when
# the signature is unchanged (catches edits that don't bump a working-paper row).
COPILOT_FILE_INDEX_TTL_SEC: int = int(os.getenv("COPILOT_FILE_INDEX_TTL_SEC", "600"))
# How many narrative chunks to retrieve before reranking, and how many to return.
COPILOT_FILE_SEARCH_CANDIDATES: int = int(os.getenv("COPILOT_FILE_SEARCH_CANDIDATES", "30"))
COPILOT_FILE_SEARCH_TOP_K: int = int(os.getenv("COPILOT_FILE_SEARCH_TOP_K", "6"))
# Safety bound on how many chunks one file's index may hold.
COPILOT_FILE_INDEX_MAX_CHUNKS: int = int(os.getenv("COPILOT_FILE_INDEX_MAX_CHUNKS", "600"))

# TTL for caching the Tier-3 LLM mapping picks in tb_mapping_engine, so re-running
# "AI auto map" over the same accounts (re-clicks, re-maps) doesn't re-spend the
# scarce Gemini quota — the deterministic Tiers 1/2 still recompute every call.
# Keyed by account content + candidate shortlist, so it's safe across requests.
# Set to 0 to disable.
COPILOT_TB_LLM_CACHE_TTL_SEC: int = int(os.getenv("COPILOT_TB_LLM_CACHE_TTL_SEC", "3600"))

# Large-dataset guards for the Tier-3 LLM tail. The deterministic Tiers 1/2 are
# fast; Gemini is the slow, quota-bound part. On a big trial balance the low-conf
# tail can be hundreds of accounts — without these it would fire dozens of
# sequential Gemini calls (slow + quota) and a single slow response would hang the
# whole request. Cap how many accounts reach Gemini per request (0 = unlimited,
# for a billing-enabled key) and bound each sub-batch call's wall-clock so it
# degrades to Tier-2 instead of hanging.
COPILOT_TB_LLM_TAIL_MAX: int = int(os.getenv("COPILOT_TB_LLM_TAIL_MAX", "45"))
COPILOT_TB_LLM_TIMEOUT_SEC: int = int(os.getenv("COPILOT_TB_LLM_TIMEOUT_SEC", "20"))
# How many LLM sub-batches may be in flight at once. 1 = sequential (safe for the
# free-tier key, whose low rate limit would 429 on concurrent calls). On a paid
# key raise this (e.g. 5) so a large tail resolves in parallel — roughly the time
# of one call instead of N. Pair with COPILOT_TB_LLM_TAIL_MAX=0 to LLM the whole tail.
COPILOT_TB_LLM_CONCURRENCY: int = int(os.getenv("COPILOT_TB_LLM_CONCURRENCY", "1"))

# Reserved "prime" organization id holding a curated golden TB-mapping seed in
# tb_mapping_memory (ticket C-2/C-3). When a brand-new org/client has no history
# of its own, the auto-map engine falls back to this seed via the SAME search
# path. The 1audit-be bridge usually passes prime_org_id explicitly; this is the
# default. Empty = no cold-start seed (Tier-2 then relies on candidate labels).
COPILOT_PRIME_MEMORY_ORG_ID: Optional[str] = os.getenv("COPILOT_PRIME_MEMORY_ORG_ID") or None

# --- Uploads ---
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
_upload_dir = os.getenv("UPLOAD_DIR", "uploads")
UPLOAD_DIR: Path = Path(_upload_dir) if os.path.isabs(_upload_dir) else BASE_DIR / _upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Models (AWS Bedrock / Claude) ---
# Two cost tiers. Each is a comma-separated failover list (left-most first):
# when a model throttles (Bedrock 429), the next id is tried, then it cools
# down briefly so later requests skip straight past it.
#   * SMART — generation, chat, TB Tier-3 (quality-sensitive, lower volume)
#   * FAST  — intent classify, query expansion, translation (cheap, high volume)
# Use cross-region INFERENCE-PROFILE ids (the "eu." prefix) so capacity is
# pooled across EU regions. Defaults below are Sonnet 4.5 (smart) and Haiku 4.5
# (fast) — verified ACTIVE in eu-west-1. To change, pick another ACTIVE profile
# from `aws bedrock list-inference-profiles --region eu-west-1` and ensure the
# IAM principal (local) or ECS task role (prod) has bedrock:InvokeModel* for it.
def _model_list(raw: str, fallback: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()] or [fallback]

BEDROCK_MODEL_SMART_RAW: str = os.getenv(
    "BEDROCK_MODEL_SMART", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
BEDROCK_MODEL_FAST_RAW: str = os.getenv(
    "BEDROCK_MODEL_FAST", "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
)
BEDROCK_MODELS_SMART: list[str] = _model_list(
    BEDROCK_MODEL_SMART_RAW, "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
)
BEDROCK_MODELS_FAST: list[str] = _model_list(
    BEDROCK_MODEL_FAST_RAW, "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
)
# Map a tier name -> its failover list (llm.py reads this).
BEDROCK_TIER_MODELS: dict[str, list[str]] = {
    "smart": BEDROCK_MODELS_SMART,
    "fast": BEDROCK_MODELS_FAST,
}
# Seconds to skip a model after it throttles, so the next request jumps to the
# next id in that tier's chain.
BEDROCK_MODEL_COOLDOWN_SEC: int = int(os.getenv("BEDROCK_MODEL_COOLDOWN_SEC", "60"))
# Per-call retry budget for transient Bedrock throttling before giving up on a
# model and failing over to the next in the tier.
BEDROCK_MAX_RETRIES: int = int(os.getenv("BEDROCK_MAX_RETRIES", "2"))

# Local embeddings (sentence-transformers)
# multilingual-e5-small is 384-dim and multilingual (~100 langs incl. Arabic),
# so Arabic TB account names / COA labels / procedures match well. Same 384-dim
# as the old English-only models (bge-small-en / MiniLM) → no schema change, but
# switching models requires re-embedding all stored vectors (see reembed_all.py).
LOCAL_EMBEDDING_MODEL: str = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# Cross-encoder reranker (loaded once at first request). Multilingual
# (mMARCO, 14 languages incl. Arabic) so Arabic KB chunks are re-scored
# correctly after the e5 multilingual embeddings retrieve them. ~117M params
# (~470 MB on disk / ~250 MB resident) — heavier than the old English-only
# ms-marco-MiniLM-L-6-v2 (~22M). Scores at query time; nothing stored, so a
# model swap needs NO re-embedding — just rebuild the image to bake it in.
# English-only fallback at lower footprint: cross-encoder/ms-marco-MiniLM-L-6-v2.
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 100  # bigger overlap reduces "number split from label" cases

# --- Retrieval ---
# Hybrid: pull more candidates from each retriever, then rerank down to TOP_K.
INITIAL_CANDIDATES: int = 20
TOP_K_CHUNKS: int = 8
SIMILARITY_THRESHOLD: float = 0.22   # was 0.30; lower so close-but-not-exact matches still answer

# Multi-query expansion: Gemini rephrases the user's question into N variants
# (with different domain vocabulary) so vague questions still surface the right
# chunks. Reranker still scores against the ORIGINAL question.
USE_QUERY_EXPANSION: bool = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"
QUERY_EXPANSION_N: int = int(os.getenv("QUERY_EXPANSION_N", "3"))

# --- Conversation memory ---
CONVERSATION_MEMORY_TURNS: int = 5

# --- Allowed file extensions ---
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}

# --- Security: rate limits ---
# slowapi limit strings, see https://limits.readthedocs.io/en/stable/quickstart.html
RATE_LIMIT_ASK: str = os.getenv("RATE_LIMIT_ASK", "30/hour;200/day")
RATE_LIMIT_SESSION_START: str = os.getenv("RATE_LIMIT_SESSION_START", "10/minute")
RATE_LIMIT_ADMIN_LOGIN: str = os.getenv("RATE_LIMIT_ADMIN_LOGIN", "5/minute;30/hour")
RATE_LIMIT_ADMIN_UPLOAD: str = os.getenv("RATE_LIMIT_ADMIN_UPLOAD", "20/hour")
RATE_LIMIT_TRANSLATE: str = os.getenv("RATE_LIMIT_TRANSLATE", "60/hour;500/day")
# Support-desk assist (routers/support.py). Keyed by the support USER (the
# X-Support-User header), not by IP: every agent arrives through the support
# backend's single egress IP, so a per-IP bucket would be shared by the whole
# desk. This is a circuit breaker; the desk applies its own per-user limit first.
RATE_LIMIT_SUPPORT_ASSIST: str = os.getenv("RATE_LIMIT_SUPPORT_ASSIST", "600/hour;5000/day")
# Help-manual excerpts handed to the model per answer, the output cap (a support
# reply is short), and the org id usage is recorded under in the ledger.
SUPPORT_ASSIST_TOP_K: int = int(os.getenv("SUPPORT_ASSIST_TOP_K", "6"))
SUPPORT_ASSIST_MAX_TOKENS: int = int(os.getenv("SUPPORT_ASSIST_MAX_TOKENS", "1200"))
SUPPORT_ASSIST_ORG_ID: str = os.getenv("SUPPORT_ASSIST_ORG_ID", "support")

# Per-session hard cap (cheaper than per-IP — applies even if same IP rotates tokens)
MAX_QUESTIONS_PER_SESSION_DAY: int = int(
    os.getenv("MAX_QUESTIONS_PER_SESSION_DAY", "100")
)

# --- Per-organization AI usage metering + monthly credit cap ---
# Bedrock is pay-per-token with no free-tier ceiling, so each org gets a monthly
# "AI credit" budget. Credits are a normalised, model-weighted unit decoupled
# from raw $ (so price changes don't touch balances and credits stay sellable):
#   1 credit ≈ $0.001 (one milli-dollar). Each call costs
#       ceil(input_tokens/1000 * IN_rate + output_tokens/1000 * OUT_rate)
#   where the per-1k rates below mirror Bedrock's input/output prices per tier.
# Metering writes an append-only ledger row + bumps the org's period counter;
# enforcement gates the 3 user-facing features (procedure/findings/chat) at 402,
# degrades the TB Tier-3 tail to Tier-2, and skips the cheap helpers when over.
AI_USAGE_METERING_ENABLED: bool = os.getenv("AI_USAGE_METERING_ENABLED", "true").lower() == "true"
# Master enforcement switch. false = still meter (write the ledger) but never
# block — useful to observe real usage before turning the cap on.
AI_CREDIT_CAP_ENABLED: bool = os.getenv("AI_CREDIT_CAP_ENABLED", "true").lower() == "true"
# Default monthly allowance for an org with no explicit ai_org_quota row.
# 50000 credits ≈ $50/month. Tune per deployment; per-org overrides live in DB.
AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT: int = int(
    os.getenv("AI_MONTHLY_CREDIT_ALLOWANCE_DEFAULT", "50000")
)
# Block once remaining credits fall to/below this floor (0 = allow until empty).
AI_CREDIT_FLOOR: int = int(os.getenv("AI_CREDIT_FLOOR", "0"))
# Short TTL (seconds) for the cached per-org remaining-credit read, so a burst of
# requests doesn't hit the DB for every pre-flight check. Kept small so a newly
# exhausted org is blocked within the window.
AI_CREDIT_BALANCE_CACHE_TTL_SEC: int = int(os.getenv("AI_CREDIT_BALANCE_CACHE_TTL_SEC", "20"))
# Per-tier credit rates (credits per 1,000 tokens), mirroring Bedrock $/1k.
AI_CREDITS_SMART_IN_PER_1K: float = float(os.getenv("AI_CREDITS_SMART_IN_PER_1K", "3.0"))
AI_CREDITS_SMART_OUT_PER_1K: float = float(os.getenv("AI_CREDITS_SMART_OUT_PER_1K", "15.0"))
AI_CREDITS_FAST_IN_PER_1K: float = float(os.getenv("AI_CREDITS_FAST_IN_PER_1K", "0.8"))
AI_CREDITS_FAST_OUT_PER_1K: float = float(os.getenv("AI_CREDITS_FAST_OUT_PER_1K", "4.0"))
AI_CREDIT_TIER_RATES: dict[str, tuple[float, float]] = {
    "smart": (AI_CREDITS_SMART_IN_PER_1K, AI_CREDITS_SMART_OUT_PER_1K),
    "fast": (AI_CREDITS_FAST_IN_PER_1K, AI_CREDITS_FAST_OUT_PER_1K),
}

# --- Supervised AI agent runtime (file review, engagement build-out, ...) ---
# Master on/off for the whole agent feature; the /agent/* routes return 403 when
# false, so the runtime ships dark and is enabled per environment.
AGENT_FEATURE_ENABLED: bool = os.getenv("AGENT_FEATURE_ENABLED", "false").lower() == "true"


# Pilot allowlist: only these orgs may start agent runs (mirror the FE widget
# gate, which today hard-codes orgs 11,32). Comma-separated; compared as STRINGS
# because the grant/session organization_id is stringified. Empty => no org
# allowed (so turning the master flag on alone does not expose it firm-wide).
def _csv_set(raw: str) -> set[str]:
    return {s.strip() for s in (raw or "").split(",") if s.strip()}


AGENT_ALLOWED_ORG_IDS: set[str] = _csv_set(os.getenv("AGENT_ALLOWED_ORG_IDS", ""))
# Guardrails — hard caps so a run can never loop unbounded or overspend.
AGENT_STEP_LIMIT: int = int(os.getenv("AGENT_STEP_LIMIT", "12"))
AGENT_RUN_TIMEOUT_SEC: int = int(os.getenv("AGENT_RUN_TIMEOUT_SEC", "120"))
AGENT_MAX_RUN_CREDITS: int = int(os.getenv("AGENT_MAX_RUN_CREDITS", "4000"))

# --- Document reader (per-document AI summary / extraction + the document_extraction agent) ---
# Master switch for /copilot/document/* (the on-click "AI summary" in All documents).
# The be FeatureAccess gate at grant mint (feature "agent") is the real per-org gate;
# this only lets an environment ship dark.
DOC_READER_ENABLED: bool = os.getenv("DOC_READER_ENABLED", "true").lower() == "true"
# Vision-first reading: scanned pages / images are sent to Claude as image blocks
# (multilingual, handles Arabic + handwriting far better than Tesseract). false =
# text-only (Tesseract OCR fallback from parser.py for scans).
DOC_READER_VISION_ENABLED: bool = os.getenv("DOC_READER_VISION_ENABLED", "true").lower() == "true"
# Hard caps so one document can never blow the token budget: pages rendered for
# vision, longest image edge (px — Claude's sweet spot is ~1568), native text chars
# sent to the model, bytes downloaded, and the text kept in the insight store for
# follow-up Q&A / future response grounding.
DOC_READER_MAX_PAGES: int = int(os.getenv("DOC_READER_MAX_PAGES", "20"))
DOC_READER_MAX_IMAGE_EDGE: int = int(os.getenv("DOC_READER_MAX_IMAGE_EDGE", "1568"))
DOC_READER_MAX_TEXT_CHARS: int = int(os.getenv("DOC_READER_MAX_TEXT_CHARS", "60000"))
DOC_READER_MAX_FILE_MB: int = int(os.getenv("DOC_READER_MAX_FILE_MB", "40"))
DOC_READER_TEXT_STORE_CHARS: int = int(os.getenv("DOC_READER_TEXT_STORE_CHARS", "200000"))
# A page whose native text layer is shorter than this is treated as scanned and
# rendered for vision instead (covers "image-only PDF with a few header glyphs").
DOC_READER_MIN_TEXT_CHARS_PER_PAGE: int = int(os.getenv("DOC_READER_MIN_TEXT_CHARS_PER_PAGE", "40"))
# (connect, read) timeout for downloading the presigned S3 URL.
DOC_READER_DOWNLOAD_TIMEOUT_SEC: int = int(os.getenv("DOC_READER_DOWNLOAD_TIMEOUT_SEC", "90"))
# The batch agent (document_extraction) reads at most this many documents per run,
# and stops early once it has spent this much wall-clock so the run never exceeds
# AGENT_RUN_TIMEOUT_SEC mid-step.
DOC_READER_MAX_BATCH_DOCS: int = int(os.getenv("DOC_READER_MAX_BATCH_DOCS", "8"))
DOC_READER_BATCH_TIME_BUDGET_SEC: int = int(os.getenv("DOC_READER_BATCH_TIME_BUDGET_SEC", "100"))
# Output-token cap for the structured insight. Output tokens cost 5x input, so
# the prompt keeps summaries tight and line items capped; the cap below fits a
# normal reading, and a reading that still truncates is retried ONCE with the
# larger cap (never a degraded result — just the rare long one costs more).
DOC_READER_MAX_OUTPUT_TOKENS: int = int(os.getenv("DOC_READER_MAX_OUTPUT_TOKENS", "3500"))
DOC_READER_MAX_OUTPUT_TOKENS_RETRY: int = int(os.getenv("DOC_READER_MAX_OUTPUT_TOKENS_RETRY", "6000"))
# Long scanned PDFs: read the first pages AND the last N pages (totals,
# signatures, schedules live at the end) instead of only the first MAX_PAGES.
DOC_READER_TAIL_PAGES: int = int(os.getenv("DOC_READER_TAIL_PAGES", "4"))
# Skip near-blank scanned pages (separator sheets, empty backs) — each page image
# costs ~2,300 tokens whatever it shows.
DOC_READER_SKIP_BLANK_PAGES: bool = os.getenv("DOC_READER_SKIP_BLANK_PAGES", "true").lower() == "true"
# OPT-IN cost lever: route short text-only documents (no page images, at most
# this many extracted characters) to the "fast" model tier (~1/3 of the price).
# 0 = off (default): every document is read by the smart tier.
DOC_READER_FAST_TIER_MAX_CHARS: int = int(os.getenv("DOC_READER_FAST_TIER_MAX_CHARS", "0"))
# Follow-up Q&A over one document: how much of the stored text is handed to the model.
DOC_READER_ASK_CONTEXT_CHARS: int = int(os.getenv("DOC_READER_ASK_CONTEXT_CHARS", "40000"))
# DEV ONLY — path of a local file the reader uses INSTEAD of downloading the
# presigned URL, so the end-to-end UI flow can be exercised on a machine that
# cannot reach the firm's S3/CloudFront. Never set in dev/beta/prod.
DOC_READER_DEV_STATIC_FILE: str = os.getenv("DOC_READER_DEV_STATIC_FILE", "") or ""

# --- Security: file encryption at rest ---
# 32 random URL-safe base64 chars. Generated with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FILE_ENCRYPTION_KEY: str = os.getenv("FILE_ENCRYPTION_KEY", "")
ENCRYPT_UPLOADS: bool = os.getenv("ENCRYPT_UPLOADS", "true").lower() == "true"

# --- Security: max question / answer payload sizes (defence-in-depth) ---
MAX_QUESTION_CHARS: int = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
