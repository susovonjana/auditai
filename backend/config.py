"""
Central configuration loader.
Reads environment variables from .env and exposes them as constants.

FREE STACK:
  - LLM:        Google Gemini API (gemini-flash-latest, with model failover)
  - Embeddings: sentence-transformers locally (bge-small-en-v1.5, 384-dim)
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

# --- AI provider keys ---
# Only Gemini is required in the free stack
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Legacy paid providers — only used if you switch back manually
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

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
ONEAUDIT_BASE_URL: str = os.getenv(
    "ONEAUDIT_BASE_URL", "http://localhost:5030/api/v1/internal"
)
# Timeout (seconds) for copilot data callbacks to 1audit-be. Some read services
# (e.g. a lead-sheet-heavy working paper's content) take ~20s, so allow headroom.
ONEAUDIT_HTTP_TIMEOUT: int = int(os.getenv("ONEAUDIT_HTTP_TIMEOUT", "45"))

# Short-TTL cache for the file-mode chat's data fetches. Within this window a
# repeated tool call (same file + endpoint + args) is served from memory instead
# of re-querying 1audit-be, so a burst of questions doesn't re-fetch the same
# trial balance / summary each time. The grant is still re-validated once per
# chat request, so caching never bypasses authorization. Set to 0 to disable.
COPILOT_DATA_CACHE_TTL_SEC: int = int(os.getenv("COPILOT_DATA_CACHE_TTL_SEC", "120"))

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

# --- Models ---
# Gemini (LLM). Comma-separated list of model names for failover when one
# model hits its per-day free-tier quota. Tried in order; left-most first.
# A single value (no comma) keeps the original single-model behaviour.
# NOTE: gemini-2.0-flash was retired by Google (2026-06-01). Default to the
# resilient `gemini-flash-latest` alias (always points at the current flash
# model) with versioned fallbacks. Override via the GEMINI_MODEL env var.
GEMINI_MODEL_RAW: str = os.getenv(
    "GEMINI_MODEL", "gemini-flash-latest,gemini-2.5-flash,gemini-2.5-flash-lite"
)
GEMINI_MODELS: list[str] = [m.strip() for m in GEMINI_MODEL_RAW.split(",") if m.strip()] or [
    "gemini-flash-latest"
]
# Kept for callers that still import the original constant; points at the
# primary (left-most) model.
GEMINI_MODEL: str = GEMINI_MODELS[0]
# How long (seconds) to skip a model after it returns a quota error, so
# subsequent requests jump straight to the next model in the chain.
GEMINI_MODEL_COOLDOWN_SEC: int = int(os.getenv("GEMINI_MODEL_COOLDOWN_SEC", "600"))

# Local embeddings (sentence-transformers)
# bge-small-en-v1.5 is 384-dim and substantially more accurate than MiniLM.
LOCAL_EMBEDDING_MODEL: str = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# Cross-encoder reranker (~80 MB, loaded once at first request)
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

# Per-session hard cap (cheaper than per-IP — applies even if same IP rotates tokens)
MAX_QUESTIONS_PER_SESSION_DAY: int = int(
    os.getenv("MAX_QUESTIONS_PER_SESSION_DAY", "100")
)

# --- Security: file encryption at rest ---
# 32 random URL-safe base64 chars. Generated with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
FILE_ENCRYPTION_KEY: str = os.getenv("FILE_ENCRYPTION_KEY", "")
ENCRYPT_UPLOADS: bool = os.getenv("ENCRYPT_UPLOADS", "true").lower() == "true"

# --- Security: max question / answer payload sizes (defence-in-depth) ---
MAX_QUESTION_CHARS: int = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
