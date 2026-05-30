"""
Central configuration loader.
Reads environment variables from .env and exposes them as constants.

FREE STACK:
  - LLM:        Google Gemini API (gemini-1.5-flash, free tier)
  - Embeddings: sentence-transformers locally (all-MiniLM-L6-v2, 384-dim)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this backend folder
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

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

# --- Uploads ---
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Models ---
# Gemini (LLM)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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
CHUNK_OVERLAP: int = 250  # bigger overlap reduces "number split from label" cases

# --- Retrieval ---
# Hybrid: pull more candidates from each retriever, then rerank down to TOP_K.
INITIAL_CANDIDATES: int = 20
TOP_K_CHUNKS: int = 8
SIMILARITY_THRESHOLD: float = 0.30

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
