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
LOCAL_EMBEDDING_MODEL: str = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# --- Retrieval ---
TOP_K_CHUNKS: int = 8
SIMILARITY_THRESHOLD: float = 0.30  # below this we consider "no answer found"

# --- Allowed file extensions ---
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}
