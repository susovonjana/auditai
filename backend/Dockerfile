# syntax=docker/dockerfile:1.7
#
# AuditAI backend — production image for ECS (EC2 or Fargate launch type).
#
# Build:
#   docker build -t auditai-api:latest .
#
# Run locally:
#   docker run --rm -p 8000:8000 --env-file .env auditai-api:latest
#
# Image targets x86_64. For Graviton (ARM) EC2 instances, build with:
#   docker buildx build --platform linux/arm64 -t auditai-api:arm64 .
# -----------------------------------------------------------------------------

# =============================================================================
# Stage 1 — install Python dependencies
#   Build toolchain lives only in this stage, never in the final image.
#   torch is pulled from PyTorch's CPU index (~200 MB) instead of the default
#   PyPI wheel (~700 MB with CUDA libs we'd never use on ECS without GPU).
# =============================================================================
FROM python:3.11-slim-bookworm AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
# --ignore-installed forces pip to write the FULL dependency closure into
# /install. Without it, packages already satisfied by the base image's
# system Python (e.g. `packaging`, pulled in by pip itself) are skipped,
# so the final runtime stage misses them and imports fail at startup.
RUN pip install --upgrade pip wheel \
    && pip install --prefix=/install --ignore-installed \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt \
    # `pip install --upgrade pip wheel` above pulls `packaging` into the build
    # base's /usr/local, so the --prefix install treats it as already satisfied
    # and never writes it to /install. Later stages only copy /install, so
    # without this packaging goes missing → ModuleNotFoundError at model bake.
    && pip install --prefix=/install --ignore-installed packaging


# =============================================================================
# Stage 2 — bake the sentence-transformers models into the image
#   On first call, sentence-transformers downloads ~213 MB of model weights.
#   Doing it here means every container starts with a warm cache and skips
#   the network hop, so ECS task startup stays fast.
#
#   Override which models are baked at build time:
#     docker build --build-arg EMBEDDING_MODEL=all-MiniLM-L6-v2 .
# =============================================================================
FROM python:3.11-slim-bookworm AS models

COPY --from=deps /install /usr/local

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')"


# =============================================================================
# Stage 3 — runtime image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Runtime OS deps:
#   poppler-utils        → pdf2image (rasterise PDFs for OCR fallback)
#   tesseract-ocr + eng  → pytesseract OCR engine + English language data
#   libgl1 / libglib2.0  → opencv-python shared libs
#   libsm6 / libxext6 / libxrender1 → opencv X11 stubs (headless-safe)
#   libpq5               → libpq runtime (asyncpg uses its own, but pandas
#                          + some helpers may dlopen it; cheap to include)
#   curl                 → ALB / ECS HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpq5 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bring across installed Python packages and pre-downloaded HF model cache.
COPY --from=deps   /install /usr/local
COPY --from=models /models  /models

# Non-root runtime user. Most ECS / compliance scans flag root containers.
ARG APP_UID=1000
RUN useradd --create-home --uid ${APP_UID} --shell /bin/bash auditai

WORKDIR /app

# Application code. .dockerignore strips .env, .venv, uploads/, .git, caches.
COPY --chown=auditai:auditai . /app

# Writable scratch for any transient file work. The durable store is S3 once
# you wire up the storage abstraction; locally and during dev this is enough.
RUN mkdir -p /tmp/uploads \
    && chown -R auditai:auditai /tmp/uploads /models

USER auditai

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    UPLOAD_DIR=/tmp/uploads \
    PORT=8000

EXPOSE 8000

# start-period gives the ALB time for model warm-up before it starts failing
# health checks and marking the task unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail --silent http://localhost:${PORT}/ || exit 1

# Single uvicorn worker per container on purpose:
#   - Each worker holds its own copy of the sentence-transformers + reranker
#     model (~250 MB resident). Two workers in a 4 GB task wastes ~250 MB
#     and risks OOM during concurrent PDF parsing.
#   - FastAPI is async; one worker handles many concurrent requests via the
#     event loop. Scale horizontally with more ECS tasks.
#   - --proxy-headers + --forwarded-allow-ips so request.client / scheme
#     reflect the real client behind the ALB.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--no-server-header"]
