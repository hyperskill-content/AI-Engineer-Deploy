# syntax=docker/dockerfile:1.7

# ---------- Stage 1: Builder ----------
FROM python:3.13-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libssl-dev \
        gcc && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download fastembed model so it's baked into the image
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')" || \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" || true

# ---------- Stage 2: Runtime ----------
FROM python:3.13-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libffi8 \
        libssl3 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && \
    useradd  --system --gid app --home /app --shell /usr/sbin/nologin app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy fastembed cache from builder (so model is pre-downloaded)
COPY --from=builder --chown=app:app /tmp/fastembed_cache /tmp/fastembed_cache
COPY --from=builder --chown=app:app /root/.cache/huggingface /app/.cache/huggingface

WORKDIR /app

# Create cache dirs owned by app user BEFORE switching user
RUN mkdir -p /app/.cache /tmp/fastembed_cache && \
    chown -R app:app /app /tmp/fastembed_cache

# Copy app code
COPY --chown=app:app . .

# Environment for HF/fastembed caching
ENV HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/tmp/fastembed_cache \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
