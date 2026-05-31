# syntax=docker/dockerfile:1.7

# ============================================================
# Stage 1 — Builder: install dependencies with uv (cached)
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# g++ is required to compile native deps that ship no wheel for the build platform
# (e.g. `annoy`, pulled in transitively by nemoguardrails).
# These tools stay in the builder image only — the runtime stage copies the
# already-compiled .venv and never carries the toolchain.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer — invalidated only when pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Project layer — source-only changes don't rebuild deps
COPY config ./config
COPY datasets ./datasets
COPY main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ============================================================
# Stage 2 — Runtime: minimal slim image, non-root user
# ============================================================
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
