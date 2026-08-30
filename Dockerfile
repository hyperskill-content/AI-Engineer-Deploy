# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# 'annoy' package requires g++
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/llib/apt/lists/*

WORKDIR /app

# Copy manifest files first so layer is cached
COPY ./app/pyproject.toml ./app/uv.lock ./

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY ./app/ .

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.13-slim AS runtime

RUN groupadd --system appuser && useradd --system --gid appuser --home /app appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app .

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

CMD ["gunicorn", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "2", \
    "--preload", \
    "main:app"]