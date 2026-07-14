# Create a uv-provided base image
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install build dependencies for packages like annoy that require C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml uv.lock ./

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY . .

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]