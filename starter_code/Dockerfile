# syntax=docker/dockerfile:1

# =========================================================
# Stage 1: builder
# Installs dependencies and the project into a virtual
# environment using uv, with caching for fast rebuilds.
# =========================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Compile Python bytecode for faster cold starts, and copy
# (rather than symlink) files from the uv cache into the venv
# so the venv is portable across the multi-stage copy below.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# ---- Build toolchain (builder stage only) ----
# nemoguardrails depends on annoy, a C++ extension with no
# prebuilt wheel for some platforms (e.g. linux/arm64). It must
# be compiled from source, which requires g++/build-essential.
# This toolchain is intentionally NOT installed in the runtime
# stage below, so it never bloats the final production image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ---- Install dependencies first (cached layer) ----
# Only pyproject.toml + uv.lock are needed for this step, so we
# copy just those. As long as they don't change, this layer is
# reused on every rebuild, even if application code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ---- Now copy the rest of the source code ----
COPY . /app

# ---- Install the project itself ----
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# =========================================================
# Stage 2: runtime
# Minimal image containing only the venv + app source,
# running as a non-root user with a proper writable home dir.
# No compiler/build tools here.
# =========================================================
FROM python:3.13-slim AS runtime

# Create a dedicated, unprivileged user/group to run the app.
# --create-home ensures libraries that expect a writable $HOME
# (e.g. cache/config dirs used by some ML/guardrails libraries)
# don't fail with permission errors at runtime.
RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

# Copy the fully-built virtual environment and application
# source from the builder stage, owned by the app user so it
# can read/write as needed at runtime.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app /app

# Make the venv's binaries (uvicorn, gunicorn, etc.) available
# on PATH without needing to activate the virtual environment.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/app

# Drop root privileges before running the app.
USER app

EXPOSE 8000

# Run with Gunicorn as the process manager, using Uvicorn's
# async worker class. --preload loads the app once before
# forking workers, so shared state (chains, guardrails config,
# clients) is initialized a single time rather than per-worker.
CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]
