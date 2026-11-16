# Production image for the Smartphone Assistant API.
# Build from the starter_code folder:  docker build -t hypersite:latest .

# ---------------------------
# Stage 1: builder (has uv; installs the dependencies into /app/.venv)
# ---------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# UV_COMPILE_BYTECODE=1: compile .py files to bytecode now, so the app starts faster
# UV_LINK_MODE=copy: copy package files into .venv instead of linking them to uv's cache
#   (the cache is a temporary mount, so links to it would break)
# UV_PYTHON_DOWNLOADS=0: use the image's own Python (/usr/local/bin/python3), which
#   the runtime stage also has, instead of letting uv download a separate one
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy only the two files that list the dependencies, then install the dependencies.
# Docker reuses this layer on later builds until pyproject.toml or uv.lock changes,
# so editing main.py doesn't reinstall every package.
# --mount=type=cache keeps uv's download cache between builds, outside the image.
# --frozen: install exactly what uv.lock says, and fail if it's out of date
# --no-install-project: dependencies only; the app's own code comes next
# --no-dev: skip the "dev" dependency group
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the app's code (.dockerignore keeps out .env, .venv, caches and editor folders)
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------
# Stage 2: runtime (no uv, no build cache; only Python, the code and .venv)
# ---------------------------
# Same Python version and Debian release (bookworm) as the builder,
# so the .venv built above works here unchanged
FROM python:3.13-slim-bookworm

# Create a non-root user and group, both named "app", with a home folder
RUN groupadd --system app && \
    useradd --system --gid app --create-home app

WORKDIR /app

# Copy the code and .venv from the builder stage, owned by the app user
COPY --from=builder --chown=app:app /app /app

# PATH: put .venv's programs (gunicorn, python) first, so they're found by name
# PYTHONUNBUFFERED=1: send print() output straight to `docker logs` instead of holding it
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# From here on, everything runs as the app user, not root
USER app

# Documents the port the app listens on (publishing it is done with -p in docker run)
EXPOSE 8000

# Gunicorn manages 2 worker processes; each runs the FastAPI app with Uvicorn's worker class.
# --preload: import main.py once before starting the workers
CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]
