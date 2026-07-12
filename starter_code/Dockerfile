# ==============================================================================
# Stage 1: Build the virtual environment and install dependencies
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory inside the builder container
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Use link mode 'copy' to ensure uv copy files instead of hardlinking (needed in Docker environments)
ENV UV_LINK_MODE=copy

# Install system build dependencies (e.g., g++) required for compilation of C++ extensions (like annoy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first using cache mounts for faster subsequent builds
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application files
COPY . /app

# Synchronize the project (ensuring everything is compiled and placed in .venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ==============================================================================
# Stage 2: Create a minimal production runtime image
# ==============================================================================
FROM python:3.13-slim

# Create a system group and user to run the app securely (non-root)
RUN groupadd -r appgroup && useradd -r -m -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy the built virtual environment and application code from the builder stage
# while giving ownership to our non-root user
COPY --from=builder --chown=appuser:appgroup /app /app

# Prepend the virtual environment's bin folder to the system PATH so python
# uses our installed packages automatically
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Expose port 8000 for FastAPI
EXPOSE 8000

# Switch to the non-root user
USER appuser

# Launch the app using Gunicorn with Uvicorn workers in production
CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app" \
]


