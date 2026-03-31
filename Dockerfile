# ---- Builder stage ----
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml poetry.lock uv.lock README.md ./
COPY src/ src/

RUN uv sync --no-dev --no-cache

# ---- Runtime stage ----
FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY config/ config/
COPY dataset/ dataset/

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "ai_engineer_deploy.service.service:api"]
