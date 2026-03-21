FROM python:3.13-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null; \
    find /opt/venv -type f -name "*.pyo" -delete 2>/dev/null; \
    true

FROM python:3.13-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home appuser
WORKDIR /home/appuser/app

COPY main.py .
COPY smartphones.json .
COPY config/ config/

RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 8000

CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]
