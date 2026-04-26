# Stage - builder
FROM python:3.13-slim AS builder
LABEL authors="Anushka Weerasooriya <Nushbox@gmail.com>"
WORKDIR /app
COPY requirements.txt .

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Clean up
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Stage - prod
FROM python:3.13-slim AS prod
LABEL authors="Anushka Weerasooriya <Nushbox@gmail.com>"
WORKDIR /app

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user with home directory
RUN groupadd -r hyper && useradd --no-log-init -r -g hyper -m hyper

# Copy app code
COPY --chown=hyper:hyper main.py .
COPY --chown=hyper:hyper config/ config/
COPY --chown=hyper:hyper datasets/ datasets/

USER hyper

# Set PATH for the user
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
CMD ["/opt/venv/bin/gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]