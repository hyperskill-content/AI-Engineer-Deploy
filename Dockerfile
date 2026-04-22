# Stage - builder
FROM python:3.13-slim AS builder
LABEL authors="Anushka Weerasooriya <Nushbox@gmail.com>"
WORKDIR /app
COPY requirements.txt .

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends build-essential libffi-dev libssl-dev
rm -rf /var/lib/apt/lists/*
pip install --require-hashes --no-cache-dir -r requirements.txt
EOF

# Stage - prod
FROM python:3.13-slim AS prod
LABEL authors="Anushka Weerasooriya <Nushbox@gmail.com>"
WORKDIR /app

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user with home directory
RUN groupadd -r hyper && useradd --no-log-init -r -g hyper -m hyper

# Copy app code
COPY --chown=hyper:hyper main.py .
COPY --chown=hyper:hyper config/ config/
COPY --chown=hyper:hyper datasets/ datasets/

USER hyper

EXPOSE 8000
CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]