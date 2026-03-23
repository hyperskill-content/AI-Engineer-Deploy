#FROM ubuntu:latest
#LABEL authors="Philipp"
#
#ENTRYPOINT ["top", "-b"]

# Stage 1: Builder
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv

WORKDIR /app

# Systempakete nur für den Build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Virtuelle Umgebung anlegen
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Anforderungen zuerst kopieren für besseres Layer-Caching
COPY requirements.txt .

# Python-Abhängigkeiten ohne Cache installieren
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Nur falls Runtime-Libs nötig sind; sonst kann dieser Block kleiner ausfallen
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Nicht als root laufen
RUN useradd --create-home --shell /bin/bash appuser

# Virtuelle Umgebung aus Builder übernehmen
COPY --from=builder /opt/venv /opt/venv

# Anwendung kopieren
COPY . .

# Eigentümer setzen
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]