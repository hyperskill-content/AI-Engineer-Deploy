#!/usr/bin/env -S docker build -t hypersite:latest .
# run container with: docker run --name hypersite --env-file .env -p 8000:8000 hypersite:latest

# Stage 1: Building the virtual environment
FROM python:3.13-slim AS compiler
LABEL author=ComfyChair

WORKDIR /app

COPY starter_code/requirements.txt .
COPY starter_code/config ./config
COPY starter_code/main.py .
COPY starter_code/smartphones.json .


RUN apt-get update && \
  apt-get -y install build-essential libffi-dev libssl-dev gcc && \
  python3 -m venv ./venv && \
 ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Creating the final image
FROM python:3.13-slim AS runner
LABEL author=ComfyChair

WORKDIR /app

COPY --from=compiler /app .
ENV PATH="/app/venv/bin:/usr/bin:$PATH"

EXPOSE 8000

RUN useradd -m app
USER app

CMD ["gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--preload", "main:app"]