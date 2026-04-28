# ---- builder stage ----
# install build tools and compile all the dependencies into a venv
FROM python:3.13-slim AS builder

# need these to compile some of the packages with native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# create a venv so we can just copy it across to the final image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- runtime stage ----
# start fresh with a clean slim image and just copy in the venv
FROM python:3.13-slim AS runtime

WORKDIR /app

# copy the venv from builder - no build tools end up in the final image
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy app code and config
COPY main.py .
COPY smartphones.json .
COPY config/ ./config/

# run as a non-root user - good security practice
RUN useradd --no-create-home appuser
USER appuser

EXPOSE 8000

# use gunicorn with uvicorn workers for production
# reload=True from development is not used here
CMD ["gunicorn", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--preload", \
     "main:app"]
