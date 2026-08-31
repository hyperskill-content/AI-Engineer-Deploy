## Production-ready Docker Image

## **Table of Contents**

- [Description](#description)
- [Recommended Development Steps](#recommended-development-steps)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
    - [Docs](#docs)

### Description

We are now preparing the application for the final stages of deployment to AWS EC2 instances. Since we are using the instance solely for this application, we may not need a Docker image. However, Docker is the industry-standard and highly recommended practice. With Docker, you ensure that your application can consistently run in any environment. This avoids the age-old issue, “it works on my machine”.

### **Recommended Development Steps**

In this task, let's work on the Dockerfile to build the image. The image must be production-ready and exclude unnecessary files. Start by creating a `.dockerignore` file to exclude `.git`, `__pycache__`, `.env`, `.venv`, and other development artifacts. This keeps the build context small and prevents copying unnecessary files into the image.

Use a uv-provided base image such as `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`. These images include uv pre-installed and are optimized for production use. Alternatively, copy uv from the official distroless image:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
```

To keep the image size small, follow these practices:

1. **Use intermediate layers for caching** — install dependencies separately from the project. Dependencies only rebuild when the lockfile changes:

    ```dockerfile
    # Install dependencies first (cached layer)
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --frozen --no-install-project --no-dev

    # Copy source code
    COPY . /app

    # Install the project
    RUN --mount=type=cache,target=/root/.cache/uv \
        uv sync --frozen --no-dev
    ```

2. **Use multi-stage builds** — build dependencies in one stage, then copy only the virtual environment to a minimal runtime stage:

    ```dockerfile
    FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
    # ... install dependencies and project ...

    FROM python:3.13-slim
    COPY --from=builder /app/.venv /app/.venv
    COPY --from=builder /app /app
    ```

3. **Set production environment variables**:

    ```dockerfile
    ENV UV_COMPILE_BYTECODE=1 \
        UV_LINK_MODE=copy
    ```

   These compile Python bytecode for faster startup and ensure proper file copying in containers.


Follow security best practices by running the application as a non-root user. Create a dedicated user in the Dockerfile to run the app. Pass environment variables at runtime (ensure your code accesses variables via `os.environ`):

```bash
docker run --env-file .env -p 8000:8000 custom-image:latest
```

For production, run the web server with Gunicorn as the process manager instead of running uvicorn directly. Set your image's entry point as follows:

```bash
CMD ["gunicorn",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:8000",
     "--workers", "2",        
     "--preload",      
     "main:app"
]
```

This starts Gunicorn using Uvicorn's async worker class with 2 workers, binding the FastAPI app (`main:app`) to port 8000 and preloading the application code before forking worker processes. Build the image:

```bash
docker build -t hypersite:latest .
```

Run the image to verify it works correctly. The image should be around ~600-700MB.

### **Deliverables**

You should have a `Dockerfile` that builds a production-ready image for your application. Verify that you're ignoring unnecessary files, optimizing image size, following security best practices, and using the correct production `CMD`. Test the image in your local environment.

### **Useful Resources**

### **Docs**
- [uv Docker Integration Guide](https://docs.astral.sh/uv/guides/integration/docker/#available-images)
- [uv Docker Example Repository](https://github.com/astral-sh/uv-docker-example)
- [Docker optimization](https://docs.docker.com/build-cloud/optimization/)
- [Python's starter `.dockerignore` template](https://github.com/themattrix/python-pypi-template/blob/master/.dockerignore)