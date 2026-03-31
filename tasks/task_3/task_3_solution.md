# AI Engineer Deploy - Setup, Build and Run Instructions

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Docker](https://docs.docker.com/get-docker/) (for containerized deployment)
- Cloud accounts for:
  - [Qdrant Cloud](https://cloud.qdrant.io/)
  - [Redis Cloud](https://redis.io/cloud/)
  - [Langfuse Cloud](https://cloud.langfuse.com/)

## 1. Environment Setup

Copy the environment template and fill in your cloud credentials:

```bash
cp .env.template .env
```

Edit `.env` with your values:

```
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_API_KEY=<your-openai-api-key>
QDRANT_CONNECTION_URL=<your-qdrant-cloud-url>
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION_NAME=smartphones
LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
LANGFUSE_HOST=https://us.cloud.langfuse.com
REDIS_CONNECTION_STRING=redis://<username>:<password>@<public_endpoint>:<port>/<database>
REDIS_TTL_S=300
```

## 2. Install Dependencies

```bash
uv sync
```

## 3. Running the Application (Local)

### CLI Mode (`ai-deploy`)

Interactive console-based smartphone assistant with a conversation loop:

```bash
uv run ai-deploy
```

### API Mode (`ask-api`)

REST API served via FastAPI + Uvicorn:

```bash
uv run ask-api
```

The server starts at `http://0.0.0.0:8000`. Send POST requests to the `/ask` endpoint:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Tell me about the iPhone 15",
    "user_id": "test-user",
    "session_id": "session-001"
  }'
```

## 4. Build and Run with Docker

### Build the Docker image

```bash
docker build -t ask-api .
```

### Run the container

Pass your environment file and expose port 8000:

```bash
docker run --env-file .env -p 8000:8000 ask-api
```

The container runs the API via Gunicorn with Uvicorn workers on port `8000`. You can then query the `/ask` endpoint the same way as in local API mode:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Tell me about the iPhone 15",
    "user_id": "test-user",
    "session_id": "session-001"
  }'
```

## Evidence

![img.png](img.png)

![img_1.png](img_1.png)

![img_2.png](img_2.png)