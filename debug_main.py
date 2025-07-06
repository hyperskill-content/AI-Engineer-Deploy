import json
import os
import sys
import uuid
from datetime import datetime, timedelta
import time
import logging

from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

import dotenv

# Add debug prints
print("Loading environment variables...")
dotenv.load_dotenv()

# Check environment variables
print(f"LITELLM_MODEL: {os.getenv('LITELLM_MODEL')}")
print(f"LITELLM_BASE_URL: {os.getenv('LITELLM_BASE_URL')}")
print(f"REDIS_URL: {os.getenv('REDIS_URL')}")


# Pydantic Request Schema
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


# Global Variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
total_user_budget = 0.001000
product_db = None

print("Starting imports...")

try:
    from langchain_community.docstore.document import Document
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.tools import tool
    from langchain_core.messages import trim_messages

    print("LangChain imports successful")
except ImportError as e:
    print(f"LangChain import error: {e}")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    print("OpenAI imports successful")
except ImportError as e:
    print(f"OpenAI import error: {e}")
    sys.exit(1)

try:
    from langchain_qdrant import QdrantVectorStore

    print("Qdrant imports successful")
except ImportError as e:
    print(f"Qdrant import error: {e}")
    sys.exit(1)

try:
    from langchain_redis import RedisChatMessageHistory

    print("Redis imports successful")
except ImportError as e:
    print(f"Redis import error: {e}")
    sys.exit(1)

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    from langfuse.decorators import observe, langfuse_context

    print("Langfuse imports successful")
except ImportError as e:
    print(f"Langfuse import error: {e}")
    sys.exit(1)

try:
    from nemoguardrails import RailsConfig
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

    print("NemoGuardRails imports successful")
except ImportError as e:
    print(f"NemoGuardRails import error: {e}")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    print("Qdrant client imports successful")
except ImportError as e:
    print(f"Qdrant client import error: {e}")
    sys.exit(1)

print("All imports successful!")

# Logging setup
logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("langfuse").setLevel(logging.WARNING)

# Test basic functionality
print("Testing basic components...")

# Test environment variables
if not os.getenv("LITELLM_MODEL"):
    print("WARNING: LITELLM_MODEL not set")
if not os.getenv("LITELLM_BASE_URL"):
    print("WARNING: LITELLM_BASE_URL not set")
if not os.getenv("LITELLM_API_KEY"):
    print("WARNING: LITELLM_API_KEY not set")

# Test file access
try:
    with open("smartphones.json", "r") as f:
        data = json.load(f)
    print(f"Successfully loaded smartphones.json with {len(data)} entries")
except Exception as e:
    print(f"Error loading smartphones.json: {e}")

# Test config access
try:
    config = RailsConfig.from_path("./config")
    print("Successfully loaded guardrails config")
except Exception as e:
    print(f"Error loading guardrails config: {e}")


# Simplified lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    global product_db
    print("Starting lifespan initialization...")
    try:
        # Skip vector store initialization for now
        product_db = None
        print("Lifespan initialization complete (vector store skipped)")
    except Exception as e:
        print(f"Error during lifespan initialization: {e}")
        product_db = None

    yield
    print("App shutdown")


# Create FastAPI app
print("Creating FastAPI app...")
app = FastAPI(lifespan=lifespan)


# Simple endpoint for testing
@app.post("/ask")
async def ask(query: QueryRequest, request: Request):
    print(f"Received request: {query}")
    return {"response": f"Hello {query.user_id}! You asked: {query.user_input}"}


@app.get("/health")
async def health():
    return {"status": "ok"}


print("FastAPI app created successfully!")

# Dev Entry Point
if __name__ == "__main__":
    print("Starting server...")
    import uvicorn

    uvicorn.run("debug_main:app", host="localhost", port=8000, reload=True)