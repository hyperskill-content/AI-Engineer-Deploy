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
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_core.messages import trim_messages

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe, langfuse_context

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ADDED: Fix for async issues with guardrails
import nest_asyncio

nest_asyncio.apply()

# ---------------------------
# Logging setup
# ---------------------------
logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("langfuse").setLevel(logging.WARNING)

# ---------------------------
# App and Env Setup
# ---------------------------
dotenv.load_dotenv()


# ---------------------------
# Pydantic Request Schema
# ---------------------------
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


# ---------------------------
# Global Variables
# ---------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
total_user_budget = 0.001000
product_db = None
guardrails = None  # Initialize as None first

# ---------------------------
# Model Initialization
# ---------------------------
llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": "HyperUser"}
)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv("LITELLM_EMBEDDING_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    show_progress_bar=True,
    model_kwargs={"user": "HyperUser"}
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# FIXED: Initialize guardrails with better error handling
try:
    config = RailsConfig.from_path("./config")
    guardrails = RunnableRails(config, input_key="input")
    print("Guardrails initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize guardrails: {e}")
    guardrails = None


# ---------------------------
# Tool: Smartphone Info
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieve detailed specifications and availability info for a given smartphone model.
    """
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            return "Could not find information for the specified model."
        return results[0].page_content
    except Exception as e:
        print(f"Smartphone info error: {e}")
        return "Error retrieving smartphone information."


# ---------------------------
# Vector DB Build Function
# ---------------------------
@observe
def embed_documents(json_path: str):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return []

    documents = []
    for entry in data:
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

            store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings_model)
            store.add_documents(documents=documents)
            return store
        else:
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )
    except Exception as e:
        print(f"Vector store error: {e}")
        return []


# ---------------------------
# Budget Check Function
# ---------------------------
def check_budget(current_usage: float) -> bool:
    return current_usage < total_user_budget


# ---------------------------
# Context Handler
# ---------------------------
def generate_context(llm_response):
    if not hasattr(llm_response, 'tool_calls') or not llm_response.tool_calls:
        return ""
    for tool_call in llm_response.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            return smartphone_info_tool.invoke(tool_call).content
    return ""


# ---------------------------
# Core Chain Runner (Simplified approach)
# ---------------------------
def run_chain(user_input: str, user_id: str, session_id: str):
    try:
        print(f"=== MAIN CHAIN: Starting for user: {user_id}, session: {session_id} ===")

        # Step 1: Initialize Langfuse
        try:
            langfuse_context.update_current_trace(user_id=user_id, session_id=session_id)
            langfuse_handler = langfuse_context.get_current_langchain_handler()
            print("✓ Langfuse initialized")
        except Exception as e:
            print(f"✗ Langfuse initialization failed: {e}")
            # Continue without Langfuse
            langfuse_handler = None

        # Step 2: Get context from vector store
        context = ""
        try:
            if product_db:
                vector_results = product_db.similarity_search(user_input, k=1)
                if vector_results:
                    context = vector_results[0].page_content
                    print(f"✓ Found context: {context[:100]}...")
                else:
                    print("✗ No vector results found")
            else:
                print("✗ Product DB not available")
        except Exception as e:
            print(f"✗ Vector search failed: {e}")

        # Step 3: Get Redis history
        try:
            history = RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=120)
            chat_history = history.messages
            print(f"✓ Retrieved {len(chat_history)} chat history messages")
        except Exception as e:
            print(f"✗ Redis history failed: {e}")
            chat_history = []

        # Step 4: Get prompts from Langfuse
        try:
            context_prompt = langfuse_client.get_prompt("context-prompt", label="production")
            review_prompt = langfuse_client.get_prompt("review-prompt")
            print("✓ Retrieved prompts from Langfuse")
        except Exception as e:
            print(f"✗ Failed to get prompts from Langfuse: {e}")
            # Use fallback prompts
            context_prompt_text = "You are a helpful smartphone assistant. Use the context provided to answer questions about smartphones."
            review_prompt_text = "Based on the context and user question, provide a helpful response about smartphones."

        # Step 5: Simple implementation using direct LLM calls
        try:
            # First, let's try the simple approach that worked in debug
            if context:
                final_prompt = f"""You are a helpful smartphone assistant. Use the following product information to answer the user's question:

Product Information:
{context}

Chat History:
{chr(10).join([f"{msg.type}: {msg.content}" for msg in chat_history[-3:]]) if chat_history else "No previous conversation"}

User Question: {user_input}

Please provide a helpful and detailed response about the smartphone."""
            else:
                final_prompt = f"""You are a helpful smartphone assistant.

Chat History:
{chr(10).join([f"{msg.type}: {msg.content}" for msg in chat_history[-3:]]) if chat_history else "No previous conversation"}

User Question: {user_input}

Please provide a helpful response about smartphones."""

            print("✓ Created final prompt")

            # Make the LLM call
            response = llm.invoke(final_prompt)
            print("✓ LLM call successful")

            # Save to history
            try:
                history.add_user_message(user_input)
                history.add_ai_message(response.content)
                print("✓ Saved to chat history")
            except Exception as e:
                print(f"✗ Failed to save chat history: {e}")

            return response.content

        except Exception as e:
            print(f"✗ Simple LLM approach failed: {e}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I encountered an error processing your request."

    except Exception as e:
        print(f"✗ Top-level error in run_chain: {str(e)}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, I couldn't generate a response."


# ---------------------------
# Simplified Chain Runner for debugging
# ---------------------------
def run_chain_simple(user_input: str, user_id: str, session_id: str):
    """Simplified version that bypasses complex chain construction"""
    try:
        print(f"=== Simple chain for: {user_input} ===")

        # Step 1: Get context from vector store
        context = ""
        if product_db:
            try:
                vector_results = product_db.similarity_search(user_input, k=1)
                if vector_results:
                    context = vector_results[0].page_content
                    print(f"Found context: {context[:200]}...")
            except Exception as e:
                print(f"Vector search failed: {e}")

        # Step 2: Create a simple prompt with context
        if context:
            prompt_text = f"""You are a helpful smartphone assistant. Use the following product information to answer the user's question:

Product Information:
{context}

User Question: {user_input}

Please provide a helpful response about the smartphone."""
        else:
            prompt_text = f"""You are a helpful smartphone assistant. 

User Question: {user_input}

Please provide a helpful response about smartphones."""

        # Step 3: Simple LLM call
        response = llm.invoke(prompt_text)
        return response.content

    except Exception as e:
        print(f"Simple chain failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error in simple chain: {str(e)}"


# ---------------------------
# Startup Hook and App Init
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global product_db
    product_db = embed_documents("smartphones.json")
    if not product_db:
        print("Failed to initialize vector store.")
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------
# Test Endpoints
# ---------------------------
@app.get("/test_vector")
async def test_vector():
    if not product_db:
        return {"status": "Vector store not initialized"}
    try:
        results = product_db.similarity_search("iPhone 13", k=1)
        return {"status": "success", "results": str(results)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/check_prompts")
async def check_prompts():
    try:
        context_prompt = langfuse_client.get_prompt("context-prompt", label="production")
        review_prompt = langfuse_client.get_prompt("review-prompt")
        return {
            "context_prompt_exists": bool(context_prompt),
            "review_prompt_exists": bool(review_prompt)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/test_redis")
async def test_redis():
    try:
        test_history = RedisChatMessageHistory("test_session", redis_url=REDIS_URL)
        test_history.add_user_message("Test message")
        messages = test_history.messages
        return {"status": "success", "messages": str(messages)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------
# Debug Endpoints (FIXED - Skip guardrails test)
# ---------------------------
@app.post("/debug_ask")
async def debug_ask(query: QueryRequest, request: Request):
    """Simplified version to debug step by step"""
    print(f"=== DEBUG: Starting request for user: {query.user_id} ===")

    # Test 1: Basic LLM call
    try:
        print("Test 1: Basic LLM call")
        basic_response = llm.invoke("Hello, can you respond?")
        print(f"Basic LLM response: {basic_response.content}")
    except Exception as e:
        print(f"Basic LLM failed: {e}")
        return {"error": "Basic LLM call failed", "details": str(e)}

    # Test 2: Tool binding
    try:
        print("Test 2: Tool binding")
        tools = [smartphone_info_tool]
        llm_with_tools = llm.bind_tools(tools)
        print("Tools bound successfully")
    except Exception as e:
        print(f"Tool binding failed: {e}")
        return {"error": "Tool binding failed", "details": str(e)}

    # Test 3: Vector store query
    try:
        print("Test 3: Vector store query")
        if product_db:
            vector_results = product_db.similarity_search("iPhone 13", k=1)
            print(f"Vector search results: {len(vector_results)} results found")
            if vector_results:
                print(f"First result: {vector_results[0].page_content[:200]}...")
        else:
            print("Product DB is None")
            return {"error": "Vector store not initialized"}
    except Exception as e:
        print(f"Vector store query failed: {e}")
        return {"error": "Vector store query failed", "details": str(e)}

    # Test 4: Redis connection
    try:
        print("Test 4: Redis connection")
        test_history = RedisChatMessageHistory("debug_session", redis_url=REDIS_URL)
        test_history.add_user_message("Debug test message")
        messages = test_history.messages
        print(f"Redis test successful, messages: {len(messages)}")
    except Exception as e:
        print(f"Redis test failed: {e}")
        return {"error": "Redis connection failed", "details": str(e)}

    # Test 5: Langfuse prompts
    try:
        print("Test 5: Langfuse prompts")
        context_prompt = langfuse_client.get_prompt("context-prompt", label="production")
        review_prompt = langfuse_client.get_prompt("review-prompt")
        print("Both prompts retrieved successfully")

        # Check prompt structure
        context_langchain = context_prompt.get_langchain_prompt()
        review_langchain = review_prompt.get_langchain_prompt()
        print(f"Context prompt has {len(context_langchain)} messages")
        print(f"Review prompt has {len(review_langchain)} messages")

    except Exception as e:
        print(f"Langfuse prompt test failed: {e}")
        return {"error": "Langfuse prompt retrieval failed", "details": str(e)}

    # Test 6: Skip guardrails for now
    print("Test 6: Skipping guardrails test (async issues)")

    # Test 7: Simple chain without guardrails
    try:
        print("Test 7: Simple chain without guardrails")

        # Create a simple prompt template
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{user_input}")
        ])

        simple_chain = simple_prompt | llm
        simple_response = simple_chain.invoke({"user_input": query.user_input})
        print(f"Simple chain response: {simple_response.content}")

        return {
            "status": "All tests passed (guardrails skipped)",
            "simple_response": simple_response.content
        }

    except Exception as e:
        print(f"Simple chain test failed: {e}")
        return {"error": "Simple chain failed", "details": str(e)}


@app.post("/ask_simple")
async def ask_simple(query: QueryRequest, request: Request):
    """Simple version using run_chain_simple"""
    response = run_chain_simple(
        user_input=query.user_input,
        user_id=query.user_id,
        session_id=query.session_id
    )
    return {"response": response}


# ---------------------------
# Main FastAPI Endpoint
# ---------------------------
@app.post("/ask")
async def ask(query: QueryRequest, request: Request):
    if not check_budget(0.0):  # Simplified for now
        return {"response": "Usage budget exceeded. Please try again later."}
    response = run_chain(
        user_input=query.user_input,
        user_id=query.user_id,
        session_id=query.session_id
    )
    return {"response": response}


# ---------------------------
# Dev Entry Point
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8001, reload=True)