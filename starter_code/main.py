import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

dotenv.load_dotenv()

REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDINGS_MODEL",
    "OPENAI_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "REDIS_URL",
    "QDRANT_URL",
)

_missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if _missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(_missing)
        + ". Check your .env file (locally) or --env-file/-e flags (in Docker)."
    )

REDIS_URL = os.environ["REDIS_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # optional — only needed for Qdrant Cloud
DATASET_PATH = os.getenv("DATASET_PATH", "datasets/smartphones.json")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "smartphones")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=False,
)

langfuse = get_client()


# ---------------------------
# Lazy vector store (built on first tool call, per Task 1 spec)
# ---------------------------
_product_db: Optional[QdrantVectorStore] = None


@observe(name="embed_documents")
def embed_documents(json_path: str) -> Optional[QdrantVectorStore]:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return None
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return None

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

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings_model,
        )
        store.add_documents(documents=documents)
        return store

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def get_product_db() -> QdrantVectorStore:
    global _product_db
    if _product_db is None:
        _product_db = embed_documents(DATASET_PATH)
        if _product_db is None:
            raise RuntimeError("Failed to initialize product vector store")
    return _product_db


# ---------------------------
# Tool
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.
    """
    try:
        results = get_product_db().similarity_search(model, k=1)
        if not results:
            return "Could not find information for the specified model."
        return results[0].page_content
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool call handling
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list, config: dict = None) -> None:
    conversation.append(ai_message)
    if not getattr(ai_message, "tool_calls", None):
        conversation.append(AIMessage(content="No tool calls found."))
        return

    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_output = smartphone_info_tool.invoke(tool_call, config=config)
            conversation.append(tool_output)


# ---------------------------
# Pipeline (built once at startup, shared across requests)
# ---------------------------
def build_pipeline() -> dict:
    llm_with_tools = llm.bind_tools([smartphone_info_tool])

    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    review_lf_prompt = langfuse.get_prompt("review_system_prompt")

    context_prompt = ChatPromptTemplate.from_messages([
        context_lf_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="conversation"),
    ])
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_prompt = ChatPromptTemplate.from_messages([
        review_lf_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="conversation"),
    ])
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=500,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    return {
        "context_chain": context_prompt | trimmer | llm_with_tools,
        "review_chain": review_prompt | llm,
        "input_rails": RunnableRails(RailsConfig.from_path("config/"), input_key="user_input"),
        "langfuse_handler": CallbackHandler(),
    }


pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.update(build_pipeline())
    yield
    pipeline.clear()


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Smartphone Assistant API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)


class QueryResponse(BaseModel):
    response: str
    rail_triggered: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest) -> QueryResponse:
    context_chain = pipeline["context_chain"]
    review_chain = pipeline["review_chain"]
    input_rails = pipeline["input_rails"]
    langfuse_handler = pipeline["langfuse_handler"]

    redis_history = RedisChatMessageHistory(
        session_id=req.session_id,
        redis_url=REDIS_URL,
        ttl=3600,
    )

    conversation = list(redis_history.messages)
    user_message = HumanMessage(req.user_input)
    conversation.append(user_message)

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="user-query",
            input={"user_input": req.user_input},
        ) as span:
            with propagate_attributes(session_id=req.session_id, user_id=req.user_id):
                validation_result = input_rails.invoke(
                    {"user_input": req.user_input},
                    config={"run_name": "input-validation", "callbacks": [langfuse_handler]},
                )
                rail_triggered = (
                    isinstance(validation_result, AIMessage)
                    and validation_result.response_metadata.get("rails_triggered", False)
                )
                if rail_triggered:
                    span.update(output={"response": validation_result.content, "rail_triggered": True})
                    return QueryResponse(response=validation_result.content, rail_triggered=True)

                ai_with_tools = context_chain.invoke(
                    {"user_input": req.user_input, "conversation": conversation},
                    config={"run_name": "context", "callbacks": [langfuse_handler]},
                )

                generate_context(ai_with_tools, conversation, config={"callbacks": [langfuse_handler]})

                response = review_chain.invoke(
                    {"user_id": req.user_id, "user_input": req.user_input, "conversation": conversation},
                    config={"run_name": "final-response", "callbacks": [langfuse_handler]},
                )

            span.update(output={"response": response.content})
    except Exception as e:
        print(f"Error processing /ask for session={req.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing request")

    redis_history.add_message(user_message)
    redis_history.add_message(response)

    return QueryResponse(response=response.content)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
