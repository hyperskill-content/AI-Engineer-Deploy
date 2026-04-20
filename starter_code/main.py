import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)


# Load environment variables from .env file
dotenv.load_dotenv()
session_name = f"session-{uuid.uuid4().hex[:8]}"
user_id = "HyperUser"
total_user_budget = 0.0010000
REDIS_URL = os.getenv("REDIS_CONN_STRING")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)

langfuse_handler = CallbackHandler()
langfuse_client = get_client()

config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config, input_key="user_input")

def check_budget(current_usage: float) -> bool:
    if current_usage < total_user_budget:
        return True
    else:
        return False


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
def embed_documents(json_path: str):
    """
    Load JSON data from the specified file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Chroma: A Chroma vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []

    documents = []
    for entry in data:
        # Build a readable content string from the JSON entry
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
        qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            qdrant_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings_model
            )

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                url=os.environ.get("QDRANT_URL"),
                api_key=os.environ.get("QDRANT_API_KEY"),
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieve information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: A summary of the smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    try:
        results = app_state["product_db"].similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        print(f"Error during smartphone information retrieval for model {model}: {e}")
        return f"Error during smartphone information retrieval: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
def generate_context(llm_tools):
    """
    Process tool calls from the language model and collect their responses.

    :param
        llm_with_tools: The language model instance with bound tools.

    :returns
        Toolresponse
    """

    # Process each tool call based on its name
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call).content
            return tool_response
    return ""


# ---------------------------
# FastAPI App
# ---------------------------
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


app_state: dict = {}


def build_chains():
    tools = [smartphone_info_tool]
    llm_with_tools = llm.bind_tools(tools)

    def get_redis_history(session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=3600)

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=1000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    langfuse_context_prompt = langfuse_client.get_prompt("context_system_prompt", label="production")
    langchain_context_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_context_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_context_prompt.get_langchain_prompt()[1],
        ]
    )
    langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}

    context_chain = langchain_context_prompt | trimmer | llm_with_tools | generate_context
    context_chain_with_history = RunnableWithMessageHistory(
        context_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )
    context_chain_with_history_and_rails = guardrails | context_chain_with_history

    langfuse_review_prompt = langfuse_client.get_prompt("review_system_prompt")
    langchain_review_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_review_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_review_prompt.get_langchain_prompt()[1],
        ]
    )
    langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}

    review_chain = langchain_review_prompt | llm
    review_chain_with_history = RunnableWithMessageHistory(
        review_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    return context_chain_with_history_and_rails, review_chain_with_history, get_redis_history


@asynccontextmanager
async def lifespan(application: FastAPI):
    app_state["product_db"] = embed_documents("smartphones.json")
    context_chain, review_chain, _ = build_chains()
    app_state["context_chain"] = context_chain
    app_state["review_chain"] = review_chain
    yield


app = FastAPI(lifespan=lifespan)


def get_user_cost(user_id: str) -> float:
    total = 0.0
    traces_response = langfuse_client.api.trace.list(user_id=user_id)
    traces = traces_response.data if hasattr(traces_response, "data") else traces_response
    for trace in traces:
        current_trace = langfuse_client.api.trace.get(trace.id)
        cost = current_trace.total_cost if hasattr(current_trace, "total_cost") and current_trace.total_cost else 0.0
        total += cost
    return total


@app.post("/ask")
def ask(request: QueryRequest):
    current_cost = get_user_cost(request.user_id)
    if not check_budget(current_cost):
        return {"error": "Budget exceeded. Please try again later.", "usage": current_cost}

    context_chain = app_state["context_chain"]
    review_chain = app_state["review_chain"]

    with propagate_attributes(session_id=request.session_id, user_id=request.user_id):
        context = context_chain.invoke(
            {"user_input": request.user_input},
            config={
                "configurable": {"session_id": request.session_id},
                "callbacks": [langfuse_handler],
                "run_name": "context",
            },
        )

        context_result = context.get("output") if isinstance(context, dict) else context
        if context_result and context_result.strip().lower() == "i'm sorry, i can't respond to that.":
            return {"response": context_result, "usage": current_cost}


        final_response = review_chain.invoke(
            {"user_input": request.user_input, "user_id": request.user_id, "context": context_result},
            config={
                "configurable": {"session_id": request.session_id},
                "callbacks": [langfuse_handler],
                "run_name": "final_response",
            },
        )

    # The cost update does not work reliably with the cloud instance of Langfuse (to slow to update traces).
    # Also, there are two traces created for each call, one for the context and one for the final response.
    latest_traces = langfuse_client.api.trace.list(limit=1)
    if hasattr(latest_traces, "data") and len(latest_traces.data) > 0:
        trace = langfuse_client.api.trace.get(latest_traces.data[0].id)
        trace_cost = trace.total_cost if hasattr(trace, "total_cost") and trace.total_cost else 0.0
        current_cost += trace_cost

    return {"response": final_response.content, "usage": current_cost}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)