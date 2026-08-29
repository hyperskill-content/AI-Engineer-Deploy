import json
import os
import re
from difflib import SequenceMatcher

import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from openai import RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions

# Load environment variables from .env file
dotenv.load_dotenv()

# Matches this repo's .env.sample naming convention exactly
REDIS_URL = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:6380")

# ---------------------------
# Pydantic request schema (as specified by the task)
# ---------------------------
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


# ---------------------------
# LLM / Embeddings clients, routed through the OpenAI-compatible
# LiteLLM gateway configured via OPENAI_* env vars (per .env.sample)
# ---------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    tiktoken_model_name="gpt-4o-mini",  # underlying model, for local token counting only
)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)

langfuse = get_client()
langfuse_handler = CallbackHandler()


# ---------------------------
# Vector store: built lazily on first use, not at import time,
# so the web server can start instantly.
# ---------------------------
_product_db = None
_known_models = None


def load_smartphone_data(json_path: str) -> list:
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []


@observe(name="embed_documents")
def embed_documents(data: list):
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
        qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if collection_exists:
            return QdrantVectorStore.from_existing_collection(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                embedding=embeddings_model,
                collection_name=collection_name,
            )

        # First-run fallback: collection doesn't exist yet on this cluster,
        # so create it and populate it once.
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        qdrant_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        qdrant_store.add_documents(documents=documents)
        return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return None



def get_product_db_and_models():
    """Lazily build the vector store + known model list on first tool invocation."""
    global _product_db, _known_models
    if _product_db is None:
        smartphone_data = load_smartphone_data("datasets/smartphones.json")
        _product_db = embed_documents(smartphone_data)
        _known_models = [entry["model"] for entry in smartphone_data]
    return _product_db, _known_models


# ---------------------------
# Tool: SmartphoneInfo (EndSession tool removed - no longer needed for API use)
# ---------------------------
def token_sort_ratio(a: str, b: str) -> float:
    a_sorted = " ".join(sorted(a.lower().split()))
    b_sorted = " ".join(sorted(b.lower().split()))
    return SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100


def extract_numbers(s: str) -> set:
    return set(re.findall(r"\d+", s))


@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Look up a smartphone model in the catalog. Uses substring matching first,
    then falls back to fuzzy matching with a number-mismatch guard to avoid
    hallucinating specs for models that are not actually in the catalog.
    """
    try:
        product_db, known_models = get_product_db_and_models()
        if not known_models or product_db is None:
            return "Smartphone catalog is currently unavailable."

        lower_query = model.lower()
        substring_matches = [m for m in known_models if lower_query in m.lower()]
        if substring_matches:
            best = min(substring_matches, key=len)
        else:
            scored = sorted(known_models, key=lambda c: -token_sort_ratio(model, c))
            top_match = scored[0]
            q_nums, m_nums = extract_numbers(model), extract_numbers(top_match)
            score = token_sort_ratio(model, top_match)
            if (q_nums and not (q_nums & m_nums)) or score < 60:
                return f"No information found for '{model}' in our catalog."
            best = top_match

        results = product_db.similarity_search(best, k=1)
        return results[0].page_content
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list) -> None:
    conversation.append(ai_message)

    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(
            AIMessage(content="No tool calls found. Please ensure the model is configured to use tools.")
        )
        return

    try:
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                tool_output = smartphone_info_tool.invoke(tool_call)
                conversation.append(tool_output)
    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(AIMessage(content=f"An error occurred while processing tool calls: {e}"))


# ---------------------------
# Build chains and guardrails ONCE at startup (cheap: no network calls,
# just prompt/config loading)
# ---------------------------
tools = [smartphone_info_tool]
llm_with_tools = llm.bind_tools(tools)

context_lf_prompt = langfuse.get_prompt("context_system_prompt")
context_prompt = ChatPromptTemplate.from_messages([
    context_lf_prompt.get_langchain_prompt()[0],
    MessagesPlaceholder(variable_name="conversation"),
])
context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

review_lf_prompt = langfuse.get_prompt("review_system_prompt")
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

context_chain = context_prompt | trimmer | llm_with_tools
review_chain = review_prompt | llm

with open("config/config.yml", "r") as f:
    config_yaml_content = f.read().replace(
        "OPENAI_BASE_URL_PLACEHOLDER", os.getenv("OPENAI_BASE_URL")
    )

with open("config/prompts.yml", "r") as f:
    prompts_yaml_content = f.read()

guardrails_config = RailsConfig.from_content(
    colang_content="",
    yaml_content=config_yaml_content + "\n" + prompts_yaml_content,
)
input_rails = RunnableRails(guardrails_config, input_key="user_input")


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Smartphone Assistant API")


@app.post("/ask")
def ask(request: QueryRequest):
    user_input = request.user_input.strip()
    user_id = request.user_id
    session_id = request.session_id

    # Bind the calling user for LiteLLM per-user budget tracking on this request
    request_llm = llm.bind(user=user_id)
    request_context_chain = context_prompt | trimmer | request_llm.bind_tools(tools)
    request_review_chain = review_prompt | request_llm

    redis_history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=REDIS_URL,
        ttl=3600,
    )

    # ---------------------------
    # Guardrails input validation
    # ---------------------------
    try:
        validation_result = input_rails.rails.generate(
            messages=[{"role": "user", "content": user_input}],
            options=GenerationOptions(
                rails=["input"],
                output_vars=["allowed", "triggered_input_rail", "bot_message"],
            ),
        )
    except RateLimitError as budget_error:
        with langfuse.start_as_current_observation(
            as_type="span", name="user-query-budget-exceeded", input={"user_input": user_input}
        ) as span:
            with propagate_attributes(session_id=session_id, user_id=user_id):
                span.update(output={"response": str(budget_error)}, level="WARNING")
        return JSONResponse(
            status_code=429,
            content={"response": f"Sorry {user_id}, you have reached your usage limit for today. Please try again later."},
        )

    validation_context = validation_result.output_data or {}
    rail_triggered = validation_context.get("allowed") is False or bool(
        validation_context.get("triggered_input_rail")
    )

    if rail_triggered:
        blocked_message = validation_result.response[0]["content"]
        with langfuse.start_as_current_observation(
            as_type="span", name="user-query-blocked", input={"user_input": user_input}
        ) as span:
            with propagate_attributes(session_id=session_id, user_id=user_id):
                span.update(output={"response": blocked_message}, level="WARNING")
        return {"response": blocked_message}

    conversation = list(redis_history.messages)
    user_message = HumanMessage(user_input)
    conversation.append(user_message)

    try:
        with langfuse.start_as_current_observation(
            as_type="span", name="user-query", input={"user_input": user_input}
        ) as span:
            with propagate_attributes(session_id=session_id, user_id=user_id):
                ai_message = request_context_chain.invoke(
                    {"user_input": user_input, "conversation": conversation},
                    config={"run_name": "context", "callbacks": [langfuse_handler]},
                )
                generate_context(ai_message, conversation)

                response = request_review_chain.invoke(
                    {"user_id": user_id, "user_input": user_input, "conversation": conversation},
                    config={"run_name": "final-response", "callbacks": [langfuse_handler]},
                )

            span.update(output={"response": response.content})

    except RateLimitError as budget_error:
        return JSONResponse(
            status_code=429,
            content={"response": f"Sorry {user_id}, you have reached your usage limit for today. Please try again later."},
        )

    redis_history.add_message(user_message)
    redis_history.add_message(response)

    return {"response": response.content}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
