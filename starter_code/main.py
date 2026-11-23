import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import dotenv
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "smartphones.json"
CONFIG_PATH = BASE_DIR / "config"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)
langfuse = get_client()
guardrails_config = RailsConfig.from_path(str(CONFIG_PATH))
input_rails = RunnableRails(guardrails_config, input_key="user_input")
product_db: QdrantVectorStore | None = None
product_db_lock = Lock()


class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


class QueryResponse(BaseModel):
    response: str


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: Literal["Yes", "No"]
    comment: str


class FeedbackResponse(BaseModel):
    message: str


@observe(name="embed_documents")
def embed_documents(json_path: Path) -> QdrantVectorStore:
    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

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

    collection_name = "smartphones"
    qdrant_client = QdrantClient(url="http://localhost:6333")
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        vector_store.add_documents(documents=documents)
        return vector_store

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        collection_name=collection_name,
    )


def get_product_db() -> QdrantVectorStore:
    global product_db
    if product_db is None:
        with product_db_lock:
            if product_db is None:
                product_db = embed_documents(DATASET_PATH)
    return product_db


@tool("SmartphonesInfo")
def smartphones_info_tool(model: str) -> str:
    """Retrieve specifications, price, and availability for a smartphone model."""
    results = get_product_db().similarity_search(model, k=1)
    if not results:
        return "Could not find information for the specified model."
    return results[0].page_content


@observe(name="generate_context")
def generate_context(
    ai_message: AIMessage,
    conversation: list[BaseMessage],
    config: dict[str, Any] | None = None,
) -> None:
    conversation.append(ai_message)
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] != "SmartphonesInfo":
            raise ValueError(f"Unexpected tool call: {tool_call['name']}")
        conversation.append(smartphones_info_tool.invoke(tool_call, config=config))


@lru_cache(maxsize=1)
def get_prompts_and_trimmer():
    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    review_lf_prompt = langfuse.get_prompt("review_system_prompt")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_prompt = ChatPromptTemplate.from_messages(
        [
            review_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    trimmer = trim_messages(
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=500,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    return context_prompt, review_prompt, trimmer


def get_chains(user_id: str, session_id: str):
    context_prompt, review_prompt, trimmer = get_prompts_and_trimmer()
    request_llm = llm.bind(
        user=user_id,
        extra_body={"metadata": {"session_id": session_id}},
    )
    llm_with_tools = llm.bind_tools(
        [smartphones_info_tool],
        user=user_id,
        extra_body={"metadata": {"session_id": session_id}},
    )
    return context_prompt | trimmer | llm_with_tools, review_prompt | request_llm


app = FastAPI()


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    context_chain, review_chain = get_chains(request.user_id, request.session_id)
    langfuse_handler = CallbackHandler()
    redis_history = RedisChatMessageHistory(
        session_id=request.session_id,
        redis_url=REDIS_URL,
        ttl=3600,
    )
    conversation = list(redis_history.messages)
    user_message = HumanMessage(request.user_input)
    conversation.append(user_message)

    with langfuse.start_as_current_observation(
        as_type="span",
        name="user-query",
        input=request.user_input,
    ) as span:
        with propagate_attributes(
            session_id=request.session_id,
            user_id=request.user_id,
        ):
            validation_result = input_rails.rails.generate(
                messages=[{"role": "user", "content": request.user_input}],
                options=GenerationOptions(
                    rails=["input"],
                    output_vars=["allowed", "triggered_input_rail", "bot_message"],
                ),
            )
            validation_context = validation_result.output_data or {}
            rail_triggered = validation_context.get("allowed") is False or bool(
                validation_context.get("triggered_input_rail")
            )
            if rail_triggered:
                rail_response = validation_result.response[0]["content"]
                span.update(
                    output=rail_response,
                    metadata={
                        "triggered_input_rail": validation_context.get(
                            "triggered_input_rail"
                        )
                    },
                )
                return QueryResponse(response=rail_response)

            ai_with_tools = context_chain.invoke(
                {"user_input": request.user_input, "conversation": conversation},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                },
            )
            generate_context(
                ai_with_tools,
                conversation,
                config={"callbacks": [langfuse_handler]},
            )
            response = review_chain.invoke(
                {
                    "user_id": request.user_id,
                    "user_input": request.user_input,
                    "conversation": conversation,
                },
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                },
            )
        if not isinstance(response.content, str):
            raise TypeError("The LLM response content must be a string.")
        span.update(output=response.content)

    redis_history.add_message(user_message)
    redis_history.add_message(response)
    return QueryResponse(response=response.content)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    langfuse.create_score(
        session_id=request.session_id,
        name="conversation_usefulness",
        value=request.feedback,
        data_type="CATEGORICAL",
        comment=request.comment,
    )
    return FeedbackResponse(message="Feedback recorded")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
