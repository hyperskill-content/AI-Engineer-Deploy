import json
import os

import dotenv
import uvicorn
from fastapi import FastAPI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import logging

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

dotenv.load_dotenv()

# redis cloud connection string
REDIS_URL = os.getenv("REDIS_CONN_STRING", "redis://localhost:6380/0")

# route through litellm proxy
llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    show_progress_bar=True,
)

langfuse_handler = CallbackHandler()
langfuse_client = get_client()

# set up guardrails as a separate input checker so it doesn't break the chain
rails_config = RailsConfig.from_path("./config")
input_checker = RunnableRails(
    rails_config,
    runnable=RunnableLambda(lambda x: {"output": "allowed"}),
    input_key="user_input",
)

app = FastAPI()

# lazy init - only build the vector store the first time the tool is called
product_db = None

blocked_message = "I'm sorry, I can't respond to that."


# schema for incoming requests
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


def embed_documents(json_path: str):
    """load smartphones into qdrant cloud - skips creating the collection if it already exists"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
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
        # connect to qdrant cloud using url and api key from env
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if collection_exists:
            # collection already in cloud so just connect to it
            qdrant_store = QdrantVectorStore.from_existing_collection(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                embedding=embeddings_model,
                collection_name=collection_name,
            )
            return qdrant_store
        else:
            # first run - create the collection and upload the data
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            qdrant_store = QdrantVectorStore(
                client=qdrant_client, collection_name=collection_name, embedding=embeddings_model
            )
            qdrant_store.add_documents(documents=documents)
            return qdrant_store

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return []


@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """Retrieves information about a smartphone model from the product database."""
    # build the vector store on first use
    global product_db
    if product_db is None:
        product_db = embed_documents("smartphones.json")
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            return "Could not find information for the specified model."
        return results[0].page_content
    except Exception as e:
        return f"Error retrieving smartphone info: {e}"


def generate_context(llm_response):
    for tool_call in llm_response.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            return smartphone_info_tool.invoke(tool_call).content
    return ""


@app.post("/ask")
def ask(request: QueryRequest):
    # check input with guardrails first before doing anything else
    guard_result = input_checker.invoke({"user_input": request.user_input})
    if isinstance(guard_result, dict) and guard_result.get("output") == blocked_message:
        return {"response": blocked_message}

    tools = [smartphone_info_tool]
    llm_with_tools = llm.bind_tools(tools)

    # pass user_id through to litellm for budget tracking
    llm_with_user = llm_with_tools.bind(user=request.user_id)

    def get_redis_history(session_id: str) -> BaseChatMessageHistory:
        # connect to redis cloud using connection string from env
        return RedisChatMessageHistory(session_id, url=REDIS_URL, ttl=3600)

    trimmer = trim_messages(
        strategy="last",
        token_counter=len,
        max_tokens=10,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    langfuse_context_prompt = langfuse_client.get_prompt("context_system_prompt", label="production")
    langchain_context_prompt = ChatPromptTemplate.from_messages([
        langfuse_context_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="chat_history"),
        langfuse_context_prompt.get_langchain_prompt()[1],
    ])
    langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}

    context_chain = langchain_context_prompt | trimmer | llm_with_user | generate_context
    context_chain_with_history = RunnableWithMessageHistory(
        context_chain,
        get_redis_history,
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )

    langfuse_review_prompt = langfuse_client.get_prompt("review_system_prompt")
    langchain_review_prompt = ChatPromptTemplate.from_messages([
        langfuse_review_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="chat_history"),
        langfuse_review_prompt.get_langchain_prompt()[1],
    ])
    langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}

    review_chain = langchain_review_prompt | llm
    review_chain_with_history = RunnableWithMessageHistory(
        review_chain,
        get_redis_history,
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )

    context = context_chain_with_history.invoke(
        {"user_input": request.user_input},
        config={
            "configurable": {"session_id": request.session_id},
            "callbacks": [langfuse_handler],
            "run_name": "context",
        },
    )

    final_response = review_chain_with_history.invoke(
        {"user_input": request.user_input, "user_id": request.user_id, "context": context},
        config={
            "configurable": {"session_id": request.session_id},
            "callbacks": [langfuse_handler],
            "run_name": "final_response",
        },
    )

    return {"response": final_response.content}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
