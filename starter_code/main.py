import json
import os
import dotenv
import uvicorn
import nest_asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse.langchain import CallbackHandler
from langfuse import get_client, propagate_attributes
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.messages import trim_messages
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

dotenv.load_dotenv()
nest_asyncio.apply()
total_user_budget = 1.0010000
REDIS_URL = os.getenv("REDIS_CONN_STRING")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

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

app = FastAPI()
product_db = None

@app.on_event("startup")
def startup_event():
    global product_db
    base_dir = os.path.dirname(os.path.abspath(__file__))
    product_db = embed_documents(os.path.join(base_dir, "smartphones.json"))

class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str

def check_budget(current_usage: float) -> bool:
    return current_usage < total_user_budget

def embed_documents(json_path: str):
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
            api_key=os.environ["QDRANT_API_KEY"]
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
        else:
            return QdrantVectorStore.from_existing_collection(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                embedding=embeddings_model,
                collection_name=collection_name,
            )
    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


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
        results = product_db.similarity_search(model, k=1)
        if not results:
            return "Could not find information for the specified model."
        return results[0].page_content
    except Exception as e:
        print(f"Error during smartphone information retrieval for model {model}: {e}")
        return f"Error during smartphone information retrieval: {e}"


def generate_context(llm_tools):
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call).content
            return tool_response
    return ""
@app.post("/ask")
def ask(request: QueryRequest):
    user_input = request.user_input
    user_id = request.user_id
    session_id = request.session_id

    tools = [smartphone_info_tool]
    llm_with_tools = llm.bind_tools(tools)

    def get_redis_history(sid: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(sid, redis_url=REDIS_URL, ttl=3600)

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=1000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    langfuse_context_prompt = langfuse_client.get_prompt("context")
    langchain_context_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_context_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ]
    )
    langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}
    context_chain = langchain_context_prompt | trimmer | llm_with_tools | generate_context
    context_chain_with_history = RunnableWithMessageHistory(
        context_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    langfuse_review_prompt = langfuse_client.get_prompt("review")
    langchain_review_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_review_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User question: {user_input}\n\nContext: {context}")
        ]
    )
    langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}
    review_chain = langchain_review_prompt | llm
    review_chain_with_history = RunnableWithMessageHistory(
        review_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    initial_cost = 0.0
    traces_response = langfuse_client.api.trace.list(user_id=user_id)
    traces = traces_response.data if hasattr(traces_response, 'data') else traces_response
    for trace in traces:
        current_trace = langfuse_client.api.trace.get(trace.id)
        cost = current_trace.total_cost if hasattr(current_trace, 'total_cost') and current_trace.total_cost else 0.0
        initial_cost += cost

    if not check_budget(initial_cost):
        return {"response": "Unfortunately, you've exceeded your current usage. Please try again later."}

    with propagate_attributes(session_id=session_id, user_id=user_id):
        guardrails_result = guardrails.invoke(
            {"user_input": user_input},
            config={"callbacks": [langfuse_handler]}
        )
        guardrails_output = guardrails_result.get("output") if isinstance(guardrails_result,
                                                                          dict) else guardrails_result
        print(f"DEBUG guardrails_output: {guardrails_output}")

        if guardrails_output and guardrails_output.strip().lower() == "i'm sorry, i can't respond to that.":
            return {"response": guardrails_output}

        context = context_chain_with_history.invoke(
            {"user_input": user_input},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [langfuse_handler],
                "run_name": "context"
            }
        )
        print(f"DEBUG context: {context}")
        context_result = context.get("output") if isinstance(context, dict) else context

        final_response = review_chain_with_history.invoke(
            {"user_input": user_input, "user_id": user_id, "context": context_result},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [langfuse_handler],
                "run_name": "final_response"
            }
        )

    return {"response": final_response.content}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)