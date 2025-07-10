# start from your final stage solution of Part Two
import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe, langfuse_context
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.messages import trim_messages
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import logging

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load environment variables from .env file
load_dotenv()

class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str

app = FastAPI()

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)

# session_name = f"session-{uuid.uuid4().hex[:8]}"
user_id = "HyperUser"
total_user_budget = 0.0010000
REDIS_URL = os.getenv("REDIS_CONN_STRING")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# embeddings_model = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004"
# )
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)

# Initialize the callback handler for Langfuse
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    trace_name="ai-response",
    # user_id=user_id,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config, input_key="user_input")

def check_budget(current_usage: float) -> bool:
    """
    Check if the current usage is less than the total user budget.

    :param
        current_usage (float): The current usage of the user.

    :returns
        bool: True if the current usage is less than the total user budget, False otherwise.
    """
    if current_usage < total_user_budget:
        return True
    else:
        return False


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
@observe
def embed_documents(json_path: str):
    """
    Load smartphone data from JSON and build or connect to a Qdrant vector store.

    :param
        json_path (str): The path to the JSON file containing smartphone data.

    :returns
        QdrantVectorStore: A Qdrant vector store instance.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
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

    collection_name = "smartphones"
    qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

    # Check if collection exists, else create and add documents
    if not qdrant_client.collection_exists(collection_name=collection_name):
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
        # Use existing collection
        qdrant_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings_model,
            collection_name=collection_name,
        )
        return qdrant_store


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
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        print(f"Error during smartphone information retrieval for model {model}: {e}")
        return f"Error during smartphone information retrieval: {e}"


@tool("EndSession")
def end_session_tool(session_status: str):
    """
    Ends the current session and outputs a goodbye message when the user
    expresses gratitude, or it is clear they would like to end the
    current conversation

    :param
        session_status (str): Status message indicating the end of session
        Should always be "exit"

    :returns
        Exits the system after printing the goodbye message.
    """
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    langfuse_goodbye_prompt = langfuse_client.get_prompt("goodbye-prompt")
    langchain_goodbye_prompt = ChatPromptTemplate.from_messages(
        langfuse_goodbye_prompt.get_langchain_prompt(),
    )

    langchain_goodbye_prompt.metadata = {"langfuse_prompt": langfuse_goodbye_prompt}

    try:

        goodbye_chain = langchain_goodbye_prompt | llm
        goodbye_message = goodbye_chain.invoke({"user_id": user_id},
                                               config={"callbacks": [langfuse_handler], "run_name": "goodbye"})

        return goodbye_message
    except Exception:
        return "Thank you for visiting. Goodbye!"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
def generate_context(llm_tools):
    """
    Process tool calls from the language model and collect their responses.

    :param
        llm_with_tools: The language model instance with bound tools.

    :returns
        Tool response
    """
    # Process each tool call based on its name
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call).content
            return tool_response
        elif tool_call["name"] == "EndSession":
            tool_response = end_session_tool.invoke(tool_call).content
            return tool_response
    return ""

def budget_exceeded():
    end_session_tool.invoke({"session_status": "exit"})
    return "Unfortunately, you've exceeded your current usage. Please try again later."


# Define a route that listens for GET requests at the root URL (/):
@app.get("/")
def hello():
    return {"message": "Hello, World!"}

# ---------------------------
# Main Conversation Loop
# ---------------------------
@app.post("/ask")
@observe(name="ai-response")
def main(request: QueryRequest):

    langfuse_context.update_current_trace(
        session_id=request.session_id,
        user_id=request.user_id
    )
    user_input = request.user_input

    langfuse_handler = langfuse_context.get_current_langchain_handler()

    # List of available tools
    tools = [smartphone_info_tool, end_session_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    def get_redis_history(session_id: str) -> BaseChatMessageHistory:
        return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=120)

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=1000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    langfuse_context_prompt = langfuse_client.get_prompt("context-prompt", label="production")
    langchain_context_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_context_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_context_prompt.get_langchain_prompt()[1]
        ]
    )

    langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}

    context_chain = langchain_context_prompt | trimmer | llm_with_tools | generate_context
    context_chain_with_history = RunnableWithMessageHistory(
        context_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    context_chain_with_history_and_rails = guardrails | context_chain_with_history

    langfuse_review_prompt = langfuse_client.get_prompt("review-prompt")
    langchain_review_prompt = ChatPromptTemplate.from_messages(
        [
            langfuse_review_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="chat_history"),
            langfuse_review_prompt.get_langchain_prompt()[1]
        ]
    )

    langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}

    review_chain = langchain_review_prompt | llm
    review_chain_with_history = RunnableWithMessageHistory(
        review_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
    )

    initial_cost = 0.0

    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=1)

    traces = langfuse_client.fetch_traces(name="ai-response", user_id="hyper-user", from_timestamp=start_time,
                                          to_timestamp=end_time).data

    for trace in traces:
        current_trace = langfuse_client.fetch_trace(id=trace.id)
        cost = current_trace.data.total_cost
        initial_cost += cost

    if not check_budget(initial_cost):
        end_session_tool.invoke({"session_status": "exit"})
        return "Unfortunately, you've exceeded your current usage. Please try again later."

    current_cost = initial_cost

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

        while True:
            if check_budget(current_cost) and user_input:

                context = context_chain_with_history_and_rails.invoke(
                    {"user_input": user_input},
                    config={
                        "configurable": {"session_id": user_id},
                        "callbacks": [langfuse_handler], "run_name": "context"
                    }
                )

                user_input = ""

                context_result = context.get("output") if isinstance(context, dict) else context
                if context_result and context_result.strip().lower() == "i'm sorry, i can't respond to that.":
                    # print(f"System: {context_result}")
                    response = f"System: {context_result}"
                else:
                    final_response = review_chain_with_history.invoke(
                        {"user_input": user_input, "user_id": user_id, "context": context},
                        config={
                            "configurable": {"session_id": user_id},
                            "callbacks": [langfuse_handler], "run_name": "final_response"
                        }
                    )

                    user_input = ""

                    # print(f"System: {final_response.content}")
                    response = f"System: {final_response.content}"

                trace_id = langfuse_client.fetch_traces().data[0].id
                current_cost += langfuse_client.fetch_trace(trace_id).data.total_cost
                return response + f"\n\nYour usage so far: {current_cost}"

            else:
                budget_exceeded()
    except KeyboardInterrupt:
        end_session_tool.invoke({"session_status": "exit"})
        return "Exited"
    except Exception as e:
        return f"An unexpected error occurred in the main loop: {e}"


if __name__ == "__main__":
    # Build the product database vector store at module level
    product_db = embed_documents("smartphones.json")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
