import json
import os
import sys
import uuid
from datetime import datetime, timedelta

import dotenv
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
from langchain_core.globals import set_debug
import logging

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)


# Load environment variables from env.env file
dotenv.load_dotenv()
session_name = f"session-{uuid.uuid4().hex[:8]}"
user_id = "HyperUser"
total_user_budget = 0.0010000
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

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

# Initialize the callback handler for Langfuse
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    trace_name="ai-response",
    user_id=user_id,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

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
@observe
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
        qdrant_client = QdrantClient("http://localhost:6333")

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
    except Exception as e:
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
        Toolresponse
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
    print("Unfortunately, you've exceeded your current usage. Please try again later.")
    end_session_tool.invoke({"session_status": "exit"})
    sys.exit(0)


# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="ai-response")
def main():
    langfuse_context.update_current_trace(
        session_id=session_name,
        user_id=user_id
    )
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
        print("Unfortunately, you've exceeded your current usage. Please try again later.")
        end_session_tool.invoke({"session_status": "exit"})

    current_cost = initial_cost

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

        while True:
            if check_budget(current_cost):
                user_input = input("User: ").strip()

                context = context_chain_with_history_and_rails.invoke(
                    {"user_input": user_input},
                    config={
                        "configurable": {"session_id": user_id},
                        "callbacks": [langfuse_handler], "run_name": "context"
                    }
                )

                context_result = context.get("output") if isinstance(context, dict) else context
                if context_result and context_result.strip().lower() == "i'm sorry, i can't respond to that.":
                    print(f"System: {context_result}")
                else:
                    final_response = review_chain_with_history.invoke(
                        {"user_input": user_input, "user_id": user_id, "context": context},
                        config={
                            "configurable": {"session_id": user_id},
                            "callbacks": [langfuse_handler], "run_name": "final_response"
                        }
                    )
                    print(f"System: {final_response.content}")

                trace_id = langfuse_client.fetch_traces().data[0].id
                current_cost += langfuse_client.fetch_trace(trace_id).data.total_cost
                print(f"Your usage so far: {current_cost}")

            else:
                budget_exceeded()
    except KeyboardInterrupt:
        end_session_tool.invoke({"session_status": "exit"})
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Build the product database vector store
    product_db = embed_documents("smartphones.json")
    main()