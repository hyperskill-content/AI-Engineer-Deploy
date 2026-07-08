# Start from the last coding stage of the previous LLM evals project
import json
import os

import dotenv
import uvicorn
from fastapi import FastAPI
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
from nemoguardrails.rails.llm.options import GenerationOptions
from pydantic import BaseModel

# ---------------------------
# Data Models
# ---------------------------
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        QdrantVectorStore | list: A Qdrant vector store built from the smartphone documents,
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


# Vector store is built lazily on first tool invocation and cached thereafter.
product_db: QdrantVectorStore | list | None = None


def get_product_db() -> QdrantVectorStore | list:
    global product_db
    if product_db is None:
        product_db = embed_documents("datasets/smartphones.json")
    return product_db


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    try:
        db = get_product_db()
        results = db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list, config: dict | None = None) -> None:
    """
    Process tool calls from the language model and append their responses as ToolMessage objects
    to the conversation history in place.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        conversation (list): The current conversation history (in-memory for this turn).
        config (dict | None): Optional configuration dictionary containing callbacks for tracing.

    :returns
        None. The conversation list is updated in place.
    """
    # construct the conversation history with the AI message containing tool calls
    conversation.append(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to the conversation
        # a message with tool calls is expected to be followed by tool responses
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                # Pass config with callbacks to ensure tool invocation is traced
                tool_output = smartphone_info_tool.invoke(tool_call, config=config)
                conversation.append(tool_output)

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )


# ---------------------------
# Chain Setup (built once at startup)
# ---------------------------
tools = [smartphone_info_tool]
llm_with_tools = llm.bind_tools(tools)

# Fetch prompts from Langfuse
context_lf_prompt = langfuse.get_prompt("context_system_prompt")
review_lf_prompt = langfuse.get_prompt("review_system_prompt")

# Extract the first (system) message and add MessagesPlaceholder for conversation history
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

# Trimmer for the context chain to limit token usage
trimmer = trim_messages(
    strategy="last",
    token_counter=llm,
    max_tokens=500,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

# Build chains (trimmer only on context_chain to manage tool call context)
context_chain = context_prompt | trimmer | llm_with_tools
review_chain = review_prompt | llm

# Load NeMo Guardrails configuration and build input-validation runnable
guardrails_config = RailsConfig.from_path("config/")
input_rails = RunnableRails(guardrails_config, input_key="user_input")

# Langfuse callback handler shared across requests
langfuse_handler = CallbackHandler()


# ---------------------------
# FastAPI Application
# ---------------------------
app = FastAPI()


@app.post("/ask")
def ask(request: QueryRequest) -> dict:
    # Per-request Redis chat history keyed by session_id (TTL: 1 hour)
    redis_history = RedisChatMessageHistory(
        session_id=request.session_id,
        redis_url=REDIS_URL,
        ttl=3600,
    )

    # Load prior conversation and append the new user message
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
                    metadata={"triggered_input_rail": validation_context.get("triggered_input_rail")},
                )
                return {"response": rail_response}

            # Context chain invocation (with trimmer to limit tokens)
            ai_with_tools = context_chain.invoke(
                {"user_input": request.user_input, "conversation": conversation},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                },
            )

            # Process tool calls and add results to in-memory conversation
            generate_context(
                ai_with_tools,
                conversation,
                config={"callbacks": [langfuse_handler]},
            )

            # Final response chain invocation
            response = review_chain.invoke(
                {"user_id": request.user_id, "user_input": request.user_input, "conversation": conversation},
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                },
            )

        span.update(output=response.content)

    # Persist only clean messages (user input and final AI response) to Redis
    redis_history.add_message(user_message)
    redis_history.add_message(response)

    return {"response": response.content}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
