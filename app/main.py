# Start from the last coding stage of the previous LLM evals project
import json
import os
import sys

import dotenv
import uvicorn
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
from fastapi import FastAPI, HTTPException
from logging import getLogger


# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize logging
logger = getLogger("app")

# Initialize FastAPI
app = FastAPI(title="Hypersite: Smartphone Assistant")

# QueryRequest model for input to /ask endpoint
class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str

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
    show_progress_bar=False
)

# Redis configuration
REDIS_URL = os.getenv("REDIS_CONNECTION_STRING")

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
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
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
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# Initialize the vector store
product_db = embed_documents("datasets/smartphones.json")


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
        results = product_db.similarity_search(model, k=1)
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


# ---------------------------------
# Initialize tooling for requests
# ---------------------------------
# List of available tools
tools = [smartphone_info_tool]

# Bind the tools to the language model instance
llm_with_tools = llm.bind_tools(tools)

# Fetch prompts from Langfuse
context_lf_prompt = langfuse.get_prompt("context_system_prompt")
review_lf_prompt = langfuse.get_prompt("review_system_prompt")

# Create LangChain prompts from Langfuse prompts
# Extract the first message (system message) and add MessagesPlaceholder for conversation history
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

# Create message trimmer for context chain to limit token usage
trimmer = trim_messages(
    strategy="last",  # Keep the most recent messages
    token_counter=llm,  # Use LLM to count tokens
    max_tokens=500,  # Maximum tokens for conversation history
    start_on="human",  # Start trimmed history with a human message
    end_on=("human", "tool"),  # End on human or tool message
    include_system=True,  # Always include system message
)

# Build chains (trimmer only on context_chain to manage tool call context)
context_chain = context_prompt | trimmer | llm_with_tools
review_chain = review_prompt | llm

# Load NeMo Guardrails configuration
guardrails_config = RailsConfig.from_path("config/")

# Create guardrails instance for input validation only
# We'll use it separately to validate user input before the chain
input_rails = RunnableRails(guardrails_config, input_key="user_input")

# Initialize the Langfuse handler once for the entire conversation
langfuse_handler = CallbackHandler()


# ---------------------------
# Process /ask request
# ---------------------------
def process_ask(request: QueryRequest):
    # Initialize Redis chat history with TTL (1 hour = 3600 seconds)
    redis_history = RedisChatMessageHistory(
        session_id=request.session_id,
        redis_url=REDIS_URL,
        ttl=3600  # Messages expire after 1 hour
    )

    try:
        # Load conversation history from Redis
        conversation = list(redis_history.messages)

        # Create user message
        user_message = HumanMessage(request.user_input)
        # Add to in-memory conversation for this turn
        conversation.append(user_message)

        # Create a parent span for this user query to group all chain invocations
        with langfuse.start_as_current_observation(
            as_type="span",
            name="user-query",
            input=request.user_input
        ) as span:
            # Propagate trace attributes to all child observations
            with propagate_attributes(
                session_id=request.session_id,
                user_id=request.user_id
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
                    # Rail triggered - skip further processing
                    rail_response = validation_result.response[0]["content"]
                    span.update(
                        output=rail_response,
                        metadata={"triggered_input_rail": validation_context.get("triggered_input_rail")},
                    )
                    return rail_response

                # Context chain invocation (with trimmer to limit tokens)
                ai_with_tools = context_chain.invoke(
                    {"user_input": request.user_input, "conversation": conversation},
                    config={
                        "run_name": "context",
                        "callbacks": [langfuse_handler]
                    }
                )

                # Process tool calls and add results to in-memory conversation
                # Pass config with callbacks to ensure tool invocations are traced
                generate_context(
                    ai_with_tools,
                    conversation,
                    config={"callbacks": [langfuse_handler]}
                )

                # Final response chain invocation
                response = review_chain.invoke(
                    {"user_id": request.user_id, "user_input": request.user_input, "conversation": conversation},
                    config={
                        "run_name": "final-response",
                        "callbacks": [langfuse_handler]
                    }
                )

            # Set the output on the parent span
            span.update(output=response.content)

        result = response.content

        # Save ONLY clean messages to Redis (user input and final AI response)
        # Tool calls and intermediate messages are NOT saved
        redis_history.add_message(user_message)
        redis_history.add_message(response)

        return result

    except Exception as e:
        logger.exception(
            "Unexpected error in main loop",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "user_input": request.user_input,
                "operation": "/ask"
            },
        )
        raise HTTPException(500, "An unexpected error occurred") from e


# ----------------------
# /ask endpoint
# ----------------------
@app.post("/ask")
def ask(request: QueryRequest):
    result = process_ask(request)
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
