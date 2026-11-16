# Start from the last coding stage of the previous LLM evals project
# Task 1: the assistant now runs as a FastAPI web service with a POST /ask endpoint
# Task 2: Qdrant, Redis and Langfuse are cloud services (addresses and keys come from .env)
import json
import os
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Load environment variables from .env file
dotenv.load_dotenv()

# session_id and user_id are no longer generated here.
# The main application sends them in each request (see QueryRequest below).

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

# Redis configuration (Redis Cloud connection string, set in .env)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Request Schema
# ---------------------------
class QueryRequest(BaseModel):
    """
    The JSON body of a POST /ask request.
    FastAPI uses this class to check the body: a missing field or a wrong type
    gets an automatic 422 error response, and ask() is never called.
    """
    user_input: str   # the question the user typed in the frontend
    user_id: str      # which user is asking (from the main application)
    session_id: str   # which chat session this belongs to (also the Redis key)


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Connect to the "smartphones" collection in Qdrant Cloud.
    If the collection doesn't exist yet (first run on a new cluster), create it and
    upload the smartphone documents from the JSON file.

    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        QdrantVectorStore | list: The Qdrant vector store, or an empty list if an error occurs.
    """
    collection_name = "smartphones"

    try:
        # Client for the Qdrant Cloud cluster (was QdrantClient("http://localhost:6333"))
        qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],          # the cluster's endpoint URL
            api_key=os.environ["QDRANT_API_KEY"],  # the cluster's API key
        )

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)

        # Usual case: the data is already in the cloud collection, so just connect to it.
        # The JSON file isn't read and nothing is embedded.
        if collection_exists:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error connecting to the vector store: {e}")
        return []

    # First run only: the collection doesn't exist, so build the documents and upload them.
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

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# The vector store is built on the first SmartphoneInfo tool call, then reused.
# It starts as None, meaning "not built yet".
product_db = None


def get_product_db() -> QdrantVectorStore | list:
    """
    Return the vector store, building it first if it hasn't been built.

    :returns
        QdrantVectorStore | list: The vector store, or an empty list if building failed.
    """
    global product_db  # assign to the module-level variable, not a new local one
    # `not product_db` is True for None (never built) and for [] (the last build failed),
    # so a failed build is retried on the next tool call
    if not product_db:
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
        # get_product_db() builds the vector store on the first call
        results = get_product_db().similarity_search(model, k=1)
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
# Chain Setup (runs once, when the server starts)
# ---------------------------
def build_pipeline() -> dict:
    """
    Fetch the prompts from Langfuse and build the chains and the guardrails.
    This used to be the top half of main().

    :returns
        dict: The context chain, the review chain, and the input guardrails.
    """
    # SmartphoneInfo is the only tool now
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    # Fetch prompts from Langfuse (goodbye_system_prompt is no longer used)
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

    # Load NeMo Guardrails configuration
    guardrails_config = RailsConfig.from_path("config/")

    return {
        # trimmer only on context_chain to manage tool call context
        "context_chain": context_prompt | trimmer | llm_with_tools,
        "review_chain": review_prompt | llm,
        # input validation only; used separately, before the chains
        "input_rails": RunnableRails(guardrails_config, input_key="user_input"),
    }


# Filled in by lifespan() when the server starts
pipeline = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI runs the code before `yield` once when the server starts,
    and the code after `yield` once when the server stops.
    """
    pipeline.update(build_pipeline())
    yield
    # Send any traces still waiting in the Langfuse client's queue before the process ends
    langfuse.flush()


# The web application. uvicorn finds it by the name "app" (see "main:app" at the bottom).
app = FastAPI(title="Smartphone Assistant", lifespan=lifespan)


# ---------------------------
# API Endpoint
# ---------------------------
# A plain `def` (not `async def`): FastAPI runs it in a separate thread,
# so the synchronous calls inside (guardrails, chains, Redis) don't block other requests.
@app.post("/ask")
def ask(request: QueryRequest) -> dict:
    """
    Answer one user question. This used to be one pass of the while loop in main().

    :param
        request (QueryRequest): The validated JSON body (user_input, user_id, session_id).

    :returns
        dict: {"response": "<the assistant's answer>"}, sent back as JSON.
    """
    try:
        # Chat history for this session. session_id now comes from the request.
        redis_history = RedisChatMessageHistory(
            session_id=request.session_id,
            redis_url=REDIS_URL,
            ttl=3600  # Messages expire after 1 hour
        )

        # Load conversation history from Redis
        conversation = list(redis_history.messages)

        # Create user message
        user_message = HumanMessage(request.user_input)
        # Add to in-memory conversation for this turn
        conversation.append(user_message)

        # A new Langfuse handler for each request, so requests don't share one handler
        langfuse_handler = CallbackHandler()

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

                validation_result = pipeline["input_rails"].rails.generate(
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
                    # Rail triggered - send the refusal back and skip saving to Redis
                    rail_response = validation_result.response[0]["content"]
                    span.update(
                        output=rail_response,
                        metadata={"triggered_input_rail": validation_context.get("triggered_input_rail")},
                    )
                    return {"response": rail_response}

                # Context chain invocation (with trimmer to limit tokens)
                ai_with_tools = pipeline["context_chain"].invoke(
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
                response = pipeline["review_chain"].invoke(
                    {"user_id": request.user_id, "user_input": request.user_input, "conversation": conversation},
                    config={
                        "run_name": "final-response",
                        "callbacks": [langfuse_handler]
                    }
                )

            # Set the output on the parent span
            span.update(output=response.content)

        # Save ONLY clean messages to Redis (user input and final AI response)
        # Tool calls and intermediate messages are NOT saved
        redis_history.add_message(user_message)
        redis_history.add_message(response)

        # FastAPI converts this dict to JSON for the response body
        return {"response": response.content}

    except Exception as e:
        # Was sys.exit(1), which would stop the whole server.
        # Now only this request fails, with HTTP status 500 ("Internal Server Error").
        print(f"An unexpected error occurred while answering: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # "main:app" means: the variable `app` in the file main.py
    # host="0.0.0.0": accept connections on all of this computer's network addresses
    # reload=True: restart the server when a .py file in this folder changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
