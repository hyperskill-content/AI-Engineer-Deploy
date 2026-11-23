# Start from the last coding stage of the previous LLM evals project
import json
import os
import sys
import uuid
from pathlib import Path

import dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
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

BASE_DIR = Path(__file__).resolve().parent
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
REDIS_URL = os.environ["REDIS_URL"]

for required_setting in (
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
):
    if not os.getenv(required_setting):
        raise RuntimeError(f"Missing required environment variable: {required_setting}")

# Generate unique session_id and user_id once
session_id = f"session-{uuid.uuid4().hex[:8]}"
users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

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

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: Path) -> QdrantVectorStore:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (Path): Path to the JSON file containing smartphone data.

    :returns
        QdrantVectorStore: A vector store connected to the cloud collection.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    collection_name = "smartphones"
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
            embedding=embeddings_model,
        )
        qdrant_store.add_documents(documents=documents)
        return qdrant_store

    return QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        embedding=embeddings_model,
        collection_name=collection_name,
    )


# Initialize the vector store
product_db = embed_documents(BASE_DIR / "datasets" / "smartphones.json")


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


# ---------------------------
# Main Conversation Loop
# ---------------------------
def main():
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    # Fetch prompts from Langfuse
    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    review_lf_prompt = langfuse.get_prompt("review_system_prompt")
    goodbye_lf_prompt = langfuse.get_prompt("goodbye_system_prompt")

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

    # goodbye_system_prompt is a text-type Langfuse prompt, so
    # get_langchain_prompt() returns the template string directly (not a list of messages)
    goodbye_prompt = PromptTemplate.from_template(
        goodbye_lf_prompt.get_langchain_prompt()
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_lf_prompt}

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
    goodbye_chain = goodbye_prompt | llm

    # Load NeMo Guardrails configuration
    guardrails_config = RailsConfig.from_path("config/")

    # Create guardrails instance for input validation only
    # We'll use it separately to validate user input before the chain
    input_rails = RunnableRails(guardrails_config, input_key="user_input")

    # Initialize the Langfuse handler once for the entire conversation
    langfuse_handler = CallbackHandler()

    # Initialize Redis chat history with TTL (1 hour = 3600 seconds)
    redis_history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=REDIS_URL,
        ttl=3600  # Messages expire after 1 hour
    )

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                # Create a parent span for the goodbye message
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query",
                    input=user_input
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id
                    ):
                        goodbye_message = goodbye_chain.invoke(
                            {"user_id": user_id},
                            config={
                                "run_name": "goodbye-message",
                                "callbacks": [langfuse_handler]
                            }
                        )

                        # Set the output on the parent span
                        span.update(output=goodbye_message.content)

                print(f"System: {goodbye_message.content}")

                # Collect user feedback about the entire conversation
                feedback = input("\nWas this conversation helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                # Score at the session level (not individual trace)
                langfuse.create_score(
                    session_id=session_id,  # Use the session_id from the start of the conversation
                    name="conversation_usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )

                print("\nThank you for your feedback!")
                break

            # Load conversation history from Redis
            conversation = list(redis_history.messages)

            # Create user message
            user_message = HumanMessage(user_input)
            # Add to in-memory conversation for this turn
            conversation.append(user_message)

            # Create a parent span for this user query to group all chain invocations
            with langfuse.start_as_current_observation(
                as_type="span",
                name="user-query",
                input=user_input
            ) as span:
                # Propagate trace attributes to all child observations
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id
                ):

                    validation_result = input_rails.rails.generate(
                        messages=[{"role": "user", "content": user_input}],
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
                        print(f"System: {rail_response}")
                        continue  # Skip saving to Redis and proceed to next input

                    # Context chain invocation (with trimmer to limit tokens)
                    ai_with_tools = context_chain.invoke(
                        {"user_input": user_input, "conversation": conversation},
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
                        {"user_id": user_id, "user_input": user_input, "conversation": conversation},
                        config={
                            "run_name": "final-response",
                            "callbacks": [langfuse_handler]
                        }
                    )

                # Set the output on the parent span
                span.update(output=response.content)

            print(f"System: {response.content}")

            # Save ONLY clean messages to Redis (user input and final AI response)
            # Tool calls and intermediate messages are NOT saved
            redis_history.add_message(user_message)
            redis_history.add_message(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
