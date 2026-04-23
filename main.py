import json
import logging
import os

import dotenv
import uvicorn
from fastapi import FastAPI
from langchain_core.messages import AIMessage, trim_messages, messages_from_dict, message_to_dict
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, Langfuse
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from pydantic import BaseModel
from qdrant_client import QdrantClient

# Load environment variables from .env file
dotenv.load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s   %(message)s', datefmt='%H:%M:%S')
# Ensure only guardrails actions are logged at INFO level to match task_3.md requirements
logging.getLogger("actions.py").setLevel(logging.INFO)

# Initialize Langfuse Callback Handler
langfuse_handler = CallbackHandler()
langfuse_client = Langfuse()

total_cost = 0.0

app = FastAPI()


class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


def update_usage(response):
    global total_cost
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        # gpt-4o-mini prices: $0.15 / 1M input, $0.60 / 1M output
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        total_cost += cost
    return total_cost


# Initialize the LLM with OpenAI API credentials (substitute for other models)
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
    model_kwargs={
        "extra_body": {"metadata": {"session_id": "embeddings-init"}}
    }
)

# Initialize Redis connection for chat history
REDIS_URL = os.environ["REDIS_CONN_STRING"]


def get_redis_history(session_id: str):
    """
    Retrieves or creates a RedisChatMessageHistory instance for the given session ID.
    :param session_id: Unique identifier for the chat session.
    :returns: A RedisChatMessageHistory object.
    """
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=600)  # Setting TTL to 10 minutes


def get_history_messages(session_id: str):
    """
    Retrieves and repairs chat history from Redis, restoring lost tool_calls in AIMessages.
    :param session_id: Unique identifier for the chat session.
    :returns: A list of Message objects.
    """
    history = get_redis_history(session_id)
    messages = history.messages
    for m in messages:
        if isinstance(m, AIMessage) and "_tool_calls" in m.additional_kwargs:
            try:
                # Restore tool_calls from serialized additional_kwargs
                m.tool_calls = json.loads(m.additional_kwargs["_tool_calls"])
            except Exception:
                pass
    return messages


def embed_documents(json_path: str, custom_embeddings=None):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.
    :param
        custom_embeddings: Optional embeddings model to use for the vector store.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    # Use provided embeddings or fallback to global embeddings_model
    embeddings = custom_embeddings if custom_embeddings else embeddings_model

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"]
        )

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if collection_exists:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                embedding=embeddings,
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
def smartphone_info_tool(model: str, user_id: str = "default", session_id: str = "default") -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.
    :param
        user_id (str): The user ID for logging.
    :param
        session_id (str): The session ID for logging.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    # Create a transient embeddings instance with session metadata for this specific tool call
    tool_embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={
            "user": user_id,
            "extra_body": {"metadata": {"session_id": session_id, "tool": "SmartphoneInfo"}}
        }
    )
    
    # Pass the custom embeddings to embed_documents if needed, 
    # but here we just need to ensure the retrieval uses them.
    # Since embed_documents uses the global embeddings_model, we should ideally pass it.
    product_db = embed_documents("datasets/smartphones.json", tool_embeddings)
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
@observe(name="generate-context")
def generate_context(ai_message: AIMessage, session_id: str, user_id: str) -> None:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects
    directly into Redis chat history.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        session_id (str): The current chat session ID.
        user_id (str): The current user ID.
    """
    history = get_redis_history(session_id)

    # Workaround for tool_calls serialization bug in langchain_redis
    if isinstance(ai_message, AIMessage):
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            ai_message.additional_kwargs["_tool_calls"] = json.dumps(ai_message.tool_calls)
        history.add_message(ai_message)
    elif isinstance(ai_message, dict) and "output" in ai_message:
        # If it's a dict from Guardrails, we add it as an AI message
        history.add_ai_message(ai_message["output"])
    else:
        # Fallback for other types
        history.add_ai_message(str(ai_message))

    # Check if the AI message has any tool calls
    if not isinstance(ai_message, AIMessage) or not ai_message.tool_calls:
        return

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to Redis
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                # Inject user_id and session_id into tool arguments
                tool_args = tool_call["args"].copy()
                tool_args["user_id"] = user_id
                tool_args["session_id"] = session_id
                
                tool_output = smartphone_info_tool.invoke(tool_args)
                from langchain_core.messages import ToolMessage
                history.add_message(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        history.add_ai_message(f"An error occurred while processing tool calls: {e}")


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/ask")
@observe(name="ai-response")
async def ask(request: QueryRequest):
    user_input = request.user_input
    user_id = request.user_id
    session_id = request.session_id

    history = get_redis_history(session_id)

    # Define a trimmer to manage message history length
    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=600,
        start_on="human",
        include_system=True,
    )

    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance and add LiteLLM user/metadata
    llm_with_tools = llm.bind_tools(tools).bind(
        user=user_id,
        extra_body={"metadata": {"session_id": session_id}}
    )

    # Bind LiteLLM user/metadata to the base LLM as well
    llm_session = llm.bind(
        user=user_id,
        extra_body={"metadata": {"session_id": session_id}}
    )

    # Get the prompts from Langfuse
    context_system_prompt = langfuse_client.get_prompt("context_system_prompt")
    review_system_prompt = langfuse_client.get_prompt("review_system_prompt")

    context_prompt_messages = context_system_prompt.get_langchain_prompt()
    context_prompt = ChatPromptTemplate.from_messages(
        [context_prompt_messages[0], MessagesPlaceholder(variable_name="conversation")] + context_prompt_messages[1:]
    )
    context_prompt.metadata = {"langfuse_prompt": context_system_prompt}

    review_prompt_messages = review_system_prompt.get_langchain_prompt()
    review_prompt = ChatPromptTemplate.from_messages(
        [review_prompt_messages[0], MessagesPlaceholder(variable_name="conversation")] + review_prompt_messages[1:]
    )
    review_prompt.metadata = {"langfuse_prompt": review_system_prompt}

    context_chain = context_prompt | llm_with_tools
    review_chain = review_prompt | llm_session

    # Load rails config
    rails_config = RailsConfig.from_path("config")

    # Wrap context_chain so it deserializes the conversation (dicts → LangChain message objects)
    # before invoking. This is needed because RunnableRails cannot serialize LangChain message
    # objects, so we convert them to dicts before passing to RunnableRails and back afterward.
    # We also explicitly pass session metadata to ensure it's not lost when called by Rails.
    context_chain_deserializer = RunnableLambda(
        lambda x: context_chain.invoke(
            {
                "user_input": x["user_input"],
                "conversation": messages_from_dict(x.get("conversation", [])),
            },
            config={
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
                "run_name": "context-with-metadata"
            }
        )
    )

    # RunnableRails calls context_chain_deserializer directly when input is not blocked,
    # avoiding the extra LLM call that occurred with the previous approach.
    context_chain_with_rails = RunnableRails(rails_config, runnable=context_chain_deserializer, input_key="user_input")

    # Retrieve and trim messages for context BEFORE adding the current user message,
    # so the user input does not appear twice (once via history, once via {user_input} template).
    conversation = trimmer.invoke(get_history_messages(session_id))

    # Add human message to Redis history
    history.add_user_message(user_input)

    # Serialize LangChain message objects to dicts so RunnableRails can handle them.
    # The deserializer wrapper inside context_chain_with_rails converts them back.
    serialized_conversation = [message_to_dict(m) for m in conversation]

    ai_msg_with_tools = await context_chain_with_rails.ainvoke(
        {"user_input": user_input, "conversation": serialized_conversation},
        config={
            "run_name": "context",
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
            },
        }
    )

    # Check if input rail was triggered
    if isinstance(ai_msg_with_tools, dict) and ai_msg_with_tools.get("output") == "I'm sorry, I can't respond to that.":
        return {"response": ai_msg_with_tools["output"]}

    update_usage(ai_msg_with_tools)

    # Process tool calls and update Redis history
    generate_context(ai_msg_with_tools, session_id, user_id)

    # Update conversation after tools for the final response
    conversation = trimmer.invoke(get_history_messages(session_id))

    response = await review_chain.ainvoke(
        {"user_id": user_id, "user_input": user_input, "conversation": conversation},
        config={
            "run_name": "final-response",
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
            },
        }
    )

    update_usage(response)
    # Add final AI response to Redis history
    history.add_ai_message(response.content)

    return {"response": response.content}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
