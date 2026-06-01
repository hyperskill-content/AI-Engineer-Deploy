# Start from the last coding stage of the previous LLM evals project
import json
import os
from copy import deepcopy

import dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_community.docstore.document import Document
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

set_debug(False)

# Load environment variables from .env file.
# Override the parent shell so a stale OPENAI_API_KEY does not shadow the
# LiteLLM virtual key configured for this project.
dotenv.load_dotenv(override=True)

# Load guardrails configuration
config = RailsConfig.from_path("config/")

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"),
                 base_url=os.getenv("OPENAI_BASE_URL"),
                 api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
                                    base_url=os.getenv("OPENAI_BASE_URL"),
                                    api_key=os.getenv("OPENAI_API_KEY"),
                                    show_progress_bar=True)

REDIS_URL = os.getenv("REDIS_CONNECTION_STRING")

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
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
        content = (f"Model: {entry.get('model', '')}\n"
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
                   f"In Stock: {entry.get('in_stock', '')}")
        documents.append(Document(page_content=content))

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, )

        collection_exists = qdrant_client.collection_exists(
            collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(collection_name=collection_name,
                                            vectors_config=VectorParams(
                                                size=1536,
                                                distance=Distance.COSINE, ), )

            qdrant_store = QdrantVectorStore(client=qdrant_client,
                                             collection_name=collection_name,
                                             embedding=embeddings_model)

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model, collection_name=collection_name,
                url=QDRANT_URL, api_key=QDRANT_API_KEY)

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


product_db = None


def get_product_db():
    """
    Lazily build or load the smartphone vector store when the tool is called.
    """
    global product_db
    if product_db is None:
        product_db = embed_documents("datasets/smartphones.json")
    return product_db


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphonesInfo")
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
        product_db = get_product_db()
        if not product_db:
            return "The smartphone product database is not available."

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
def generate_context(ai_message: AIMessage, conversation: list) -> None:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        None.  Mutates the conversation list in place.
    """
    # construct the conversation history with the AI message containing tool calls
    conversation.append(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(AIMessage(
            content="No tool calls found. Please ensure the model is configured to use tools."))

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to the conversation
        # a message with tool calls is expected to be followed by tool responses
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] in {"SmartphoneInfo", "SmartphonesInfo"}:
                tool_output = smartphone_info_tool.invoke(tool_call)
                conversation.append(tool_output)

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(AIMessage(
            content=f"An error occurred while processing tool calls: {e}"))


trimmer = trim_messages(strategy="last", token_counter=llm, max_tokens=500,
                        start_on="human", end_on=("human", "tool"),
                        include_system=True)

tools = [smartphone_info_tool]

context_lf_prompt = langfuse.get_prompt("context_system_prompt", type="chat")
context_prompt = ChatPromptTemplate.from_messages(
    [context_lf_prompt.get_langchain_prompt()[0],
     MessagesPlaceholder("conversation")])
context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

review_lf_prompt = langfuse.get_prompt("review_system_prompt", type="chat")
review_prompt = ChatPromptTemplate.from_messages(
    [review_lf_prompt.get_langchain_prompt()[0],
     MessagesPlaceholder("conversation")])
review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

langfuse_handler = CallbackHandler()

app = FastAPI()


def get_user_llm(user_id: str):
    """
    Bind request-specific LiteLLM user metadata for budget attribution.
    """
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL"),
                      base_url=os.getenv("OPENAI_BASE_URL"),
                      api_key=os.getenv("OPENAI_API_KEY"),
                      model_kwargs={"user": user_id})


def get_input_rails(user_id: str) -> RunnableRails:
    """
    Create request-scoped input rails so guardrail LLM calls use the request user.
    """
    rails_config = deepcopy(config)
    rails_config.models = []
    return RunnableRails(rails_config, llm=get_user_llm(user_id),
                         input_key="user_input")


def process_user_query(user_input: str, user_id: str, session_id: str) -> str:
    history = RedisChatMessageHistory(session_id=session_id,
                                      redis_url=REDIS_URL, ttl=3600)
    conversation = list(history.messages)
    user_message = HumanMessage(content=user_input)
    conversation.append(user_message)

    input_rails = get_input_rails(user_id)
    validation_result = input_rails.rails.generate(
        messages=[{"role": "user", "content": user_input}],
        options=GenerationOptions(rails=["input"], output_vars=["allowed",
                                                                "triggered_input_rail",
                                                                "bot_message"], ), )
    validation_context = validation_result.output_data or {}
    rail_triggered = (validation_context.get("allowed") is False or bool(
        validation_context.get("triggered_input_rail")))

    if rail_triggered:
        return validation_context.get("bot_message",
                                      "I cannot answer that request.")

    with langfuse.start_as_current_observation(as_type="span",
                                               name="user-query", input={
                "user_input": user_input}) as span:
        with propagate_attributes(session_id=session_id, user_id=user_id):
            user_llm = get_user_llm(user_id)
            context_chain = context_prompt | user_llm.bind_tools(tools)
            review_chain = review_prompt | user_llm
            trimmed_conversation = trimmer.invoke(conversation)
            ai_message = context_chain.invoke({"user_input": user_input,
                                               "conversation": trimmed_conversation},
                                              config={"run_name": "context",
                                                      "callbacks": [
                                                          langfuse_handler]})
            generate_context(ai_message, conversation)

            response = review_chain.invoke(
                {"user_id": user_id, "user_input": user_input,
                 "conversation": conversation},
                config={"run_name": "final-response",
                        "callbacks": [langfuse_handler]})

        span.update(output={"response": response.content})

    history.add_message(user_message)
    history.add_message(response)

    return response.content


class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


class QueryResponse(BaseModel):
    response: str


@app.post("/ask")
def ask(request: QueryRequest) -> QueryResponse:
    try:
        response = process_user_query(user_input=request.user_input,
                                      user_id=request.user_id,
                                      session_id=request.session_id)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
