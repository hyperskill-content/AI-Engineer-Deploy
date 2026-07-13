from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions
from pathlib import Path

BASE_DIR = Path (__file__).resolve ().parent

def initialize_components(llm, langfuse, smartphone_info_tool, config_path) -> tuple:
    """
    Initialize reusable LangChain, Guardrails, and Langfuse components.
    Returns:
        tuple: A tuple containing:
            - context_chain: Chain used to produce tool-aware context.
            - review_chain: Chain used to generate the final response.
            - input_rails: Guardrails runnable used to validate user input.
            - langfuse_handler: Callback handler used for Langfuse tracing.
    """

    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools (tools)

    # Fetch prompts from Langfuse
    context_lf_prompt = langfuse.get_prompt ("context_system_prompt")
    review_lf_prompt = langfuse.get_prompt ("review_system_prompt")

    # Create LangChain prompts from Langfuse prompts
    context_prompt = ChatPromptTemplate.from_messages ([
        context_lf_prompt.get_langchain_prompt ()[0],
        MessagesPlaceholder (variable_name="conversation"),
    ])
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_prompt = ChatPromptTemplate.from_messages ([
        review_lf_prompt.get_langchain_prompt ()[0],
        MessagesPlaceholder (variable_name="conversation"),
    ])
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    # Create message trimmer for context chain to limit token usage
    trimmer = trim_messages (
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
    # guardrails_config = RailsConfig.from_path("config/")
    guardrails_config = RailsConfig.from_path (str (BASE_DIR / "config"))

    # Create a guardrails instance for input validation only.
    # We'll use it separately to validate user input before the chain.
    input_rails = RunnableRails (guardrails_config, input_key="user_input")

    # Initialize the Langfuse handler once for the entire conversation
    langfuse_handler = CallbackHandler ()

    return context_chain, review_chain, input_rails, langfuse_handler
