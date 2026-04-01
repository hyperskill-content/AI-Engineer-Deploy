import os
import sys
import dotenv
import logging
from datetime import datetime, timedelta

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory
from langfuse import Langfuse, observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import trim_messages
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.runnables import RunnableLambda
from tools import Tools

logging.getLogger("nemoguardrails").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.actions").setLevel(logging.ERROR)
logging.getLogger("nemoguardrails.colang").setLevel(logging.ERROR)

# Load environment variables from .env file
dotenv.load_dotenv()

class AssistantSmartPhone:
    def __init__(self):
        self.total_user_budget = 1.1110000
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6380/0")
        self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        self.embeddings_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                show_progress_bar=True,
            )
        Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                 base_url=os.getenv("LANGFUSE_HOST"))
        self.langfuse_client = get_client(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
        )
        self.langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),

        )
        self.config = RailsConfig.from_path("./config")
        self.guardrails = RunnableRails(self.config, input_key="user_input")


    def check_budget(self, current_usage: float) -> bool:
        if current_usage < self.total_user_budget:
            return True
        else:
            return False


    # ---------------------------
    # Tool Call Handling and Response Generation
    # ---------------------------
    def generate_context(self, llm_tools):
        """
        Process tool calls from the language model and collect their responses.

        :param
            llm_with_tools: The language model instance with bound tools.

        :returns
            Toolresponse
        """

        # Process each tool call based on its name
        tools_instance = Tools()
        for tool_call in llm_tools.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                tool_args = tool_call.get("args", {})
                if isinstance(tool_args, dict):
                    return tools_instance.smartphone_info_tool(tool_args.get("model", ""))
                return tools_instance.smartphone_info_tool(tool_args.get("model", ""))
        return ""

    @staticmethod
    def budget_exceeded():
        print("Unfortunately, you've exceeded your current usage. Please try again later.")
        #end_session_tool.invoke({"session_status": "exit"})
        sys.exit(0)


    # ---------------------------
    # Main Conversation Loop
    # ---------------------------
    @observe(name="ai-response")
    def response_generator(self,user_input: str, session_name: str, user_id: str):
        self.langfuse_client.update_current_trace(
            session_id=session_name,
            user_id=user_id
        )
        # langfuse_handler = langfuse_context.get_current_langchain_handler()
        tools_instance = Tools()

        tools = [
            Tool(
                name="SmartphoneInfo",
                func=tools_instance.smartphone_info_tool,
                description="Process tool calls from the language model and collect their responses."
            )
        ]

        # Bind the tools to the language model instance
        llm_with_tools = self.llm.bind_tools(tools)

        def get_redis_history(session_id: str) -> BaseChatMessageHistory:
            return RedisChatMessageHistory(session_id, redis_url=self.redis_url, ttl=120)

        trimmer = trim_messages(
            strategy="last",
            token_counter=self.llm,
            max_tokens=1000,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
        )

        langfuse_context_prompt =  self.langfuse_client.get_prompt(name="smartphone/context", type="chat", label="production")
        langchain_context_prompt = ChatPromptTemplate.from_messages(
            [
                langfuse_context_prompt.get_langchain_prompt()[0],
                MessagesPlaceholder(variable_name="chat_history"),
                langfuse_context_prompt.get_langchain_prompt()[1]
            ]
        )

        langchain_context_prompt.metadata = {"langfuse_prompt": langfuse_context_prompt}

        context_chain = langchain_context_prompt | trimmer | llm_with_tools | RunnableLambda(self.generate_context)
        context_chain_with_history = RunnableWithMessageHistory(
            context_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
        )

        context_chain_with_history_and_rails = self.guardrails | context_chain_with_history

        langfuse_review_prompt = self.langfuse_client.get_prompt("smartphone/review", label="production")
        langchain_review_prompt = ChatPromptTemplate.from_messages(
            [
                langfuse_review_prompt.get_langchain_prompt()[0],
                MessagesPlaceholder(variable_name="chat_history"),
                langfuse_review_prompt.get_langchain_prompt()[1]
            ]
        )

        langchain_review_prompt.metadata = {"langfuse_prompt": langfuse_review_prompt}

        review_chain = langchain_review_prompt | self.llm
        review_chain_with_history = RunnableWithMessageHistory(
            review_chain, get_redis_history, input_messages_key="user_input", history_messages_key="chat_history"
        )

        initial_cost = 0.0

        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=1)

        traces = self.langfuse_client.api.trace.list(user_id="hyper-user", from_timestamp=start_time,
                                              to_timestamp=end_time).data

        for trace in traces:
            current_trace = self.langfuse_client.api.trace.get(trace_id=trace.id)
            cost = current_trace.data.total_cost
            initial_cost += cost

        if not self.check_budget(initial_cost):
            print("Unfortunately, you've exceeded your current usage. Please try again later.")

        current_cost = initial_cost

        try:
            print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
            finalize_response = ""
            if self.check_budget(current_cost):
                context = context_chain_with_history_and_rails.invoke(
                    {"user_input": user_input},
                    config={
                        "configurable": {"session_id": user_id},
                        "callbacks": [self.langfuse_handler], "run_name": "context"
                    }
                )

                context_result = context.get("output") if isinstance(context, dict) else context
                if context_result and context_result.strip().lower() == "i'm sorry, i can't respond to that.":
                    print(f"System: {context_result}")
                    final_response = context_result
                else:
                    final_response_chain = review_chain_with_history.invoke(
                        {"user_input": user_input, "user_id": user_id, "context": context},
                        config={
                            "configurable": {"session_id": user_id},
                            "callbacks": [self.langfuse_handler], "run_name": "final_response"
                        }
                    )
                    print(f"System: {final_response_chain.content}")
                    final_response = final_response_chain.content

                trace_id = self.langfuse_client.api.trace.list().data[0].id
                current_cost += self.langfuse_client.api.trace.get(trace_id).total_cost
                print(f"Your usage so far: {current_cost}")
                return final_response
            else:
                self.budget_exceeded()
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            sys.exit(1)

