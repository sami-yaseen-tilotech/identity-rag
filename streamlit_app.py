import operator
from typing import *
import requests
import tempfile
import os
import streamlit as st

# Show title and description.
st.title("💬 IdentityRAG using Tilores")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# LangChain
from langchain.tools import BaseTool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.tools import Tool

# Tilores
from tilores import TiloresAPI
from langchain_tilores import TiloresTools

class HumanInputStreamlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name: str = "human"
    description: str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        return st.text_input("Agent question:", query)

    async def _arun(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        return self._run(query)

class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]

def initialize_session():
    if 'runnable' not in st.session_state:
        if os.environ.get("LLM_PROVIDER") == "Bedrock":
            llm = ChatBedrock(
                credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
                region_name=os.environ["BEDROCK_REGION"],
                model_id=os.environ["BEDROCK_MODEL_ID"],
                streaming=True,
                model_kwargs={"temperature": 0},
            )
        else:
            model_name = "gpt-4-mini"
            if os.environ.get("OPENAI_MODEL_NAME"):
                model_name = os.environ.get("OPENAI_MODEL_NAME")
            llm = ChatOpenAI(temperature=0, streaming=True, model_name=model_name)
        
        # Setup a connection to the Tilores instance and provide it as a tool
        tilores = TiloresAPI.from_environ()
        tilores_tools = TiloresTools(tilores)
        tools = [
            HumanInputStreamlit(),
            tilores_tools.search_tool(),
            # Note: pdf_tool is not defined in the original code, so I've commented it out
            # pdf_tool,
        ]
        # Use MemorySaver to use the full conversation
        memory = MemorySaver()
        # Use a LangGraph agent
        agent = create_react_agent(llm, tools, checkpointer=memory)

        st.session_state.runnable = agent
        st.session_state.state = ChatState(messages=[])

def main():
    st.title("AI Assistant")

    initialize_session()

    # Display chat history
    for message in st.session_state.state['messages']:
        if isinstance(message, HumanMessage):
            st.chat_message("human").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    # Get user input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Append the new message to the state
        st.session_state.state['messages'] += [HumanMessage(content=user_input)]

        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for event in st.session_state.runnable.stream_events(st.session_state.state, version="v1", config={'configurable': {'thread_id': 'thread-1'}}):
                if event["event"] == "on_chat_model_stream":
                    c = event["data"]["chunk"].content
                    if c and len(c) > 0 and isinstance(c[0], dict) and c[0]["type"] == "text":
                        content = c[0]["text"]
                    elif isinstance(c, str):
                        content = c
                    else:
                        content = ""
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            # Update the placeholder with the full response
            message_placeholder.markdown(full_response)

        # Append the assistant's response to the state
        st.session_state.state['messages'] += [HumanMessage(content=full_response)]

if __name__ == "__main__":
    main()