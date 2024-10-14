import operator
from typing import *
import requests
import tempfile
import os
import streamlit as st
import asyncio

# LangChain
from langchain.tools import BaseTool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
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

# Show title and description.
st.title("💬 IdentityRAG using Tilores")
st.write(
    "This demo demonstrates how connecting an LLM to Tilores, an entity resolution system, "
    "via IdentityRAG improves data analysis by resolving ambiguities. "
    "Tilores identifies and merges duplicate or inconsistent records, "
    "while the LLM provides accurate insights using the clean, disambiguated data. "
    "This approach leads to more precise and context-aware analysis."
)
st.write(
    "Want to use your own data? setup your own [here](https://app.tilores.io)."
)
st.write(
    "Or reach out to us by email: identityrag@tilores.io or use the chat on [tilores.io](https://tilores.io/RAG)"
)

st.write(
    "We will help you setup your own app while keeping your data secured."
)

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
                credentials_profile_name=None,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.environ["AWS_SESSION_TOKEN"],
                region_name=os.environ["BEDROCK_REGION"],
                model_id=os.environ["BEDROCK_MODEL_ID"],
                streaming=True,
                model_kwargs={"temperature": 0},
            )
        else:
            model_name = "gpt-4o-mini"
            if os.environ.get("OPENAI_MODEL_NAME"):
                model_name = os.environ.get("OPENAI_MODEL_NAME")
            llm = ChatOpenAI(temperature=0, streaming=True, model_name=model_name)
        
        tilores = TiloresAPI.from_environ()
        tilores_tools = TiloresTools(tilores)
        tools = [
            HumanInputStreamlit(),
            tilores_tools.search_tool(),
        ]
        memory = MemorySaver()
        agent = create_react_agent(llm, tools, checkpointer=memory)

        st.session_state.runnable = agent
        st.session_state.state = ChatState(messages=[])

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main():
    st.subheader("Try asking: search for Sophie Muller")

    initialize_session()

    for message in st.session_state.state['messages']:
        if isinstance(message, HumanMessage):
            st.chat_message("human").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

    user_input = st.chat_input("Try asking to search for Sophie Muller, then ask follow up questions")

    if user_input:
        st.session_state.state['messages'] += [HumanMessage(content=user_input)]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            async def process_stream():
                nonlocal full_response
                async for event in st.session_state.runnable.astream_events(
                    st.session_state.state,
                    version="v1",
                    config={'configurable': {'thread_id': 'thread-1'}}
                ):
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

            run_async(process_stream())

            message_placeholder.markdown(full_response)

        st.session_state.state['messages'] += [AIMessage(content=full_response)]

if __name__ == "__main__":
    main()