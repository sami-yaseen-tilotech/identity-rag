import operator
from typing import *
import requests
import tempfile
import os
import streamlit as st
import asyncio

# LangChain
from langchain.tools import BaseTool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
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
st.title("IdentityRAG using Tilores 💬")
st.write(
    "This demo shows how to use Tilores IdentityRAG with a Large Language Model (LLM) to create an internal customer service chat bot, based on customer data from scattered internal sources. This can work with any internal or externally sourced data."
)
st.write(
    "Want to use your own data? Contact Us by email: identityrag@tilores.io or visit [tilores.io](https://tilores.io/RAG?utm_source=streamlit&utm_medium=embed&utm_campaign=identityrag-demo)."
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

def format_messages_for_bedrock(messages):
    formatted_messages = []
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            formatted_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
        
        # Ensure alternating user/assistant messages
        if i > 0 and formatted_messages[-1]["role"] == formatted_messages[-2]["role"] == "assistant":
            formatted_messages.insert(-1, {"role": "user", "content": "Please continue."})
    
    return formatted_messages

def initialize_session():
    if 'runnable' not in st.session_state:
        llm_provider = os.environ.get("LLM_PROVIDER", "OpenAI").lower()
        
        if llm_provider == "bedrock":
            llm = ChatBedrock(
                credentials_profile_name=None,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region_name=os.environ["BEDROCK_REGION"],
                model_id=os.environ["BEDROCK_MODEL_ID"],
                streaming=True,
                model_kwargs={"temperature": 0},
            )
        else:  # Default to OpenAI
            model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
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
        st.session_state.llm_provider = llm_provider

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main():
    st.subheader("Try asking: search for Sophie Muller")

    initialize_session()

    # Display chat history
    for message in st.session_state.state['messages']:
        if isinstance(message, HumanMessage):
            st.chat_message("human").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

    # Get user input
    user_input = st.chat_input("Try asking 'search for Sophie Muller', then ask follow-up questions")

    if user_input:
        # Display the new user message immediately
        st.chat_message("human").write(user_input)

        # Append the new message to the state
        st.session_state.state['messages'] += [HumanMessage(content=user_input)]

        # Format messages for Bedrock if necessary
        if st.session_state.llm_provider == "bedrock":
            formatted_messages = format_messages_for_bedrock(st.session_state.state['messages'])
            state_for_llm = ChatState(messages=formatted_messages)
        else:
            state_for_llm = st.session_state.state

        # Create placeholders for the assistant's response and agent questions
        assistant_placeholder = st.empty()
        agent_question_placeholder = st.empty()

        full_response = ""
        agent_question = ""

        async def process_stream():
            nonlocal full_response, agent_question
            async for event in st.session_state.runnable.astream_events(
                state_for_llm,
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
                    with assistant_placeholder.container():
                        st.markdown(full_response + "▌")
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    tool_input = event["data"]["input"]
                    if tool_name == "human":
                        agent_question = tool_input
                        with agent_question_placeholder.container():
                            st.info(f"Agent question: {agent_question}")
                            user_answer = st.text_input("Your answer:")
                            if user_answer:
                                return user_answer

        while True:
            user_answer = run_async(process_stream())
            if user_answer:
                st.session_state.state['messages'] += [HumanMessage(content=user_answer)]
            else:
                break

        # Update the assistant's response
        with assistant_placeholder.container():
            st.markdown(full_response)

        # Append the assistant's response to the state
        st.session_state.state['messages'] += [AIMessage(content=full_response)]

if __name__ == "__main__":
    main()