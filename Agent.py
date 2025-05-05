# Step1: Setup API keys for groq and tavily
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY=os.environ.get("Groq_API_Key")
Tavily_API_KEY=os.environ.get("Tavily_API_Key")
OpenAI_API_KEY=os.environ.get("OpenAI_API_Key")

# GROQ_API_KEY=st.secrets("Groq_API_Key")
# Tavily_API_KEY=st.secrets("Tavily_API_Key")
# OpenAI_API_KEY=st.secrets("OpenAI_API_Key")

# Step 2: Setup LLM and Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools .tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model='gpt-4o-mini')
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#step 3: Setup Ai Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
system_prompt="Act as an AI chatbot who is smart and friendly."

def get_response_from_AI_Agent(llm_id,query,allow_search,system_prompt,provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)
    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier= system_prompt
    )

    # query="Tell me about the trends in crypto markets"
    state={"message":query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]
    return(ai_messages[-1])
