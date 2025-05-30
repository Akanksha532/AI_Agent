#Step 1: Setup Pydantic Model (Schema Validation)
import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from typing import List
class RequestState(BaseModel):
    model_name:str
    model_provider:str
    system_prompt:str
    messages:List[str]
    allow_search:bool

#step 2: setup AI agent from frontend request
from fastapi import FastAPI
from Agent import get_response_from_AI_Agent

ALLOWED_MODEL_NAMES=["llama-70b-8192","mixtral-8x7b-32768",'llama-3.3-70b-versatile',"gpt-4o-mini"]
app=FastAPI(title="LangGraph AI Agent")
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the Chatbot using LangGraph and search tools.
    It dynamically selects the model specified in the request.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error: Invalid model name. Kindly select a valid AI Model"}
# Create AI agent and get response
    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_search
    system_prompt=request.system_prompt
    provider=request.model_provider

    response=get_response_from_AI_Agent(llm_id,query,allow_search,system_prompt,provider)
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
