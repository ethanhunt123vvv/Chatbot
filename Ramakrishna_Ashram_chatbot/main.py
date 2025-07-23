from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent100years import agent

app = FastAPI()

class QueryInput(BaseModel):
    question : str

@app.post("/ask")
async def ask_question(data:QueryInput):
    user_input = data.question
    if user_input.lower() in ['exit','stop']:
        return {'answer': "Session ended."}
    message = [HumanMessage(content=user_input)]
    result = agent.invoke({'messages':message})
    return {'answer':result['messages'][-1].content}
