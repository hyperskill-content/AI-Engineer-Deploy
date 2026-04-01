from fastapi import FastAPI, Depends

from assistant import AssistantSmartPhone
from model import QueryRequest

#2. Create an instance of app with FastAPI, which initializes Starlette
app = FastAPI()

def get_assistant():
    return AssistantSmartPhone()

@app.get("/hello/")
async def hello():
    return {"message": "Hello World"}

@app.post("/ask/")
def ask(query_request: QueryRequest, assistant: AssistantSmartPhone = Depends(get_assistant)):
    answer = assistant.response_generator(query_request.user_input, query_request.session_id, query_request.user_id)
    return {"answer": answer}