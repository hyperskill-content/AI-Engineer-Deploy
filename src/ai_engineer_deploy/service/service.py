from fastapi import FastAPI, HTTPException

from ai_engineer_deploy import query_assistant
from ai_engineer_deploy.model import QueryRequest, QueryResponse
import uvicorn

api = FastAPI()

@api.post("/ask")
def ask(req: QueryRequest) -> QueryResponse:
    try:
        return query_assistant(
            query=req.user_input,
            user_id=req.user_id,
            session_name=req.session_id
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while querying the model: {e}"
        )



def start():
    uvicorn.run("ai_engineer_deploy.service.service:api", host="0.0.0.0", port=8000, reload=True, log_level="debug")