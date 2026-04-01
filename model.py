
from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str

