from pydantic import BaseModel, Field
from typing import Optional, Any

class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 4
    admin: bool = False
