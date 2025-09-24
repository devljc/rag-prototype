# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryBody
from app.services.llm_http import request as llm_request  # ← rename
from app.services.llm_stream import stream as llm_stream

app = FastAPI(title="RAG API (vLLM + Redis)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
async def query(body: QueryBody):
    try:
        return await llm_request(body)
    except Exception as e:
        # 내부 예외를 통일된 형태로 노출
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


@app.post("/stream")
async def stream_route(body: QueryBody):
    # llm_stream은 EventSourceResponse를 반환(내부에서 SSE 생성)
    return await llm_stream(body)


@app.get("/health")
async def health():
    return {"ok": True}
