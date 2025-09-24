# app/services/helpers.py
import json
import os
from typing import Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from sse_starlette.sse import ServerSentEvent

from app.config import Config

SYSTEM_KO = Config.SYSTEM_PROMPT


# =========================================
# Prompt helpers
# =========================================
def build_prompt(question: str, docs: List[Any]) -> list[SystemMessage | HumanMessage]:
    context = _join_context(docs)
    return [
        SystemMessage(content=SYSTEM_KO),  # 한국어 고정 지시
        HumanMessage(content=f"[Question]\n{question}\n\n[Context]\n{context}")
    ]


def _join_context(docs: List[Any]) -> str:
    """RAG 컨텍스트를 사람이 읽기 쉬운 블록으로 결합."""
    blocks: List[str] = []
    for i, d in enumerate(docs, 1):
        content = getattr(d, "page_content", "") or ""
        blocks.append(content)
    return "\n\n---\n\n".join(blocks)


# =========================================
# Source helpers
# =========================================
def list_sources(docs: List[Any]) -> List[str]:
    """각 문서의 source 메타에서 파일명만 추출."""
    out: List[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source")
        if src:
            out.append(os.path.basename(src))
    # 중복 제거 (순서 유지)
    return list(dict.fromkeys(out))


# =========================================
# SSE helpers
# =========================================
def _to_data_payload(obj: Any) -> str:
    """SSE data에 실릴 문자열로 변환.
       - str: 그대로
       - dict/list: JSON 문자열
       - 기타: str()
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    try:
        return str(obj)
    except Exception:
        return ""


def send(obj: Any) -> str:
    """EventSourceResponse가 data: ...\\n\\n로 감쌉니다."""
    return _to_data_payload(obj)


def event(name: str, obj: Any) -> ServerSentEvent:
    """특정 event명으로 data 전송"""
    return ServerSentEvent(event=name, data=_to_data_payload(obj))


def ping() -> ServerSentEvent:
    """코멘트 기반 heartbeat (data 프레임 아님)"""
    return ServerSentEvent(comment="ping")
