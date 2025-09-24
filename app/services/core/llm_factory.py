# app/services/core/llm_factory.py
from typing import Optional, Union

from app.config import Config

TEMP = Config.RUNTIME_TEMPERATURE
MAX_NEW = Config.RUNTIME_MAX_TOKENS

# vLLM(OpenAI 호환)
from langchain_openai import ChatOpenAI

# Ollama
from langchain_community.chat_models import ChatOllama

# 반환 타입(둘 다 동일한 Runnable 인터페이스를 가짐)
LLMType = Union[ChatOpenAI, ChatOllama]


def get_model(*, stream: bool, timeout: Optional[float]) -> LLMType:
    if Config.USE_OLLAMA:
        return ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=TEMP,
            num_predict=MAX_NEW,  # Ollama 용
            disable_streaming=not stream,
            keep_alive="180s",
            timeout=timeout,
        )
    else:
        return ChatOpenAI(
            base_url=Config.VLLM_BASE_URL.rstrip("/"),
            api_key=(Config.VLLM_API_KEY or None),
            model=Config.VLLM_MODEL,
            temperature=TEMP,
            max_tokens=MAX_NEW,  # OpenAI 호환(vLLM) 용
            streaming=stream,
            timeout=timeout,
        )
