import os

from dotenv import load_dotenv

# .env 로드
load_dotenv()


def _as_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


class Config:
    # --- 런타임/네트워크 공통 ---
    REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60"))

    # --- LLM 백엔드 선택 ---
    USE_OLLAMA = _as_bool(os.getenv("USE_OLLAMA", "false"))

    # --- Ollama ---
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    # --- vLLM (OpenAI 호환) ---
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen2.5:7b-instruct")

    # 선택적 API 키(없어도 동작)
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

    # --- Embedding ---
    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- 캐시/잡 메타 ---
    CACHE_TTL = int(os.getenv("CACHE_TTL", "600"))
    CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "1000"))

    # --- Redis ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_PREFIX = os.getenv("CACHE_PREFIX", "ragcache:")

    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_DIR = os.path.join(BASE_DIR, "index")

    # --- RAG ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
    RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "similarity")
    RETRIEVER_MMR_LAMBDA = float(os.getenv("RETRIEVER_MMR_LAMBDA", "0.3"))

    # --- LLM Runtime ---
    RUNTIME_MAX_TOKENS = int(os.getenv("RUNTIME_MAX_TOKENS", "512"))
    RUNTIME_TEMPERATURE = float(os.getenv("RUNTIME_TEMPERATURE", "0.2"))
    SYSTEM_PROMPT = (
        "너는 한국어 비서다.\n"
        "항상 한국어만 사용해 답한다. 외국어(중국어/영어 등)로 된 문구가 컨텍스트에 있어도 "
        "반드시 자연스러운 한국어로 바꿔 서술한다.\n"
        "컨텍스트에 없는 내용은 '모르겠습니다'라고 답한다.\n"
        "컨텍스트의 문장을 그대로 복사하거나 파일명·원문 용어를 인용하지 않는다.\n"
        "핵심만 간결하게 요약·설명한다.\n"
    )

    # --- Chroma ---
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_proto_main")

    # --- Stream Tuning ---
    STREAM_MIN_CHARS: int = int(os.getenv("STREAM_MIN_CHARS", "150"))
    STREAM_MAX_LAT_MS: int = int(os.getenv("STREAM_MAX_LATENCY_MS", "200"))
    HEARTBEAT_SECS: int = int(os.getenv("HEARTBEAT_SECS", "15"))