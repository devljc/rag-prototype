# RAG 프로토타입 프로젝트(학습/데모)

- 이 프로젝트는 FastAPI를 기반으로 한 RAG(Retrieval-Augmented Generation) 프로토타입입니다. 
- 벡터 데이터베이스(Chroma)를 사용하여 문서 임베딩을 저장하고, vLLM 또는 Ollama와 같은 LLM(Large Language Model)을 통해 사용자 질문에 대한 답변을 생성합니다.

주요 기능은 다음과 같습니다:
- **FastAPI 기반 API 서버**: 비동기 처리를 통해 높은 성능을 제공합니다.
- **RAG 파이프라인**: LangChain을 활용하여 문서 검색 및 생성 흐름을 구현합니다.
- **실시간 스트리밍 응답**: SSE(Server-Sent Events)를 통해 LLM의 답변을 토큰 단위로 실시간 전송합니다.
- **유연한 LLM 백엔드**: 환경변수 설정을 통해 vLLM과 Ollama 백엔드를 쉽게 전환할 수 있습니다.
- **Redis 캐싱**: 반복적인 질문에 대해 빠른 응답을 제공하기 위해 RAG 결과를 캐싱합니다.
- **Docker 기반 인프라**: Redis, vLLM 등 외부 서비스를 Docker Compose로 관리하여 개발 환경을 통일합니다.

---

## 📂 프로젝트 구조

```
/
├───app/                  # FastAPI 애플리케이션 소스 코드
│   ├───main.py           # API 엔드포인트 정의 (query, stream)
│   ├───config.py         # 환경변수 및 설정 관리
│   ├───schemas.py        # API 요청/응답 모델 (Pydantic)
│   ├───services/         # 비즈니스 로직
│   │   ├───rag.py        # ChromaDB 문서 검색(Retrieval) 로직
│   │   ├───llm_http.py   # 일반 HTTP 요청/응답 RAG 파이프라인
│   │   ├───llm_stream.py # 스트리밍 RAG 파이프라인 (SSE)
│   │   └───core/         # LLM 백엔드 실제 구현체
│   │       ├───llm_factory.py # LLM 클라이언트 생성 팩토리
│   │       └───stream_handler.py # 스트리밍 응답 핸들러
│   ├───deps/             # 의존성 관리 (Redis 클라이언트 등)
│   └───utils/            # 유틸리티 (캐시 키 생성 등)
├───data/                 # RAG에 사용할 원본 문서 (PDF, TXT, MD 등)
│   └───policy.md
├───index/                # 생성된 ChromaDB 벡터 인덱스 저장 위치
├───infra/                # Docker Compose 등 인프라 설정
│   ├───compose/
│   └───nginx/
├───.env                  # (생성 필요) 환경변수 설정 파일
├───ingest.py             # 문서를 읽어 벡터 임베딩을 생성하고 ChromaDB에 저장하는 스크립트
├───requirements.txt      # Python 패키지 의존성
├───docker-compose.yml    # 서비스 실행을 위한 Docker Compose 설정
└───README.md             # 프로젝트 설명 파일
```

---

## ⚙️ 주요 실행 흐름

### 1. 데이터 수집 (Ingestion)

`ingest.py` 스크립트를 통해 `data/` 폴더의 문서를 RAG에 사용할 수 있도록 준비합니다.

1.  **문서 로드**: `data/` 디렉토리에서 `.pdf`, `.txt`, `.md` 파일을 읽어들입니다.
2.  **분할**: 문서를 환경변수(`CHUNK_SIZE`, `CHUNK_OVERLAP`)에 따라 작은 조각으로 나눕니다.
3.  **임베딩**: HuggingFace의 `sentence-transformers` 모델을 사용해 각 조각을 벡터로 변환합니다.
4.  **저장**: 변환된 벡터를 `index/` 디렉토리에 ChromaDB 포맷으로 저장합니다.

### 2. RAG 질의 응답 (API)

사용자가 API 서버에 질문을 보내면 다음과 같은 과정으로 답변이 생성됩니다.

1.  **API 요청**: 사용자가 `/query` (일반) 또는 `/stream` (스트리밍) 엔드포인트로 질문을 보냅니다.
2.  **캐시 확인**: Redis에 동일한 질문에 대한 캐시된 결과가 있는지 확인합니다.
    - **Hit**: 캐시된 결과를 즉시 반환합니다. (`/stream`의 경우도 전체 답변을 한 번에 보냅니다.)
3.  **문서 검색 (Retrieval)**:
    - **Miss**: ChromaDB에서 사용자 질문과 의미적으로 가장 유사한 문서 조각(chunk)들을 `top_k` 개수만큼 검색합니다.
4.  **프롬프트 생성**: 검색된 문서 조각들과 사용자 질문을 조합하여 LLM에게 전달할 프롬프트를 구성합니다.
5.  **LLM 답변 생성**:
    - 설정된 LLM 백엔드(vLLM 또는 Ollama)에 프롬프트를 전달하여 답변 생성을 요청합니다.
    - **`/query`**: 전체 답변이 생성될 때까지 기다립니다.
    - **`/stream`**: 생성되는 토큰을 SSE 스트림으로 받아 문장 단위로 묶어 실시간으로 클라이언트에 전송합니다.
6.  **캐시 저장 및 응답**: 생성된 최종 답변과 출처 문서를 Redis에 캐싱하고 사용자에게 반환합니다.

---

## 🚀 설치 및 실행 방법

### 1. 사전 준비

- Python 3.10+
- Docker 및 Docker Compose

### 2. 환경 설정

1.  **저장소 복제**:
    ```bash
    git clone <repository_url>
    cd rag-proto
    ```

2.  **Python 의존성 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **환경변수 파일 생성**:
    `.env` 파일을 프로젝트 루트에 생성하고 아래 내용을 필요에 맞게 수정합니다.
    ```env
    # --- LLM 백엔드 선택 ---
    # Ollama를 사용하려면 "true"로, vLLM을 사용하려면 "false"로 설정
    USE_OLLAMA="false"

    # --- Ollama 설정 (USE_OLLAMA="true"일 경우) ---
    OLLAMA_BASE_URL="http://localhost:11434"
    OLLAMA_MODEL="qwen2.5:7b-instruct" # 로컬에 설치된 모델명

    # --- vLLM 설정 (USE_OLLAMA="false"일 경우) ---
    # infra/compose/docker-compose.vllm.yml에 정의된 vLLM 컨테이너 주소
    VLLM_BASE_URL="http://localhost:8000/v1"
    VLLM_MODEL="qwen2.5:7b-instruct" # vLLM에서 사용하는 모델명

    # --- Redis 설정 ---
    REDIS_URL="redis://localhost:6379/0"

    # --- Embedding 모델 ---
    HF_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

    # --- RAG 설정 (선택적) ---
    # CHUNK_SIZE=600
    # CHUNK_OVERLAP=120
    # RETRIEVER_TOP_K=4
    ```

### 3. 서비스 실행

1.  **인프라 서비스 시작 (Redis 등)**:
    필요한 서비스를 Docker Compose로 실행합니다. vLLM을 사용하려면 `docker-compose.vllm.yml`도 함께 지정합니다.

    ```bash
    # Redis만 실행
    docker-compose -f docker-compose.yml -f infra/compose/docker-compose.redis.yml up -d

    # Redis와 vLLM 함께 실행
    docker-compose -f docker-compose.yml -f infra/compose/docker-compose.redis.yml -f infra/compose/docker-compose.vllm.yml up -d
    ```

2.  **데이터 수집**:
    `data` 폴더에 있는 문서를 벡터화하여 ChromaDB에 저장합니다.
    ```bash
    python ingest.py
    ```

3.  **API 서버 실행**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
    ```
    서버가 `http://localhost:8001`에서 실행됩니다.

### 4. API 테스트

`requests.http` 파일을 사용하거나 `curl`을 통해 API를 테스트할 수 있습니다.

**스트리밍 요청 예시:**
```bash
curl -X POST http://localhost:8001/stream \
-H "Content-Type: application/json" \
-d '{"question": "보안 정책에 대해 알려줘"}'
```

---

## ✨ 추가 개선 아이디어

- **웹 UI/UX**: Streamlit, Gradio 또는 간단한 HTML/JS 페이지를 추가하여 사용자가 더 쉽게 상호작용할 수 있는 인터페이스를 제공할 수 있습니다.
- **고급 검색 전략**: 단순 유사도 검색 외에 HyDE(Hypothetical Document Embeddings), Multi-Query Retriever 등 더 정교한 검색 기법을 도입하여 검색 정확도를 높일 수 있습니다.
- **RAG 성능 평가**: 생성된 답변의 품질(Faithfulness, Answer Relevancy 등)을 측정하고 평가할 수 있는 프레임워크를 도입하여 모델과 프롬프트를 체계적으로 개선할 수 있습니다.
- **비동기 데이터 수집**: `ingest.py` 스크립트를 API 엔드포인트를 통해 트리거되는 백그라운드 작업으로 전환하여, API를 통해 문서를 동적으로 추가/삭제할 수 있도록 확장할 수 있습니다.
- **보안 강화**: 실제 프로덕션 환경을 위해 API 키 인증, 더 엄격한 CORS 정책 적용 등 보안 기능을 강화할 수 있습니다.
