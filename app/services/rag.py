from typing import List

from chromadb.config import Settings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Config


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=Config.HF_EMBED_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=Config.INDEX_DIR,
        embedding_function=get_embeddings(),
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        collection_name=Config.CHROMA_COLLECTION,
    )


def retrieve(query: str, k: int | None = None) -> List[Document]:
    k = k or Config.RETRIEVER_TOP_K
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k}).invoke(query)


def format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source") or f"doc_{i}"
        blocks.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)
