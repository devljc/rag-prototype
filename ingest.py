# ingest.py
import os
import glob
import argparse

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Config


def load_documents(path: str):
    """지정된 경로에서 PDF, TXT 문서를 읽어 LangChain Document 리스트로 반환"""
    docs = []
    for file in glob.glob(os.path.join(path, "**/*.*"), recursive=True):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file)
        elif file.lower().endswith((".txt", ".md")):
            loader = TextLoader(file, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    """Config 기반으로 문서를 chunk 단위로 자르기"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_embeddings():
    """HuggingFace SentenceTransformers 기반 Embedding 모델 생성"""
    return HuggingFaceEmbeddings(model_name=Config.HF_EMBED_MODEL)


def persist_embeddings(docs, persist_dir: str):
    """Chroma 벡터스토어에 임베딩 후 저장"""
    embeddings = build_embeddings()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=Config.CHROMA_COLLECTION,
    )
    print(f"[OK] Saved {len(docs)} chunks into {persist_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="원본 문서가 들어 있는 폴더 경로",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default=Config.INDEX_DIR,
        help="Chroma 인덱스를 저장할 경로",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading documents from {args.data_dir} ...")
    docs = load_documents(args.data_dir)
    print(f"[INFO] Loaded {len(docs)} docs")

    print("[INFO] Splitting documents ...")
    chunks = split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks "
          f"(chunk_size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})")

    print(f"[INFO] Building embeddings with {Config.HF_EMBED_MODEL} ...")
    persist_embeddings(chunks, args.persist_dir)


if __name__ == "__main__":
    main()