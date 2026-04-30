"""
app/rag_engine.py - PDF ingestion, embedding, ChromaDB storage, and retrieval.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from pydantic import BaseModel

from app.config import DocumentInfo, settings


# ──────────────────────────────────────────────────────────────
# Singleton embedding model
# ──────────────────────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: {}", settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},  # embeddings always on CPU to save GPU VRAM
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embeddings


# ──────────────────────────────────────────────────────────────
# Vector store helper
# ──────────────────────────────────────────────────────────────

def get_vector_store() -> Chroma:
    """Return a LangChain Chroma vector store backed by a persistent directory."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


# ──────────────────────────────────────────────────────────────
# Document metadata store (simple in-memory + JSON persistence)
# ──────────────────────────────────────────────────────────────

import json

_META_FILE = Path(settings.chroma_persist_dir) / "doc_metadata.json"


def _load_meta() -> dict:
    if _META_FILE.exists():
        return json.loads(_META_FILE.read_text())
    return {}


def _save_meta(meta: dict) -> None:
    _META_FILE.parent.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(json.dumps(meta, indent=2))


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def ingest_pdf(file_path: Path) -> DocumentInfo:
    """
    Load a PDF, split into chunks, embed and store in ChromaDB.
    Skips re-ingestion if the file hash already exists.
    """
    file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
    meta = _load_meta()

    if file_hash in meta:
        logger.info("PDF '{}' already indexed. Skipping.", file_path.name)
        return DocumentInfo(**meta[file_hash])

    logger.info("Ingesting PDF: {}", file_path.name)

    loader = PyPDFLoader(str(file_path))
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Tag each chunk with source metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = file_path.name
        chunk.metadata["file_hash"] = file_hash

    vs = get_vector_store()
    vs.add_documents(chunks)

    doc_info = DocumentInfo(
        filename=file_path.name,
        num_chunks=len(chunks),
        upload_time=datetime.now(timezone.utc).isoformat(),
        file_size_kb=round(file_path.stat().st_size / 1024, 2),
    )

    meta[file_hash] = doc_info.model_dump()
    _save_meta(meta)

    logger.info("Indexed {} chunks from '{}'.", len(chunks), file_path.name)
    return doc_info


def delete_document(filename: str) -> bool:
    """Remove all chunks belonging to a document from ChromaDB."""
    meta = _load_meta()

    # Find the hash for this filename
    target_hash = None
    for h, info in meta.items():
        if info["filename"] == filename:
            target_hash = h
            break

    if target_hash is None:
        logger.warning("Document '{}' not found in index.", filename)
        return False

    vs = get_vector_store()
    # Delete by metadata filter
    vs._collection.delete(where={"file_hash": target_hash})

    del meta[target_hash]
    _save_meta(meta)
    logger.info("Deleted document '{}' from index.", filename)
    return True


def list_documents() -> List[DocumentInfo]:
    """Return all indexed documents."""
    meta = _load_meta()
    return [DocumentInfo(**v) for v in meta.values()]


def get_document_count() -> int:
    return len(_load_meta())


def retrieve_context(query: str, k: int = None) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.
    Returns them as a single concatenated string.
    """
    k = k or settings.rag_top_k
    vs = get_vector_store()
    docs = vs.similarity_search(query, k=k)

    if not docs:
        return ""

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)
