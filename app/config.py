"""
app/config.py - Pydantic settings & shared data models
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────────────────────
# Settings (loaded from .env)
# ──────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gmail
    gmail_credentials_path: str = "credentials.json"
    gmail_token_path: str = "token.json"
    gmail_address: str = "your@gmail.com"

    # HuggingFace / Mistral
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_device: str = "auto"
    hf_load_in_4bit: bool = True
    hf_max_new_tokens: int = 512

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "email_agent_knowledge"

    # File storage
    upload_dir: str = "./uploads"

    # Email polling
    email_poll_interval: int = 60
    email_fetch_limit: int = 10

    # RAG
    rag_top_k: int = 5
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("upload_dir", "chroma_persist_dir", mode="after")
    @classmethod
    def ensure_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


settings = Settings()


# ──────────────────────────────────────────────────────────────
# Pydantic data models
# ──────────────────────────────────────────────────────────────

class EmailMessage(BaseModel):
    """Represents an incoming Gmail message."""
    message_id: str
    thread_id: str
    sender: str
    sender_name: Optional[str] = None
    subject: str
    body: str
    received_at: str


class AgentConfig(BaseModel):
    """Agent behaviour stored in memory / persisted as JSON."""
    system_prompt: str = Field(
        default=(
            "You are a helpful email assistant. "
            "Use the provided context from the knowledge base to answer emails accurately and professionally. "
            "Keep replies concise and polite. "
            "If the context does not contain enough information to answer, say so honestly."
        ),
        description="System prompt that controls the agent's persona and behaviour.",
    )
    reply_signature: str = Field(
        default="\n\nBest regards,\nYour AI Assistant",
        description="Signature appended to every reply.",
    )
    auto_reply_enabled: bool = Field(
        default=False,
        description="When False the agent drafts replies but does not send them.",
    )
    max_reply_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum tokens for generated reply.",
    )


class AgentConfigUpdate(BaseModel):
    """Partial update payload for AgentConfig."""
    system_prompt: Optional[str] = None
    reply_signature: Optional[str] = None
    auto_reply_enabled: Optional[bool] = None
    max_reply_tokens: Optional[int] = None


class DocumentInfo(BaseModel):
    """Metadata for an uploaded PDF document."""
    filename: str
    num_chunks: int
    upload_time: str
    file_size_kb: float


class ReplyLog(BaseModel):
    """Record of a sent / drafted reply."""
    email_id: str
    subject: str
    sender: str
    reply_preview: str
    sent: bool
    timestamp: str


class StatusResponse(BaseModel):
    status: str
    message: str


class PollingStatus(BaseModel):
    running: bool                # True = job is active and scheduled, False = job is paused
    interval_seconds: int
    gmail_address: str
    auto_reply_enabled: bool
    documents_indexed: int
