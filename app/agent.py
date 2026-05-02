"""
app/agent.py - Core RAG email agent orchestrator.

Logging structure:
  chroma_db/reply_log.json        — summary log (API-facing, last 200 entries)
  ai_logs/YYYY-MM-DD/             — detailed per-email JSON logs (one file per email)
      <timestamp>_<email_id>.json — full log: email content, context, prompt, response
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.config import AgentConfig, EmailMessage, ReplyLog, settings
from app.gmail_service import fetch_unread_emails, get_gmail_service, send_reply
from app.llm_services import generate_reply
from app.llm_services.base import build_user_message
from app.rag_engine import get_document_count, retrieve_context

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

_CONFIG_FILE = Path(settings.chroma_persist_dir) / "agent_config.json"
_SUMMARY_LOG_FILE = Path(settings.chroma_persist_dir) / "reply_log.json"
_AI_LOGS_DIR = Path("ai_logs")


# ──────────────────────────────────────────────────────────────
# Agent config persistence
# ──────────────────────────────────────────────────────────────

_agent_config: Optional[AgentConfig] = None


def load_agent_config() -> AgentConfig:
    global _agent_config
    if _agent_config is not None:
        return _agent_config
    if _CONFIG_FILE.exists():
        data = json.loads(_CONFIG_FILE.read_text())
        _agent_config = AgentConfig(**data)
    else:
        _agent_config = AgentConfig()
    return _agent_config


def save_agent_config(config: AgentConfig) -> None:
    global _agent_config
    _agent_config = config
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(config.model_dump_json(indent=2))
    logger.info("Agent config saved.")


def update_agent_config(**kwargs) -> AgentConfig:
    config = load_agent_config()
    updated = config.model_copy(update={k: v for k, v in kwargs.items() if v is not None})
    save_agent_config(updated)
    return updated


# ──────────────────────────────────────────────────────────────
# Summary log  (used by /logs API endpoint)
# ──────────────────────────────────────────────────────────────

def _load_summary_logs() -> List[dict]:
    if _SUMMARY_LOG_FILE.exists():
        return json.loads(_SUMMARY_LOG_FILE.read_text())
    return []


def _append_summary_log(entry: ReplyLog) -> None:
    logs = _load_summary_logs()
    logs.insert(0, entry.model_dump())   # newest first
    _SUMMARY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_LOG_FILE.write_text(json.dumps(logs[:200], indent=2))


def get_reply_logs(limit: int = 50) -> List[ReplyLog]:
    return [ReplyLog(**e) for e in _load_summary_logs()[:limit]]


# ──────────────────────────────────────────────────────────────
# Detailed AI log  (one JSON file per email in ai_logs/)
# ──────────────────────────────────────────────────────────────

def _write_detailed_log(
    email: EmailMessage,
    context: str,
    system_prompt: str,
    full_prompt: str,
    reply_body: str,
    reply_with_signature: str,
    sent: bool,
    timestamp: str,
    error: Optional[str] = None,
) -> Path:
    """
    Write a detailed JSON log file for a single email processing event.

    File path: ai_logs/YYYY-MM-DD/<HH-MM-SS>_<email_id>.json

    Contains:
      - Full email metadata and body
      - RAG context chunks retrieved
      - System prompt sent to the LLM
      - Full prompt (system + user message) sent to the LLM
      - Raw LLM response (before signature)
      - Final reply sent (with signature)
      - Whether it was sent or drafted
      - Any error that occurred
    """
    now = datetime.fromisoformat(timestamp)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    log_dir = _AI_LOGS_DIR / date_str
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{time_str}_{email.message_id}.json"

    payload = {
        "meta": {
            "timestamp": timestamp,
            "email_id": email.message_id,
            "thread_id": email.thread_id,
            "sent": sent,
            "auto_reply_enabled": load_agent_config().auto_reply_enabled,
            "llm_provider": settings.llm_provider,
            "model": (
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else settings.hf_model_id
            ),
            "error": error,
        },
        "email": {
            "sender": email.sender,
            "sender_name": email.sender_name,
            "subject": email.subject,
            "received_at": email.received_at,
            "body": email.body,
        },
        "rag": {
            "query": f"{email.subject}\n{email.body[:500]}",
            "top_k": settings.rag_top_k,
            "context_chars": len(context),
            "context": context if context else "(no relevant documents found)",
        },
        "llm": {
            "system_prompt": system_prompt,
            "full_prompt": full_prompt,
            "response_body": reply_body,
            "response_with_signature": reply_with_signature,
            "response_chars": len(reply_with_signature),
        },
    }

    log_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.debug("Detailed log written → {}", log_file)
    return log_file


# ──────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────

def process_email(email: EmailMessage) -> ReplyLog:
    """
    Full pipeline for a single email:
      1. Retrieve relevant context from ChromaDB (RAG)
      2. Build prompt and generate reply via LLM
      3. Send (or draft) the reply via Gmail
      4. Write detailed JSON log to ai_logs/
      5. Append summary entry to reply_log.json
    """
    config = load_agent_config()
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Processing email from '{}' | Subject: '{}'",
        email.sender, email.subject,
    )

    # 1. RAG retrieval
    query = f"{email.subject}\n{email.body[:500]}"
    context = retrieve_context(query, k=settings.rag_top_k)
    logger.debug("Retrieved context ({} chars).", len(context))

    # 2. Build full prompt for logging, then generate reply
    user_message = build_user_message(
        email_subject=email.subject,
        email_body=email.body,
        sender_name=email.sender_name,
        context=context,
    )
    # Full prompt string (Mistral format — also used as reference for Ollama)
    full_prompt = (
        f"<s>[INST] <<SYS>>\n{config.system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    )

    reply_with_signature = generate_reply(
        email_subject=email.subject,
        email_body=email.body,
        sender_name=email.sender_name,
        context=context,
        system_prompt=config.system_prompt,
        reply_signature=config.reply_signature,
        max_tokens=config.max_reply_tokens,
    )
    # Separate reply body from signature for clean logging
    reply_body = reply_with_signature
    if config.reply_signature and reply_with_signature.endswith(config.reply_signature):
        reply_body = reply_with_signature[: -len(config.reply_signature)].rstrip()

    logger.debug("Generated reply ({} chars).", len(reply_with_signature))

    # 3. Send / draft
    sent = False
    if config.auto_reply_enabled:
        service = get_gmail_service()
        sent = send_reply(service, email, reply_with_signature)
    else:
        logger.info(
            "Auto-reply is OFF — reply drafted but NOT sent. "
            "Enable via PUT /agent/config {{\"auto_reply_enabled\": true}}"
        )

    # 4. Write detailed log
    _write_detailed_log(
        email=email,
        context=context,
        system_prompt=config.system_prompt,
        full_prompt=full_prompt,
        reply_body=reply_body,
        reply_with_signature=reply_with_signature,
        sent=sent,
        timestamp=timestamp,
    )

    # 5. Summary log entry (used by /logs API)
    log_entry = ReplyLog(
        email_id=email.message_id,
        subject=email.subject,
        sender=email.sender,
        reply_preview=reply_body[:300],   # body only, no signature padding
        sent=sent,
        timestamp=timestamp,
    )
    _append_summary_log(log_entry)
    return log_entry


# ──────────────────────────────────────────────────────────────
# Polling cycle
# ──────────────────────────────────────────────────────────────

def run_polling_cycle() -> List[ReplyLog]:
    """
    Called by the APScheduler every EMAIL_POLL_INTERVAL seconds.
    Fetches unread emails and processes each one.
    """
    logger.info("Starting email polling cycle…")
    results: List[ReplyLog] = []

    try:
        service = get_gmail_service()
        emails = fetch_unread_emails(service, max_results=settings.email_fetch_limit)
    except Exception as exc:
        logger.error("Failed to fetch emails: {}", exc)
        return results

    if not emails:
        logger.info("No new emails.")
        return results

    for email in emails:
        try:
            log = process_email(email)
            results.append(log)
        except Exception as exc:
            logger.error("Error processing email {}: {}", email.message_id, exc)
            # Still write an error log so nothing is silently lost
            try:
                _write_detailed_log(
                    email=email,
                    context="",
                    system_prompt="",
                    full_prompt="",
                    reply_body="",
                    reply_with_signature="",
                    sent=False,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )
            except Exception:
                pass

    logger.info("Polling cycle complete. Processed {} email(s).", len(results))
    return results


def generate_test_reply(subject: str, body: str, sender_name: str = "Test User") -> dict:
    """
    Simulate the RAG + LLM pipeline and write a detailed log.
    Does NOT interact with Gmail or write summary logs.
    """
    config = load_agent_config()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # 1. Mock email message for logging
    mock_email = EmailMessage(
        message_id=f"test_{uuid.uuid4().hex[:8]}",
        thread_id=f"thread_{uuid.uuid4().hex[:8]}",
        sender="test-sender@example.com",
        sender_name=sender_name,
        subject=subject,
        body=body,
        received_at=timestamp,
    )

    # 2. RAG retrieval
    query = f"{subject}\n{body[:500]}"
    context = retrieve_context(query, k=settings.rag_top_k)

    # 3. Build full prompt for logging
    user_message = build_user_message(
        email_subject=subject,
        email_body=body,
        sender_name=sender_name,
        context=context,
    )
    full_prompt = (
        f"<s>[INST] <<SYS>>\n{config.system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"
    )

    # 4. Generate reply
    reply_with_signature = generate_reply(
        email_subject=subject,
        email_body=body,
        sender_name=sender_name,
        context=context,
        system_prompt=config.system_prompt,
        reply_signature=config.reply_signature,
        max_tokens=config.max_reply_tokens,
    )
    
    reply_body = reply_with_signature
    if config.reply_signature and reply_with_signature.endswith(config.reply_signature):
        reply_body = reply_with_signature[: -len(config.reply_signature)].rstrip()

    # 5. Write detailed log
    log_file = _write_detailed_log(
        email=mock_email,
        context=context,
        system_prompt=config.system_prompt,
        full_prompt=full_prompt,
        reply_body=reply_body,
        reply_with_signature=reply_with_signature,
        sent=False, # Tests are never "sent"
        timestamp=timestamp,
    )

    return {
        "reply": reply_with_signature,
        "context": context if context else "(no relevant documents found)",
        "model": (
            settings.ollama_model
            if settings.llm_provider == "ollama"
            else settings.hf_model_id
        ),
        "log_path": str(log_file),
    }
