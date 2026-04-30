"""
app/agent.py - Core RAG email agent orchestrator.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.config import AgentConfig, EmailMessage, ReplyLog, settings
from app.gmail_service import fetch_unread_emails, get_gmail_service, send_reply
from app.llm_service import generate_reply
from app.rag_engine import get_document_count, retrieve_context

# ──────────────────────────────────────────────────────────────
# Agent config persistence
# ──────────────────────────────────────────────────────────────

_CONFIG_FILE = Path(settings.chroma_persist_dir) / "agent_config.json"
_LOG_FILE = Path(settings.chroma_persist_dir) / "reply_log.json"

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
# Reply log
# ──────────────────────────────────────────────────────────────

def _load_logs() -> List[dict]:
    if _LOG_FILE.exists():
        return json.loads(_LOG_FILE.read_text())
    return []


def _append_log(entry: ReplyLog) -> None:
    logs = _load_logs()
    logs.insert(0, entry.model_dump())  # newest first
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE.write_text(json.dumps(logs[:200], indent=2))  # keep last 200


def get_reply_logs(limit: int = 50) -> List[ReplyLog]:
    return [ReplyLog(**e) for e in _load_logs()[:limit]]


# ──────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────

def process_email(email: EmailMessage) -> ReplyLog:
    """
    Full pipeline for a single email:
    1. Retrieve relevant context from the vector store.
    2. Generate a reply using the local Mistral LLM.
    3. Send (or draft) the reply via Gmail.
    4. Log the result.
    """
    config = load_agent_config()
    logger.info("Processing email from '{}' | Subject: '{}'", email.sender, email.subject)

    # 1. RAG retrieval
    query = f"{email.subject}\n{email.body[:500]}"
    context = retrieve_context(query, k=settings.rag_top_k)
    logger.debug("Retrieved context ({} chars).", len(context))

    # 2. LLM generation
    reply_text = generate_reply(
        email_subject=email.subject,
        email_body=email.body,
        sender_name=email.sender_name,
        context=context,
        system_prompt=config.system_prompt,
        reply_signature=config.reply_signature,
        max_tokens=config.max_reply_tokens,
    )
    logger.debug("Generated reply ({} chars).", len(reply_text))

    # 3. Send / draft
    sent = False
    if config.auto_reply_enabled:
        service = get_gmail_service()
        sent = send_reply(service, email, reply_text)
    else:
        logger.info("Auto-reply disabled. Reply drafted but NOT sent.")

    # 4. Log
    log_entry = ReplyLog(
        email_id=email.message_id,
        subject=email.subject,
        sender=email.sender,
        reply_preview=reply_text[:300],
        sent=sent,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _append_log(log_entry)
    return log_entry


# ──────────────────────────────────────────────────────────────
# Polling cycle
# ──────────────────────────────────────────────────────────────

def run_polling_cycle() -> List[ReplyLog]:
    """
    Called by the scheduler.  Fetches unread emails and processes each one.
    Returns a list of ReplyLog entries created in this cycle.
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

    logger.info("Polling cycle complete. Processed {} email(s).", len(results))
    return results
