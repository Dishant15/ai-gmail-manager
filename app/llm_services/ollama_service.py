"""
app/llm_services/ollama_service.py

LLM provider: Ollama (local model server).

Ollama runs as a background service on your machine and exposes a local HTTP API.
This provider sends requests to that API — no model is loaded into Python memory.
Ollama handles all memory management and Apple Silicon (MPS) optimisation natively.

Prerequisites:
  1. Install Ollama:        https://ollama.com/download
  2. Pull your model:       ollama pull qwen2.5:7b
  3. Verify it is running:  ollama list

Recommended models for Apple Silicon (set OLLAMA_MODEL in .env):
  qwen2.5:7b    — best balance of speed and quality on M-series (~4–8 tok/s)
  qwen2.5:14b   — higher quality, needs ~16 GB free RAM
  mistral:7b    — good alternative if you prefer Mistral
  llama3.2:3b   — fastest, lower quality, good for testing

Env variables used:
  OLLAMA_MODEL       — model tag to use (e.g. qwen2.5:7b)
  OLLAMA_BASE_URL    — Ollama API base URL (default: http://localhost:11434)
  OLLAMA_TEMPERATURE — generation temperature (default: 0 = deterministic)
  OLLAMA_MAX_TOKENS  — max tokens to generate (default: 512)
  OLLAMA_KEEP_ALIVE  — how long Ollama keeps model in memory (default: 5m)
"""
from __future__ import annotations

from typing import Optional

import ollama
from loguru import logger
from ollama import ResponseError

from app.config import settings
from app.llm_services.base import build_user_message

_verified: bool = False


def _verify_ollama() -> None:
    """
    Check Ollama is reachable and the configured model is pulled.
    Runs once per process lifetime.
    """
    global _verified
    if _verified:
        return

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        models = client.list()
        available = [m.model for m in models.models]
        model_id = settings.ollama_model

        if not any(model_id in m for m in available):
            logger.warning(
                "Ollama | Model '{}' not found. Available: {}. "
                "Run:  ollama pull {}",
                model_id, available, model_id,
            )
        else:
            logger.info(
                "Ollama | Connected. Model '{}' is ready.", model_id
            )
        _verified = True

    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at '{settings.ollama_base_url}'. "
            f"Make sure Ollama is running — open the Ollama app "
            f"or run 'ollama serve' in a terminal. Error: {exc}"
        )


def generate_reply(
    email_subject: str,
    email_body: str,
    sender_name: Optional[str],
    context: str,
    system_prompt: str,
    reply_signature: str,
    max_tokens: Optional[int] = 512,
) -> str:
    """
    Generate an email reply via the local Ollama API.
    Uses chat-style messages (system + user roles) for clean prompt handling.
    """
    _verify_ollama()

    user_message = build_user_message(email_subject, email_body, sender_name, context)

    logger.debug(
        "Ollama | Generating reply  model={}  subject='{}'",
        settings.ollama_model, email_subject,
    )

    # Build options — only include num_predict if max_tokens is set
    options = {
        "temperature":    settings.ollama_temperature,
        "repeat_penalty": 1.1,
    }
    if max_tokens:
        options["num_predict"] = max_tokens

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            options=options,
            keep_alive=settings.ollama_keep_alive,
        )
        print("responce from OLLAMA : ", response)
    except ResponseError as exc:
        raise RuntimeError(
            f"Ollama API error for model '{settings.ollama_model}': {exc}. "
            f"Make sure the model is pulled: ollama pull {settings.ollama_model}"
        ) from exc

    reply = response.message.content.strip()
    logger.debug("Ollama | Reply generated ({} chars).", len(reply))

    return reply + "\n" + reply_signature
