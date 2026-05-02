"""
app/llm_services/__init__.py

Provider registry. Import generate_reply from here — it automatically
delegates to the correct provider based on LLM_PROVIDER in .env.

Usage anywhere in the app:
    from app.llm_services import generate_reply
"""
from __future__ import annotations

from app.config import settings
from loguru import logger


def generate_reply(
    email_subject: str,
    email_body: str,
    sender_name: str | None,
    context: str,
    system_prompt: str,
    reply_signature: str,
    max_tokens: int | None = 512,
) -> str:
    provider = settings.llm_provider.lower()
    logger.debug("LLM provider: {}", provider)

    if provider == "ollama":
        from app.llm_services.ollama_service import generate_reply as _gen
    elif provider == "huggingface":
        from app.llm_services.huggingface_service import generate_reply as _gen
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Valid options: 'ollama', 'huggingface'"
        )

    return _gen(
        email_subject=email_subject,
        email_body=email_body,
        sender_name=sender_name,
        context=context,
        system_prompt=system_prompt,
        reply_signature=reply_signature,
        max_tokens=max_tokens,
    )
