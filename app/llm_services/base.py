"""
app/llm_services/base.py

Shared prompt builder used by all providers.
Keeps prompt logic in one place so both HuggingFace and Ollama
produce identical prompts — only the delivery mechanism differs.
"""
from __future__ import annotations


def build_user_message(
    email_subject: str,
    email_body: str,
    sender_name: str | None,
    context: str,
) -> str:
    """Build the user-facing part of the prompt (context + email content)."""
    sender_label = sender_name or "the sender"
    context_block = (
        f"### Relevant Knowledge Base Context:\n{context}\n\n"
        if context
        else "### Knowledge Base Context:\n(No relevant documents found)\n\n"
    )
    return (
        f"{context_block}"
        f"### Email to Reply To:\n"
        f"From: {sender_label}\n"
        f"Subject: {email_subject}\n\n"
        f"{email_body}\n\n"
        f"### Task:\n"
        f"Write a professional, helpful reply to the email above. "
        f"Use the knowledge base context if relevant. "
        f"Do NOT repeat the subject line or add 'Subject:' at the start. "
        f"Write only the reply body."
    )
