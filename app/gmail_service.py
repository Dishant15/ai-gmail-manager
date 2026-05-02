"""
app/gmail_service.py - Gmail OAuth2 authentication, email fetching and reply sending.

Email body cleaning is handled by app.utils.clean_email_body which converts
HTML emails to Markdown before they enter the RAG pipeline and LLM prompt.
"""
from __future__ import annotations

import base64
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger

from app.config import EmailMessage, settings
from app.utils import clean_email_body

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


# ──────────────────────────────────────────────────────────────
# OAuth
# ──────────────────────────────────────────────────────────────

def get_gmail_service():
    """
    Authenticate with Gmail via OAuth2 and return a service object.
    On first run opens a browser for authorisation.
    Subsequent runs reuse the cached token.json.
    """
    creds: Optional[Credentials] = None
    token_path = Path(settings.gmail_token_path)
    creds_path = Path(settings.gmail_credentials_path)

    if not creds_path.exists():
        raise FileNotFoundError(
            f"Gmail credentials file not found at '{creds_path}'. "
            "Download credentials.json from Google Cloud Console → APIs & Services → Credentials."
        )

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        logger.info("Gmail OAuth token saved to {}", token_path)

    service = build("gmail", "v1", credentials=creds)
    logger.info("Gmail service authenticated successfully.")
    return service


# ──────────────────────────────────────────────────────────────
# Body decoding
# ──────────────────────────────────────────────────────────────

def _extract_raw_body(payload: dict) -> tuple[str, str]:
    """
    Recursively extract body content from a Gmail message payload.

    Returns (body_text, mime_type) where mime_type is either
    'text/plain' or 'text/html' — plain text is always preferred
    over HTML when both are present in a multipart message.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    # Plain text — best case, return immediately
    if mime_type == "text/plain" and body_data:
        text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        return text, "text/plain"

    # Multipart — recurse, preferring plain text parts first
    if mime_type.startswith("multipart/"):
        parts = payload.get("parts", [])

        # First pass: look for plain text part
        for part in parts:
            text, found_mime = _extract_raw_body(part)
            if text and found_mime == "text/plain":
                return text, "text/plain"

        # Second pass: accept HTML if no plain text found
        for part in parts:
            text, found_mime = _extract_raw_body(part)
            if text:
                return text, found_mime

    # HTML fallback
    if mime_type == "text/html" and body_data:
        html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        return html, "text/html"

    return "", ""


def _parse_address(raw: str) -> tuple[str, str]:
    """Return (name, address) from a 'Name <email>' header."""
    match = re.match(r"^(.*?)\s*<(.+?)>$", raw.strip())
    if match:
        return match.group(1).strip().strip('"'), match.group(2).strip()
    return "", raw.strip()


# ──────────────────────────────────────────────────────────────
# Fetch
# ──────────────────────────────────────────────────────────────

def fetch_unread_emails(service, max_results: int = 10) -> List[EmailMessage]:
    """
    Fetch unread emails from the inbox.

    Body processing pipeline per email:
      1. Extract raw body (plain text preferred, HTML fallback)
      2. Pass through clean_email_body() from utils:
           - HTML bodies are converted to Markdown
           - Plain text bodies are whitespace-normalised
      3. Store the clean body in EmailMessage.body

    Marks each fetched email as read so it is not processed again.
    """
    messages: List[EmailMessage] = []
    try:
        result = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread in:inbox", maxResults=max_results)
            .execute()
        )
        items = result.get("messages", [])
        logger.info("Found {} unread message(s).", len(items))

        for item in items:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=item["id"], format="full")
                .execute()
            )

            headers = {
                h["name"].lower(): h["value"]
                for h in msg["payload"].get("headers", [])
            }
            sender_raw = headers.get("from", "")
            sender_name, sender_addr = _parse_address(sender_raw)
            subject = headers.get("subject", "(no subject)")
            date_str = headers.get("date", "")

            # Extract raw body then clean it
            raw_body, mime_type = _extract_raw_body(msg["payload"])
            clean_body = clean_email_body(raw_body)

            logger.debug(
                "Email body: mime={} raw_chars={} clean_chars={}",
                mime_type, len(raw_body), len(clean_body),
            )

            messages.append(
                EmailMessage(
                    message_id=msg["id"],
                    thread_id=msg["threadId"],
                    sender=sender_addr,
                    sender_name=sender_name or None,
                    subject=subject,
                    body=clean_body,
                    received_at=date_str,
                )
            )

            # Mark as read so we don't process it again
            service.users().messages().modify(
                userId="me",
                id=item["id"],
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()

    except HttpError as e:
        logger.error("Gmail API error while fetching emails: {}", e)

    return messages


# ──────────────────────────────────────────────────────────────
# Send
# ──────────────────────────────────────────────────────────────

def send_reply(service, original: EmailMessage, reply_body: str) -> bool:
    """
    Send a reply in the same thread as the original email.
    Returns True on success.
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["To"] = original.sender
        msg["Subject"] = (
            original.subject
            if original.subject.lower().startswith("re:")
            else f"Re: {original.subject}"
        )
        msg["In-Reply-To"] = original.message_id
        msg["References"] = original.message_id
        msg.attach(MIMEText(reply_body, "plain"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        service.users().messages().send(
            userId="me",
            body={"raw": raw, "threadId": original.thread_id},
        ).execute()
        logger.info("Reply sent to {} for subject '{}'", original.sender, original.subject)
        return True

    except HttpError as e:
        logger.error("Failed to send reply: {}", e)
        return False