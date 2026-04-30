"""
app/gmail_service.py - Gmail OAuth2 authentication, email fetching and reply sending.
"""
from __future__ import annotations

import base64
import email as email_lib
import os
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

# Scopes required by the application
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",  # needed to mark as read
]


def get_gmail_service():
    """
    Authenticate with Gmail via OAuth2 and return a service object.
    On first run this will open a browser window for the user to authorise.
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


def _decode_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    if mime_type.startswith("multipart/"):
        for part in payload.get("parts", []):
            result = _decode_body(part)
            if result:
                return result

    # Fallback: try text/html and strip tags
    if mime_type == "text/html" and body_data:
        html = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", html).strip()

    return ""


def _parse_address(raw: str) -> tuple[str, str]:
    """Return (name, address) from a 'Name <email>' header."""
    match = re.match(r"^(.*?)\s*<(.+?)>$", raw.strip())
    if match:
        return match.group(1).strip().strip('"'), match.group(2).strip()
    return "", raw.strip()


def fetch_unread_emails(service, max_results: int = 10) -> List[EmailMessage]:
    """
    Fetch unread emails from the inbox.
    Marks each fetched email as read so it is not processed again.
    """
    messages: List[EmailMessage] = []
    try:
        result = (
            service.users()
            .messages()
            .list(
                userId="me",
                q="is:unread in:inbox",
                maxResults=max_results,
            )
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

            headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
            sender_raw = headers.get("from", "")
            sender_name, sender_addr = _parse_address(sender_raw)
            subject = headers.get("subject", "(no subject)")
            date_str = headers.get("date", "")
            body = _decode_body(msg["payload"])

            messages.append(
                EmailMessage(
                    message_id=msg["id"],
                    thread_id=msg["threadId"],
                    sender=sender_addr,
                    sender_name=sender_name or None,
                    subject=subject,
                    body=body.strip(),
                    received_at=date_str,
                )
            )

            # Mark as read
            service.users().messages().modify(
                userId="me",
                id=item["id"],
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()

    except HttpError as e:
        logger.error("Gmail API error while fetching emails: {}", e)

    return messages


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
