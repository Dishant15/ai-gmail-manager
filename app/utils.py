"""
app/utils.py - Shared utility functions used across the application.

Functions here are intentionally generic and stateless so they can be
imported by any service (gmail_service, rag_engine, agent, etc.)
without creating circular dependencies.
"""
from __future__ import annotations

import re

from loguru import logger
from markdownify import markdownify


# ──────────────────────────────────────────────────────────────
# HTML → Markdown / plain text conversion
# ──────────────────────────────────────────────────────────────

# Patterns to detect HTML content reliably
_HTML_TAG_RE = re.compile(r"<(html|body|div|span|p|table|td|tr|a|img|br|hr|h[1-6]|ul|ol|li|head|meta|style|script)[^>]*>", re.IGNORECASE)

# Collapse 3+ consecutive blank lines into 2
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Remove markdown image syntax — emails often have tracking pixels
_MD_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")

# Remove bare URLs left by markdownify for tracking links/pixels
_TRACKING_URL_RE = re.compile(r"\[?\s*https?://[^\s\]]{60,}\s*\]?")

# Remove repeated horizontal rules markdownify adds from <hr> tags
_EXCESS_HR_RE = re.compile(r"(\n[-*]{3,}\n){2,}")


def is_html(text: str) -> bool:
    """
    Return True if the string appears to contain HTML markup.
    Checks for common HTML tags rather than just angle brackets
    to avoid false positives on plain text with < > characters.
    """
    return bool(_HTML_TAG_RE.search(text))


def html_to_markdown(html: str) -> str:
    """
    Convert an HTML email body to clean Markdown.

    Steps:
      1. markdownify converts HTML structure to Markdown syntax
         (headings, bold, links, lists are preserved meaningfully)
      2. Tracking images and overly long URLs are stripped
         (these are noise for the LLM — pixel trackers, CDN links)
      3. Whitespace is normalised
         (excess blank lines, leading/trailing space)

    Returns clean Markdown string suitable for RAG retrieval and LLM input.
    """
    md = markdownify(
        html,
        heading_style="ATX",        # use # ## ### style headings
        bullets="-",                 # normalise list bullets to -
        strip=["script", "style", "head", "meta"],  # drop non-content tags
        convert_links=True,          # keep links as [text](url)
    )

    # Remove tracking pixel images  ![](https://...)
    md = _MD_IMAGE_RE.sub("", md)

    # Remove long bare URLs (tracking links, CDN URLs — not readable prose)
    md = _TRACKING_URL_RE.sub("", md)

    # Collapse excess horizontal rules
    md = _EXCESS_HR_RE.sub("\n---\n", md)

    # Normalise whitespace
    md = _EXCESS_BLANK_LINES_RE.sub("\n\n", md)
    md = md.strip()

    return md


def clean_email_body(raw_body: str) -> str:
    """
    Main entry point for processing an email body before it enters
    the RAG pipeline or LLM prompt.

    - If the body is HTML: convert to Markdown via html_to_markdown()
    - If the body is plain text: normalise whitespace only

    Logs which path was taken so it is visible in server output.
    """
    if not raw_body:
        return ""

    if is_html(raw_body):
        logger.debug("Email body detected as HTML — converting to Markdown.")
        cleaned = html_to_markdown(raw_body)
        logger.debug(
            "HTML->Markdown: {} chars -> {} chars.",
            len(raw_body), len(cleaned),
        )
        return cleaned

    # Plain text — just normalise whitespace
    cleaned = _EXCESS_BLANK_LINES_RE.sub("\n\n", raw_body).strip()
    return cleaned