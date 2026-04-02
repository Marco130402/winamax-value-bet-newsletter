"""Send messages to Telegram via the Bot API."""

import logging

import requests

log = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_LEN = 4096


def _split_text(text: str, max_len: int = _MAX_LEN) -> list[str]:
    """Split text into chunks ≤ max_len, breaking on double newlines where possible."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def send_message(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> None:
    """Send text to a Telegram chat, splitting if it exceeds 4096 characters."""
    url = _API_BASE.format(token=token)
    chunks = _split_text(text)
    for i, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
        }
        resp = requests.post(url, json=payload, timeout=30)
        if not resp.ok:
            log.error("Telegram send failed (chunk %d/%d): %s", i, len(chunks), resp.text)
            resp.raise_for_status()
        log.info("Telegram message sent (chunk %d/%d, %d chars).", i, len(chunks), len(chunk))
