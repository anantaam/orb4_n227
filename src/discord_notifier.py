"""
Post action/error messages to Discord via webhook. No-op if DISCORD_WEBHOOK_URL is not set.
"""
import logging
import os

import requests

MAX_CONTENT_LENGTH = 1900
_logger = logging.getLogger("e3.discord")


def get_webhook_url() -> str:
    """Read webhook on each use so .env / systemd EnvironmentFile is respected after load."""
    return os.environ.get("DISCORD_WEBHOOK_URL", "").strip()


def __getattr__(name: str):
    if name == "WEBHOOK_URL":
        return get_webhook_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Prefix for paper vs live, e.g. "[PAPER] " — set by engine at startup
_message_prefix: str = ""


def set_mode_prefix(mode: str) -> None:
    """Tag Discord messages with [PAPER] or [LIVE] for clarity."""
    global _message_prefix
    m = (mode or "paper").lower()
    _message_prefix = "[PAPER] " if m == "paper" else "[LIVE] " if m == "live" else ""


def _post(content: str, *, is_error: bool = False) -> None:
    url = get_webhook_url()
    if not url:
        return
    if _message_prefix:
        content = _message_prefix + content
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[: MAX_CONTENT_LENGTH - 3] + "..."
    try:
        payload = {"content": content}
        if is_error:
            payload["content"] = "**ERROR**\n" + payload["content"]
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code >= 400:
            _logger.warning("Discord webhook returned %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        _logger.warning("Discord webhook failed: %s", e)


def notify(message: str) -> None:
    """Send an action/status message to Discord."""
    _post(message, is_error=False)


def notify_error(message: str) -> None:
    """Send an error message to Discord."""
    _post(message, is_error=True)
