from __future__ import annotations

from typing import Any

import requests


def _post(webhook_url: str, payload: dict[str, Any]) -> None:
    url = (webhook_url or "").strip()
    if not url:
        return
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception:
        # Notifications must never crash trading flow.
        return


def notify(webhook_url: str, message: str) -> None:
    m = (message or "").strip()
    if len(m) > 1900:
        m = m[:1897] + "..."
    _post(webhook_url, {"content": m})


def notify_error(webhook_url: str, message: str) -> None:
    m = f"ORB ERROR: {message or ''}".strip()
    if len(m) > 1900:
        m = m[:1897] + "..."
    _post(webhook_url, {"content": m})
