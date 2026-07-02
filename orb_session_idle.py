"""IST session gates: NSE equity trading day via vendored calendar + optional idle shutdown."""
from __future__ import annotations

import subprocess
from datetime import date, datetime
from zoneinfo import ZoneInfo

from orb_config import OrbConfig
from orb_notifier import notify
from orb_state import append_jsonl, utc_now


def ist_today_date() -> date:
    return datetime.now(ZoneInfo("Asia/Kolkata")).date()


def ist_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def is_nse_equity_session_day(d: date) -> bool:
    """
    Uses src.trading_calendar.is_trading_day: weekends off; holidays from NSE CM API
    persisted to state/nse_holidays_<year>.json. Falls back to Mon-Fri if import fails.
    """
    try:
        from src.trading_calendar import is_trading_day
        return bool(is_trading_day(d))
    except Exception:
        return d.weekday() < 5


def recent_non_session_skip_logged(cfg: OrbConfig, d: date, *, max_lines: int = 160) -> bool:
    """Avoid duplicate Discord/shutdown when premarket + session both run on same holiday."""
    p = cfg.journal_path
    if not p.is_file():
        return False
    ds = d.isoformat()
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-max_lines:]
    except OSError:
        return False
    for line in reversed(lines):
        if "non_trading_day_skip" in line and ds in line:
            return True
    return False


def handle_non_session_day(cfg: OrbConfig, phase: str, d: date | None = None) -> bool:
    """
    If d (default IST today) is not an NSE equity session: journal, Discord, optional halt.
    Returns True if caller should abort.
    """
    d = d or ist_today_date()
    if is_nse_equity_session_day(d):
        return False
    if recent_non_session_skip_logged(cfg, d):
        return True
    ds = d.isoformat()
    append_jsonl(
        cfg.journal_path,
        {
            "ts_utc": utc_now(),
            "event": "non_trading_day_skip",
            "trade_date": ds,
            "phase": phase,
            "calendar": "orb.src.trading_calendar",
        },
    )
    notify(
        cfg.discord_webhook_url,
        f"ORB SKIP [{phase}] -- {ds} IST is NOT an NSE equity session day. Engine idle.",
    )
    cmd = (cfg.non_trading_shutdown_command or "").strip()
    if cmd:
        append_jsonl(
            cfg.journal_path,
            {"ts_utc": utc_now(), "event": "non_trading_shutdown_scheduled", "phase": phase, "trade_date": ds},
        )
        subprocess.run(cmd, shell=True, check=False)
    return True
