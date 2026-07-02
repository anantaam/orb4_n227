from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def state_file(state_dir: Path, mode: str) -> Path:
    return state_dir / mode / "engine_state.json"


def load_engine_state(state_dir: Path, mode: str) -> dict[str, Any]:
    return load_json(state_file(state_dir, mode), {"positions": [], "updated_at": None})


def save_engine_state(state_dir: Path, mode: str, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    save_json(state_file(state_dir, mode), state)


def candidates_file(state_dir: Path, mode: str, trade_date: str) -> Path:
    return state_dir / mode / f"candidates_{trade_date}.json"


def preranking_file(state_dir: Path, mode: str, trade_date: str) -> Path:
    return state_dir.parent / f"preranking_{trade_date}.json"  # ponytail: shared across tracks


# ── Position dict schema ──────────────────────────────────────────────────────
#
# {
#   "symbol": "RELIANCE",
#   "direction": "LONG",          # "LONG" or "SHORT"
#   "quantity": 10,
#   "entry_price": 2850.50,
#   "or_high": 2848.00,
#   "or_low": 2830.00,
#   "or_width": 18.00,
#   "sl_price": 2830.00,          # opposite OR boundary
#   "target_price": 2886.00,      # entry ± 2*or_width
#   "sl_order_id": "...",
#   "target_order_id": "...",
#   "software_sl_active": False,  # True if exchange SL-M was rejected
#   "entry_date": "2026-05-22",
#   "exec_type": "LIMIT",         # "LIMIT", "MARKET", or "LIMIT+MARKET"
#   "product": "MIS"
# }
