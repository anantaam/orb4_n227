from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class OrbConfig:
    mode: str
    max_positions: int
    engine_capital_inr: float
    market_protection: int
    # Timing
    or_bars: int
    entry_start_ist: str
    entry_cutoff_ist: str
    hard_exit_ist: str
    # Risk
    or_atr_lookback: int
    rvol_lookback: int
    max_effective_leverage_cap: float
    equity_mis_margin_buffer_fraction: float
    margin_probe_order_type: str
    # Order mechanics — entry
    entry_limit_poll_ms: int
    entry_limit_timeout_seconds: int
    entry_limit_ticks_from_trigger: int
    # Order mechanics — EOD exit
    exit_limit_poll_ms: int
    exit_limit_timeout_seconds: int
    exit_limit_bid_fraction: float
    exit_limit_ask_fraction: float
    limit_price_offset_fraction: float
    # Gap filter
    gap_filter_pct: float
    # Paths
    data_1m_dir: Path
    state_dir: Path
    journal_path: Path
    use_websocket: bool
    # Strategy (backtest-aligned rf=0.15)
    r_factor: float
    risk_pct: float
    atr_mult: float
    rel_vol_min: float
    min_turn: float
    min_atr: float
    top_k: int
    max_day_loss: float   # malfunction tripwire: flatten+halt if day realized P&L <= this (Rs)
    # Selectivity
    min_score: float
    # Notifications
    discord_webhook_url: str
    non_trading_shutdown_command: str


def _defaults(base_dir: Path) -> dict:
    return {
        "mode": "paper",
        "max_positions": 5,
        "engine_capital_inr": 100000,
        "market_protection": -1,
        "or_bars": 60,
        "entry_start_ist": "09:45",
        "entry_cutoff_ist": "15:19",
        "hard_exit_ist": "15:20",
        "or_atr_lookback": 14,
        "rvol_lookback": 20,
        "max_effective_leverage_cap": 5.0,
        "equity_mis_margin_buffer_fraction": 0.90,
        "margin_probe_order_type": "MARKET",
        "entry_limit_poll_ms": 200,
        "entry_limit_timeout_seconds": 3,
        "entry_limit_ticks_from_trigger": 2,
        "exit_limit_poll_ms": 200,
        "exit_limit_timeout_seconds": 2,
        "exit_limit_bid_fraction": 0.9995,
        "exit_limit_ask_fraction": 1.0005,
        "limit_price_offset_fraction": 0.0015,
        "gap_filter_pct": 0.1,
        "data_1m_dir": str(base_dir / "data"),
        "state_dir": str(base_dir / "state"),
        "journal_path": str(base_dir / "state" / "{mode}" / "trade_journal.jsonl"),
        "use_websocket": True,
        "r_factor": 0.15,
        "risk_pct": 0.0125,
        "atr_mult": 1.5,
        "rel_vol_min": 1.8,
        "min_turn": 25e7,
        "min_atr": 4.0,
        "top_k": 8,
        "max_day_loss": -25000.0,
        "min_score": 0.0,
        "discord_webhook_url": "",
        "non_trading_shutdown_command": "",
    }


def load_config(base_dir: Path | None = None) -> OrbConfig:
    base_dir = base_dir or Path(__file__).resolve().parent
    cfg_path = Path(os.getenv("ORB_CONFIG", str(base_dir / "config" / "orb_engine.yaml")))
    data = _defaults(base_dir)
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        data.update(loaded)

    mode = str(data["mode"]).lower()
    if mode not in ("paper", "live"):
        raise ValueError("mode must be paper|live")

    journal_raw = str(data["journal_path"])
    journal_path = Path(journal_raw.replace("{mode}", mode))

    return OrbConfig(
        mode=mode,
        max_positions=int(data["max_positions"]),
        engine_capital_inr=float(data["engine_capital_inr"]),
        market_protection=int(data.get("market_protection", -1)),
        or_bars=int(data["or_bars"]),
        entry_start_ist=str(data.get("entry_start_ist", "09:45")),
        entry_cutoff_ist=str(data["entry_cutoff_ist"]),
        hard_exit_ist=str(data["hard_exit_ist"]),
        or_atr_lookback=int(data["or_atr_lookback"]),
        rvol_lookback=int(data["rvol_lookback"]),
        max_effective_leverage_cap=float(data["max_effective_leverage_cap"]),
        equity_mis_margin_buffer_fraction=float(data["equity_mis_margin_buffer_fraction"]),
        margin_probe_order_type=str(data["margin_probe_order_type"]).upper(),
        entry_limit_poll_ms=int(data["entry_limit_poll_ms"]),
        entry_limit_timeout_seconds=int(data["entry_limit_timeout_seconds"]),
        entry_limit_ticks_from_trigger=int(data["entry_limit_ticks_from_trigger"]),
        exit_limit_poll_ms=int(data["exit_limit_poll_ms"]),
        exit_limit_timeout_seconds=int(data["exit_limit_timeout_seconds"]),
        exit_limit_bid_fraction=float(data["exit_limit_bid_fraction"]),
        exit_limit_ask_fraction=float(data["exit_limit_ask_fraction"]),
        limit_price_offset_fraction=float(data["limit_price_offset_fraction"]),
        gap_filter_pct=float(data["gap_filter_pct"]),
        data_1m_dir=Path(data["data_1m_dir"]),
        state_dir=Path(data["state_dir"]),
        journal_path=journal_path,
        use_websocket=bool(data.get("use_websocket", True)),
        r_factor=float(data.get("r_factor", 0.15)),
        risk_pct=float(data.get("risk_pct", 0.0125)),
        atr_mult=float(data.get("atr_mult", 1.5)),
        rel_vol_min=float(data.get("rel_vol_min", 1.8)),
        min_turn=float(data.get("min_turn", 25e7)),
        min_atr=float(data.get("min_atr", 4.0)),
        top_k=int(data.get("top_k", 8)),
        max_day_loss=float(data.get("max_day_loss", -25000.0)),
        min_score=float(data.get("min_score", 0.0)),
        discord_webhook_url=str(data.get("discord_webhook_url") or ""),
        non_trading_shutdown_command=str(data.get("non_trading_shutdown_command") or ""),
    )
