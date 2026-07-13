"""
ORB Session daemon (fires at 10:14 IST, runs until ~15:20 IST)
==============================================================
PHASE A  (10:14-10:15):  Compute OR for top candidates
PHASE B  (10:15-15:19):  Monitor loop -- trade on breakout
PHASE C  (15:20):        Hard exit -- cancel all SL/targets, close positions
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from orb_broker_adapter import BrokerAdapter
from orb_config import OrbConfig, load_config
from orb_costs import zerodha_intraday_cost_orb
from orb_notifier import notify, notify_error
from orb_session_idle import handle_non_session_day, ist_now, ist_today_date
# orb_sizing (margin-probe) is no longer used — sizing is risk-based (see entry loop).
from orb_state import (
    append_jsonl, candidates_file, ensure_dirs, load_engine_state,
    preranking_file, save_engine_state, save_json, utc_now,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


# ── WebSocket ticker manager ───────────────────────────────────────────────────


class TickerManager:
    """
    Background KiteTicker thread delivering live LTP via WebSocket.
    Drives tick-based trailing-stop monitoring: the trail LEVEL (sl_cur) is recomputed
    each 1-min bar by the main loop and pushed here via update_stop(); breach detection
    runs per-tick in _on_ticks and fires exit_fn immediately.
    Entry signals stay on 1-min completed bars. Degrades to bar-based SL if WS unavailable.
    """

    def __init__(self, kite):
        self.kite = kite
        self.ltp_cache: dict[int, float] = {}
        self._subscribed: set[int] = set()
        self._ticker = None
        self._started = False
        # tick-based stop registry: token -> {symbol, direction(1/-1), sl_cur, active}
        self.stops: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.exit_fn = None  # set by run(): exit_fn(token, info) -> None
        self.exited: list[str] = []  # symbols exited by tick SL, drained by main thread

    def drain_exited(self) -> set[str]:
        with self.lock:
            out = set(self.exited)
            self.exited.clear()
        return out

    # ── Stop registry (tick-based SL) ──────────────────────────────────────────

    def register_stop(self, token: int, symbol: str, direction: int, sl_cur: float) -> None:
        with self.lock:
            self.stops[token] = {"symbol": symbol, "direction": direction,
                                 "sl_cur": sl_cur, "active": True}

    def update_stop(self, token: int, sl_cur: float) -> None:
        with self.lock:
            if token in self.stops and self.stops[token]["active"]:
                self.stops[token]["sl_cur"] = sl_cur

    def unregister_stop(self, token: int) -> None:
        with self.lock:
            self.stops.pop(token, None)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        try:
            from kiteconnect import KiteTicker  # type: ignore
            self._ticker = KiteTicker(
                api_key=self.kite.api_key,
                access_token=self.kite.access_token,
            )
            self._ticker.on_ticks = self._on_ticks
            self._ticker.on_connect = self._on_connect
            self._ticker.on_error = self._on_error
            self._ticker.on_close = self._on_close
            self._ticker.connect(threaded=True)
            self._started = True
            logger.info("TickerManager: WebSocket started")
        except Exception as exc:
            logger.warning("TickerManager: failed to start WebSocket: %s -- software SL will use historical_data fallback", exc)
            self._started = False

    def subscribe(self, token: int) -> None:
        """Subscribe a token for LTP ticks (idempotent)."""
        if token in self._subscribed:
            return
        self._subscribed.add(token)
        if self._ticker and self._started:
            try:
                self._ticker.subscribe([token])
                self._ticker.set_mode(self._ticker.MODE_LTP, [token])
                logger.info("TickerManager: subscribed token %d", token)
            except Exception as exc:
                logger.warning("TickerManager: subscribe failed token=%d: %s", token, exc)

    def get_ltp(self, token: int) -> float | None:
        return self.ltp_cache.get(token)

    @property
    def is_available(self) -> bool:
        return self._started and self._ticker is not None

    def stop(self) -> None:
        if self._ticker and self._started:
            try:
                self._ticker.close()
                logger.info("TickerManager: WebSocket stopped")
            except Exception:
                pass

    # ── KiteTicker callbacks ──────────────────────────────────────────────────

    def _on_ticks(self, ws, ticks):  # noqa: ANN001
        for tick in ticks:
            token = tick.get("instrument_token")
            ltp = tick.get("last_price")
            if token is None or ltp is None:
                continue
            ltp = float(ltp)
            self.ltp_cache[token] = ltp
            # Tick-based trailing-stop breach check
            breached_info = None
            with self.lock:
                st = self.stops.get(token)
                if st and st["active"]:
                    hit = (st["direction"] == 1 and ltp <= st["sl_cur"]) or \
                          (st["direction"] == -1 and ltp >= st["sl_cur"])
                    if hit:
                        st["active"] = False
                        breached_info = dict(st, ltp=ltp, token=token)
            if breached_info and self.exit_fn:
                try:
                    self.exit_fn(token, breached_info)
                except Exception as exc:
                    logger.error("tick exit_fn failed %s: %s", breached_info.get("symbol"), exc)

    def _on_connect(self, ws, response):  # noqa: ANN001
        # Re-subscribe on every reconnect
        if self._subscribed:
            tokens = list(self._subscribed)
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_LTP, tokens)
            logger.info("TickerManager: re-subscribed %d tokens after connect", len(tokens))

    def _on_error(self, ws, code, reason):  # noqa: ANN001
        logger.warning("TickerManager: WebSocket error code=%s reason=%s", code, reason)

    def _on_close(self, ws, code, reason):  # noqa: ANN001
        logger.info("TickerManager: WebSocket closed code=%s reason=%s", code, reason)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Skip holiday gate and time checks")
    p.add_argument("--paper", action="store_true", help="Force paper mode")
    return p.parse_args()


# ── Time helpers ──────────────────────────────────────────────────────────────


def _ist_hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _next_minute_boundary() -> float:
    """Return seconds to sleep until the start of the next IST minute + 5s buffer."""
    now = ist_now()
    secs_into_minute = now.second + now.microsecond / 1e6
    return max(0.0, 60.0 - secs_into_minute + 5.0)


def _past_time(hhmm: str) -> bool:
    """True if IST clock is past HH:MM."""
    now = ist_now()
    h, m = map(int, hhmm.split(":"))
    threshold = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return now >= threshold


# ── OR computation ────────────────────────────────────────────────────────────


def fetch_or_bars(
    broker: BrokerAdapter,
    symbol: str,
    trade_date: date,
    or_bars: int,
    instrument_token: int | None = None,
) -> pd.DataFrame | None:
    """Fetch 1m bars for the OR window via kite.historical_data.

    If `instrument_token` is supplied (pre-resolved, e.g. cached from Phase A token
    lookup) it is used directly, avoiding a second kite.ltp() round-trip.
    """
    try:
        token = instrument_token if instrument_token is not None else _get_instrument_token(broker, symbol)
        # Fetch 09:15–09:36 so the ATR(14) rolling has enough 1-min bars to be valid at
        # the 09:35 bar. OR high/low/atr_orb are computed on the 09:20–09:35 slice downstream.
        from_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 15, tzinfo=IST)
        to_dt   = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 36, tzinfo=IST)
        bars = broker.kite.historical_data(
            instrument_token=token,
            from_date=from_dt,
            to_date=to_dt,
            interval="minute",
        )
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception as exc:
        logger.warning("OR fetch failed %s: %s", symbol, exc)
        return None


def _get_instrument_token(broker: BrokerAdapter, symbol: str) -> int:
    """Resolve NSE instrument token. Uses kite.ltp as a lightweight lookup."""
    q = broker.kite.ltp([f"NSE:{symbol}"])
    return q[f"NSE:{symbol}"]["instrument_token"]


ORB_START = dtime(9, 20)
ORB_END   = dtime(9, 35)
E_BPS = 5.0   # entry slippage
SL_BPS = 3.0  # stop exit slippage
T_BPS = 2.0   # time exit slippage


def slip_price(price: float, direction: int, side: str, bps: float) -> float:
    """Backtest-identical slippage. direction: 1=LONG, -1=SHORT."""
    s = bps / 10_000.0
    if side == "entry":
        return price * (1.0 + s) if direction == 1 else price * (1.0 - s)
    return price * (1.0 - s) if direction == 1 else price * (1.0 + s)


def compute_orb_metrics(df: pd.DataFrame) -> dict | None:
    """From 09:15–09:36 1-min bars: OR on 09:20–09:35 slice + atr_orb (ATR-14 on 1-min TR
    at the 09:35 bar). Mirrors backtest db_loader._sym_daily_metrics."""
    if df is None or df.empty:
        return None
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d = d.sort_index()
    # 1-min true range + rolling ATR(14) over the fetched window
    prev_c = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_c).abs(),
        (d["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    t = d.index.time
    orb = d[(t >= ORB_START) & (t <= ORB_END)]
    if orb.empty:
        return None
    atr_orb = float(orb["atr"].iloc[-1]) if not pd.isna(orb["atr"].iloc[-1]) else None
    if atr_orb is None or atr_orb <= 0:
        return None
    return {
        "orb_open":  float(orb["open"].iloc[0]),
        "orb_high":  float(orb["high"].max()),
        "orb_low":   float(orb["low"].min()),
        "orb_close": float(orb["close"].iloc[-1]),
        "orb_vol":   float(orb["volume"].sum()),
        "orb_turn":  float((orb["close"] * orb["volume"]).sum()),
        "atr_orb":   atr_orb,
    }


def compute_gap_pct(df: pd.DataFrame, prev_close: float) -> float:
    """Gap % = (first bar open - prev_close) / prev_close * 100."""
    if prev_close <= 0 or df.empty:
        return 0.0
    first_open = float(df["open"].iloc[0])
    return (first_open - prev_close) / prev_close * 100.0


def _get_prev_close(data_dir: Path, symbol: str, trade_date: date) -> float | None:
    """Read prev_close from local 1m CSV (filename: SYMBOL_1min_kite.csv)."""
    csv_path = data_dir / f"{symbol}_1min_kite.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        prior = df[df.index.date < trade_date]
        if prior.empty:
            return None
        last_day = prior.index.date[-1]
        day_bars = prior[prior.index.date == last_day]
        return float(day_bars["close"].iloc[-1]) if not day_bars.empty else None
    except Exception:
        return None


# ── Ranking ───────────────────────────────────────────────────────────────────


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Backtest ranking: sort by score = rel_vol * atr_pct, descending (highest = best)."""
    return sorted(candidates, key=lambda c: -c.get("score", 0.0))


# ── Monitoring loop helpers ───────────────────────────────────────────────────


def fetch_latest_close(broker: BrokerAdapter, symbol: str, trade_date: date) -> float | None:
    """Fetch the most recent completed 1m bar's close via historical_data."""
    try:
        now = ist_now()
        # Fetch last 2 minutes to ensure we get a completed bar
        from_dt = now - timedelta(minutes=2)
        bars = broker.kite.historical_data(
            instrument_token=_get_instrument_token(broker, symbol),
            from_date=from_dt,
            to_date=now,
            interval="minute",
        )
        if not bars:
            return None
        # Last completed bar (not the current forming one)
        completed = [b for b in bars if pd.to_datetime(b["date"]).replace(tzinfo=IST) < now.replace(second=0, microsecond=0)]
        if not completed:
            return None
        return float(completed[-1]["close"])
    except Exception as exc:
        logger.warning("LTP fetch failed %s: %s", symbol, exc)
        return None


def fetch_latest_bar(broker: BrokerAdapter, symbol: str, trade_date: date,
                     instrument_token: int | None = None) -> dict | None:
    """Most recent completed 1m bar as {high, low, close}. Used for intrabar entry
    trigger and per-bar trail updates (matches backtest bar-granular logic)."""
    try:
        now = ist_now()
        tok = instrument_token if instrument_token is not None else _get_instrument_token(broker, symbol)
        bars = broker.kite.historical_data(
            instrument_token=tok,
            from_date=now - timedelta(minutes=3),
            to_date=now,
            interval="minute",
        )
        if not bars:
            return None
        cutoff = now.replace(second=0, microsecond=0)
        completed = [b for b in bars if pd.to_datetime(b["date"]).replace(tzinfo=IST) < cutoff]
        if not completed:
            return None
        b = completed[-1]
        return {"high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"])}
    except Exception as exc:
        logger.warning("bar fetch failed %s: %s", symbol, exc)
        return None


def _day_realized_pnl(journal_path: Path, today_str: str) -> float:
    """Sum today's realized net P&L from the journal (for the malfunction tripwire)."""
    if not journal_path.exists():
        return 0.0
    total = 0.0
    try:
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("event") in ("exit_execution", "software_sl_exit") and r.get("ts_utc", "")[:10] == today_str:
                total += r.get("net_pnl", 0) or 0
    except Exception:
        return 0.0
    return total


def _do_stop_exit(broker: BrokerAdapter, cfg: OrbConfig, pos: dict, sl_cur: float,
                  exit_label: str = "trail_sl") -> float:
    """Place (or simulate) a trailing-stop exit and journal it. Does NOT mutate the
    state file — the main thread reaps closed symbols. Safe to call from the WS thread.
    Returns net P&L."""
    sym = pos["symbol"]
    direction = pos.get("dir_i", 1 if pos["direction"] == "LONG" else -1)
    qty = int(pos["quantity"])
    entry_p = float(pos["entry_price"])
    if cfg.mode == "paper":
        exit_p = round(slip_price(sl_cur, direction, "exit", SL_BPS), 2)
    else:
        try:
            fill = broker.place_exit_order(
                symbol=sym, side=("SELL" if direction == 1 else "BUY"),
                quantity=qty, sl_order_id=pos.get("sl_order_id", ""),
            )
            exit_p = float(fill.average_price or sl_cur)
        except Exception as exc:
            logger.error("stop exit order failed %s: %s", sym, exc)
            exit_p = sl_cur
    gross = (exit_p - entry_p) * qty * direction
    net = gross - zerodha_intraday_cost_orb(entry_p * qty, exit_p * qty, pos["direction"])
    append_jsonl(cfg.journal_path, {
        "ts_utc": utc_now(), "event": "exit_execution", "symbol": sym,
        "direction": pos["direction"], "quantity": qty, "entry_price": entry_p,
        "exit_price": round(exit_p, 2), "gross_pnl": round(gross, 2),
        "net_pnl": round(net, 2), "exit_type": "trailing_sl", "exit_label": exit_label,
    })
    notify(cfg.discord_webhook_url,
           f"ORB TRAIL SL [{cfg.mode}] {pos['direction']} {sym} exit @ {exit_p:.2f} net={net:+.0f}")
    logger.info("TRAIL SL EXIT %s %s @ %.2f net=%.2f", pos["direction"], sym, exit_p, net)
    return net


def _infer_exit_type(
    broker: BrokerAdapter,
    sl_oid: str,
    tgt_oid: str,
    sl_price: float,
    target_price: float,
) -> tuple[float | None, str]:
    """
    Returns (exit_price, exit_label) by checking order histories.
    exit_label is one of: "SL", "TARGET", "UNKNOWN"
    """
    # Check SL order first
    for oid, label in [(sl_oid, "SL"), (tgt_oid, "TARGET")]:
        if not oid:
            continue
        try:
            for h in reversed(broker.get_order_history(oid)):
                if h.get("status") == "COMPLETE":
                    px = float(h.get("average_price") or 0) or None
                    if px:
                        return px, label
        except Exception:
            pass
    return None, "UNKNOWN"


def reconcile_positions(broker: BrokerAdapter, state: dict, cfg: OrbConfig, journal_path: Path) -> dict:
    """
    Compare engine_state positions vs broker positions.
    If a position disappeared from broker (SL or target hit), notify Discord + record event.

    Paper mode: broker.get_positions() always returns [] (no real MIS orders placed),
    so broker comparison is meaningless and is skipped entirely. Software SL in the
    main loop is the only intraday protection in paper mode.
    """
    if broker.paper:
        return state  # Paper mode: cannot reconcile against real broker -- skip

    broker_pos = broker.get_positions()
    broker_symbols = {p["tradingsymbol"]: p for p in broker_pos if p.get("product") == "MIS"}

    remaining = []
    for pos in state.get("positions", []):
        sym = pos["symbol"]
        if sym not in broker_symbols or abs(int(broker_symbols[sym].get("quantity", 0))) == 0:
            # Position gone -- SL or target was hit intraday
            logger.info("RECONCILE: %s disappeared from broker -- SL/target hit", sym)

            direction = pos.get("direction", "LONG")
            entry_p = float(pos.get("entry_price") or 0)
            qty = int(pos.get("quantity") or 0)
            sl_price = float(pos.get("sl_price") or 0)
            target_price = float(pos.get("target_price") or 0)

            exit_price, exit_label = _infer_exit_type(
                broker,
                pos.get("sl_order_id", ""),
                pos.get("target_order_id", ""),
                sl_price,
                target_price,
            )

            gross_pnl = ((exit_price or entry_p) - entry_p) * qty * (1 if direction == "LONG" else -1)
            net_pnl = gross_pnl - zerodha_intraday_cost_orb(
                entry_p * qty, (exit_price or entry_p) * qty, direction
            )

            pnl_sign = "+" if net_pnl >= 0 else ""
            emoji = "🟢" if net_pnl >= 0 else "🔴"
            label_str = "TARGET HIT" if exit_label == "TARGET" else ("SL HIT" if exit_label == "SL" else "CLOSED")
            exit_str = f"{exit_price:.2f}" if exit_price is not None else "?"  # Bug 1 fix: pre-build string
            notify(
                cfg.discord_webhook_url,
                f"{emoji} ORB {label_str} [{cfg.mode}]  {sym} {direction}"
                f"  qty={qty}  entry={entry_p:.2f}  exit={exit_str}"
                f"  net PnL: Rs {pnl_sign}{net_pnl:.0f}",
            )
            append_jsonl(journal_path, {
                "ts_utc": utc_now(),
                "event": "sl_or_target_exit",
                "symbol": sym,
                "direction": direction,
                "quantity": qty,
                "entry_price": entry_p,
                "exit_price": exit_price,
                "exit_label": exit_label,
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
            })
        else:
            remaining.append(pos)

    state["positions"] = remaining
    return state


# ── Phase C: Hard exit ────────────────────────────────────────────────────────


def hard_exit_all(broker: BrokerAdapter, state: dict, cfg: OrbConfig, journal_path: Path) -> None:
    """Cancel SL/target orders and exit all open positions. Sends per-trade and summary Discord messages."""
    broker_pos = broker.get_positions()
    broker_map = {p["tradingsymbol"]: p for p in broker_pos if p.get("product") == "MIS"}

    total_net_pnl = 0.0
    trade_lines: list[str] = []  # Per-trade lines for EOD summary

    for pos in state.get("positions", []):
        sym = pos["symbol"]
        direction = pos.get("direction", "LONG")
        qty = int(pos.get("quantity") or 0)
        entry_price = float(pos.get("entry_price") or 0)
        sl_oid = pos.get("sl_order_id", "")
        tgt_oid = pos.get("target_order_id", "")
        sl_price = float(pos.get("sl_price") or 0)
        target_price_val = float(pos.get("target_price") or 0)

        # Cancel target order (SL-M is handled inside place_exit_order via _ensure_sl_order_cleared)
        if tgt_oid:
            try:
                broker.cancel_order(tgt_oid)
                logger.info("Cancelled target order %s for %s", tgt_oid, sym)
            except Exception as exc:
                logger.info("Cancel target %s: %s (may be already done)", tgt_oid, exc)

        # Refresh qty from broker
        broker_qty = abs(int((broker_map.get(sym) or {}).get("quantity", 0)))

        if broker_qty == 0:
            # Already exited during the session (SL-M fired or target hit)
            logger.info("%s: broker qty=0, already exited intraday", sym)
            exit_price, exit_label = _infer_exit_type(
                broker, sl_oid, tgt_oid, sl_price, target_price_val
            )
            gross = ((exit_price or entry_price) - entry_price) * qty * (1 if direction == "LONG" else -1)
            net = gross - zerodha_intraday_cost_orb(
                entry_price * qty, (exit_price or entry_price) * qty, direction
            )
            total_net_pnl += net

            pnl_sign = "+" if net >= 0 else ""
            emoji = "🟢" if net >= 0 else "🔴"
            label_str = (
                "TARGET HIT" if exit_label == "TARGET"
                else ("SL HIT" if exit_label == "SL" else "CLOSED")
            )
            exit_str = f"{exit_price:.2f}" if exit_price else "?"
            notify(
                cfg.discord_webhook_url,
                f"{emoji} ORB {label_str} [{cfg.mode}]  {sym} {direction}"
                f"  qty={qty}  entry={entry_price:.2f}  exit={exit_str}"
                f"  net PnL: Rs {pnl_sign}{net:.0f}",
            )
            trade_lines.append(
                f"  {sym:12s} {direction:5s} {label_str:11s}"
                f"  entry={entry_price:.2f}  exit={exit_str}"
                f"  net=Rs {pnl_sign}{net:.0f}"
            )
            append_jsonl(journal_path, {
                "ts_utc": utc_now(), "event": "exit_execution", "symbol": sym,
                "direction": direction, "quantity": qty, "entry_price": entry_price,
                "exit_price": exit_price, "gross_pnl": round(gross, 2), "net_pnl": round(net, 2),
                "exit_type": "intraday_sl_or_target", "exit_label": exit_label,
            })
            continue

        # Active position -- exit via LIMIT with MARKET fallback
        exit_side = "SELL" if direction == "LONG" else "BUY"
        logger.info("HARD EXIT %s %s qty=%d", exit_side, sym, broker_qty)
        try:
            fill = broker.place_exit_order(
                symbol=sym,
                side=exit_side,
                quantity=broker_qty,
                sl_order_id=sl_oid,
            )
            exit_price = fill.average_price
        except Exception as exc:
            logger.error("Exit order failed for %s: %s", sym, exc)
            notify_error(cfg.discord_webhook_url, f"Exit order FAILED for {sym}: {exc}")
            exit_price = entry_price  # worst-case fallback for PnL accounting

        gross = ((exit_price or entry_price) - entry_price) * qty * (1 if direction == "LONG" else -1)
        net = gross - zerodha_intraday_cost_orb(
            entry_price * qty, (exit_price or entry_price) * qty, direction
        )
        total_net_pnl += net

        pnl_sign = "+" if net >= 0 else ""
        emoji = "🟢" if net >= 0 else "🔴"
        exit_str = f"{exit_price:.2f}" if exit_price else "?"
        notify(
            cfg.discord_webhook_url,
            f"{emoji} ORB EOD CLOSE [{cfg.mode}]  {sym} {direction}"
            f"  qty={qty}  entry={entry_price:.2f}  exit={exit_str}"
            f"  net PnL: Rs {pnl_sign}{net:.0f}",
        )
        trade_lines.append(
            f"  {sym:12s} {direction:5s} EOD CLOSE  "
            f"  entry={entry_price:.2f}  exit={exit_str}"
            f"  net=Rs {pnl_sign}{net:.0f}"
        )
        append_jsonl(journal_path, {
            "ts_utc": utc_now(), "event": "exit_execution", "symbol": sym,
            "direction": direction, "quantity": qty, "entry_price": entry_price,
            "exit_price": exit_price, "gross_pnl": round(gross, 2), "net_pnl": round(net, 2),
            "exit_type": "hard_exit_eod",
        })
        logger.info("%s EXIT done: gross=%.2f net=%.2f", sym, gross, net)

    state["positions"] = []

    # ── EOD summary ──────────────────────────────────────────────────────────
    n_trades = len(trade_lines)
    total_sign = "+" if total_net_pnl >= 0 else ""
    total_emoji = "🟢" if total_net_pnl >= 0 else "🔴"
    summary_body = "\n".join(trade_lines) if trade_lines else "  (no trades today)"
    trade_word = "trade" if n_trades == 1 else "trades"
    notify(
        cfg.discord_webhook_url,
        f"{total_emoji} ORB EOD SUMMARY [{cfg.mode}] {ist_now().strftime('%Y-%m-%d')}\n"
        f"{summary_body}\n"
        f"{'─' * 48}\n"
        f"Total net PnL: Rs {total_sign}{total_net_pnl:.0f}  ({n_trades} {trade_word})",
    )
    logger.info("Hard exit complete. net_pnl=%.2f", total_net_pnl)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = load_config(BASE_DIR)
    if args.paper:
        import dataclasses
        cfg = dataclasses.replace(cfg, mode="paper")

    ensure_dirs(cfg.data_1m_dir, cfg.state_dir / cfg.mode)

    trade_date = ist_today_date()
    today_str = trade_date.isoformat()

    if not args.force:
        if handle_non_session_day(cfg, phase="session"):
            logger.info("Non-trading day -- exiting.")
            sys.exit(0)

    logger.info("ORB SESSION start: %s mode=%s", today_str, cfg.mode)

    # ── Purge stale positions from previous day ──────────────────────────────
    state = load_engine_state(cfg.state_dir, cfg.mode)
    stale = [p for p in state.get("positions", []) if p.get("entry_date", "") < today_str]
    if stale:
        logger.warning("Purging %d stale positions from previous day", len(stale))
        state["positions"] = [p for p in state["positions"] if p.get("entry_date", "") >= today_str]
        save_engine_state(cfg.state_dir, cfg.mode, state)

    # ── Load preranking ───────────────────────────────────────────────────────
    pr_path = preranking_file(cfg.state_dir, cfg.mode, today_str)
    if not pr_path.exists():
        logger.error("Preranking file not found: %s -- run premarket.py first", pr_path)
        notify_error(cfg.discord_webhook_url, f"No preranking for {today_str} -- premarket.py may have failed.")
        sys.exit(1)

    import json
    preranking = json.loads(pr_path.read_text(encoding="utf-8"))
    all_stocks = preranking.get("stocks", [])
    # Backtest ranks ALL symbols by score, then takes top_k. Prefilter loosely by prior-day
    # turnover (turn3) to trim clearly-illiquid names and bound the OR-fetch API load;
    # the exact turn_B>=min_turn filter is applied after today's OR turnover is known.
    universe = [s for s in all_stocks
                if (s.get("turn3") or 0) >= cfg.min_turn * 0.5
                and (s.get("or_vol_baseline") or 0) > 0]
    universe.sort(key=lambda x: -float(x.get("turn3") or 0))
    logger.info("Loaded %d stocks from preranking, scoring universe=%d", len(all_stocks), len(universe))

    # ── Kite login ────────────────────────────────────────────────────────────
    broker = BrokerAdapter(paper=(cfg.mode == "paper"), cfg=cfg)
    try:
        broker.login()
        logger.info("Kite login OK")
    except Exception as exc:
        notify_error(cfg.discord_webhook_url, f"Session Kite login failed: {exc}")
        sys.exit(1)

    # ── Instrument token cache (populated in Phase A, used for WebSocket subscriptions) ──
    token_cache: dict[str, int] = {}

    # ── WebSocket ticker: tick-based trailing-stop monitoring ────────────────
    ticker = TickerManager(broker.kite)

    def _tick_exit(token: int, info: dict) -> None:
        """Runs on the WS thread when a tick breaches sl_cur. Places the exit and
        queues the symbol; the main loop reaps it from state."""
        pos = {
            "symbol": info["symbol"], "direction": ("LONG" if info["direction"] == 1 else "SHORT"),
            "dir_i": info["direction"], "quantity": info.get("quantity", 0),
            "entry_price": info.get("entry_price", 0.0), "sl_order_id": info.get("sl_order_id", ""),
        }
        # Pull authoritative position fields from state for accurate qty/entry
        st = load_engine_state(cfg.state_dir, cfg.mode)
        for p in st.get("positions", []):
            if p["symbol"] == info["symbol"]:
                pos.update({"quantity": p["quantity"], "entry_price": p["entry_price"],
                            "sl_order_id": p.get("sl_order_id", "")})
                break
        _do_stop_exit(broker, cfg, pos, info["sl_cur"], exit_label="tick_sl")
        ticker.unregister_stop(token)
        with ticker.lock:
            ticker.exited.append(info["symbol"])

    ticker.exit_fn = _tick_exit
    if cfg.use_websocket:
        ticker.start()

    # ════════════════════════════════════════════════════════════════════════
    # PHASE A: OR Computation (10:14-10:15)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("PHASE A: Computing ORB metrics for %d stocks", len(universe))

    candidates = []
    for stock in universe:
        symbol = stock["symbol"]
        or_vol_baseline = float(stock.get("or_vol_baseline") or 0)
        turn3 = float(stock.get("turn3") or 0)
        if or_vol_baseline <= 0:
            continue

        # ── Resolve instrument token (cached to avoid double API call) ────────
        try:
            tok = _get_instrument_token(broker, symbol)
            token_cache[symbol] = tok
        except Exception as _te:
            logger.warning("Token lookup failed %s: %s", symbol, _te)
            tok = None

        df_or = fetch_or_bars(broker, symbol, trade_date, cfg.or_bars, instrument_token=tok)
        m = compute_orb_metrics(df_or)
        if m is None:
            logger.debug("No ORB metrics for %s", symbol)
            continue

        or_high, or_low = m["orb_high"], m["orb_low"]
        or_width = or_high - or_low
        atr_orb = m["atr_orb"]
        if or_width <= 0:
            continue

        # direction = sign(orb_close - orb_open); single direction per stock (backtest)
        direction = 1 if m["orb_close"] > m["orb_open"] else (-1 if m["orb_close"] < m["orb_open"] else 0)
        if direction == 0:
            continue

        rel_vol = m["orb_vol"] / or_vol_baseline if or_vol_baseline > 0 else 0.0
        turn_B  = turn3 + m["orb_turn"]
        atr_pct = atr_orb / m["orb_close"] * 100 if m["orb_close"] > 0 else 0.0
        score   = round(rel_vol * atr_pct, 4)

        # ── Backtest filters: atr_orb>=min_atr, rel_vol>=rel_vol_min, turn_B>=min_turn ──
        if atr_orb < cfg.min_atr:      continue
        if rel_vol < cfg.rel_vol_min:  continue
        if turn_B  < cfg.min_turn:     continue

        candidates.append({
            "symbol": symbol,
            "direction": direction,          # 1=LONG, -1=SHORT
            "rel_vol": round(rel_vol, 4),
            "or_high": or_high,
            "or_low": or_low,
            "or_width": or_width,
            "orb_close": m["orb_close"],
            "atr_orb": atr_orb,
            "turn_B": round(turn_B, 2),
            "traded": False,
            "score": score,
        })

    # Rank by score desc; apply score gate (score20 track); take top_k
    candidates = rank_candidates(candidates)
    if cfg.min_score > 0:
        candidates = [c for c in candidates if c.get("score", 0) >= cfg.min_score]
    top_n = candidates[: cfg.top_k]

    # Save candidates
    cand_path = candidates_file(cfg.state_dir, cfg.mode, today_str)
    save_json(cand_path, {"date": today_str, "candidates": top_n})

    logger.info("Top %d candidates: %s", len(top_n), [c["symbol"] for c in top_n])
    notify(
        cfg.discord_webhook_url,
        f"ORB SESSION STARTING [{cfg.mode}] {today_str} -- {len(top_n)} candidates selected, OR computed.\n"
        + "\n".join(f"  {c['symbol']:12s} {'LONG' if c['direction']==1 else 'SHORT'} "
                    f"OR={c['or_high']:.2f}/{c['or_low']:.2f} score={c['score']:.2f}" for c in top_n),
    )

    if not top_n:
        logger.warning("No candidates after OR computation -- exiting.")
        sys.exit(0)

    # ── Subscribe WebSocket tokens for already-open positions ─────────────────
    # Needed on crash+restart: positions that were entered before the crash must be
    # subscribed so software SL monitoring works immediately from Phase B.
    _open_state = load_engine_state(cfg.state_dir, cfg.mode)
    for _pos in _open_state.get("positions", []):
        _sym = _pos["symbol"]
        if _sym not in token_cache:
            try:
                _tok = _get_instrument_token(broker, _sym)
                token_cache[_sym] = _tok
            except Exception as _te:
                logger.warning("Token lookup (existing pos) failed %s: %s", _sym, _te)
        _tok = token_cache.get(_sym)
        if _tok:
            ticker.subscribe(_tok)
            if cfg.mode == "paper":
                _dir = _pos.get("dir_i", 1 if _pos["direction"] == "LONG" else -1)
                ticker.register_stop(_tok, _sym, _dir, float(_pos["sl_price"]))
    del _open_state  # use fresh load() inside the loop

    # Sizing is risk-based and computed at entry (qty = risk_budget / stop_distance),
    # which needs no live LTP — so there is no Phase-A pre-sizing step anymore.

    # ════════════════════════════════════════════════════════════════════════
    # PHASE B: Monitoring loop (10:15 - 15:19)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("PHASE B: Monitoring loop started")
    reconcile_counter = 0

    while not _past_time(cfg.hard_exit_ist):
        # Sleep to next minute boundary
        sleep_s = _next_minute_boundary()
        logger.info("Sleeping %.1fs to next bar...", sleep_s)
        time.sleep(sleep_s)

        now_ist = ist_now()
        if _past_time(cfg.hard_exit_ist):
            break

        # ── Kill switch + malfunction tripwire ───────────────────────────────
        # These are operational backstops, NOT a strategy rule: the kill file lets
        # you flatten on demand; the tripwire only fires on a loss far outside normal
        # range (a bug/runaway can't be reached by normal trading), so it never alters
        # strategy behaviour. Either one breaks to Phase C, which flattens immediately.
        if (BASE_DIR / "KILL").exists():
            logger.error("KILL file present -- flattening all positions and exiting")
            notify_error(cfg.discord_webhook_url, f"[{cfg.mode}] KILL switch tripped -- flattening")
            break
        day_pnl_so_far = _day_realized_pnl(cfg.journal_path, today_str)
        if day_pnl_so_far <= cfg.max_day_loss:
            logger.error("MALFUNCTION TRIPWIRE: day realized %.0f <= limit %.0f -- flattening & exit",
                         day_pnl_so_far, cfg.max_day_loss)
            notify_error(cfg.discord_webhook_url,
                         f"[{cfg.mode}] MALFUNCTION TRIPWIRE day P&L Rs {day_pnl_so_far:.0f} <= Rs {cfg.max_day_loss:.0f} -- flattening")
            break

        # Entry window: backtest enters only in [entry_start, entry_cutoff].
        entry_open = _past_time(cfg.entry_start_ist) and not _past_time(cfg.entry_cutoff_ist)

        # Check: all slots filled?
        state = load_engine_state(cfg.state_dir, cfg.mode)
        n_open = len(state.get("positions", []))
        if n_open >= cfg.max_positions or not entry_open:
            if not entry_open:
                logger.info("Outside entry window (%s-%s) -- trailing only",
                            cfg.entry_start_ist, cfg.entry_cutoff_ist)
        else:
            # Build set of symbols already open (survives crash+restart)
            open_symbols = {p["symbol"] for p in state.get("positions", [])}

            # Monitor untraded candidates
            for cand in top_n:
                # Enforce position cap WITHIN the bar: state["positions"] grows as we
                # enter, so re-check each iteration (else multiple candidates triggering
                # on the same bar could all fill and exceed max_positions).
                if len(state.get("positions", [])) >= cfg.max_positions:
                    break
                if cand.get("traded"):
                    continue
                sym = cand["symbol"]

                # Bug 3 fix: skip if this symbol already has an open position
                # (prevents re-entry after crash+restart resets in-memory traded flag)
                if sym in open_symbols:
                    cand["traded"] = True
                    continue

                # (a) Re-resolve the instrument token if Phase A's lookup failed
                # (transient kite.ltp blip). Without a token the position can't be
                # WebSocket-subscribed and its tick-SL would never fire.
                tok = token_cache.get(sym)
                if tok is None:
                    try:
                        tok = _get_instrument_token(broker, sym)
                        token_cache[sym] = tok
                    except Exception as _te:
                        logger.warning("Token re-resolve failed %s: %s", sym, _te)
                        tok = None
                bar = fetch_latest_bar(broker, sym, trade_date, instrument_token=tok)
                if bar is None:
                    continue

                direction = cand["direction"]           # 1=LONG, -1=SHORT
                or_high = cand["or_high"]
                or_low = cand["or_low"]
                or_width = cand["or_width"]
                atr_orb = cand["atr_orb"]

                # Intrabar breakout trigger (backtest: high>=or_high / low<=or_low)
                triggered = (direction == 1 and bar["high"] >= or_high) or \
                            (direction == -1 and bar["low"] <= or_low)
                if not triggered:
                    continue

                dir_str = "LONG" if direction == 1 else "SHORT"
                entry_level = or_high if direction == 1 else or_low
                logger.info("%s SIGNAL: %s bar_h/l=%.2f/%.2f vs OR=%.2f",
                            dir_str, sym, bar["high"], bar["low"], entry_level)

                # ── Risk-based sizing (backtest-aligned) ─────────────────────────
                # qty = risk_budget / stop_distance, so every trade risks the same rupee
                # amount (risk_pct * capital). Needs no live LTP — depends only on the OR
                # width, which is known from Phase A. Mirrors sensitivity_backtest.py.
                r_val = or_width * cfg.r_factor
                if r_val <= 0:
                    continue
                ep_est = round(slip_price(entry_level, direction, "entry", E_BPS), 2)
                qty = int((cfg.risk_pct * cfg.engine_capital_inr) / r_val)
                if qty < 1:
                    logger.info("risk-sizing qty<1 for %s (r_val=%.2f) -- skip", sym, r_val)
                    continue
                # Portfolio leverage cap: skip if total MIS margin (notional / leverage)
                # across open positions + this one would exceed capital*1.05 (backtest rule).
                lev = cfg.max_effective_leverage_cap
                margin_used = sum(float(p["entry_price"]) * int(p["quantity"]) / lev
                                  for p in state.get("positions", []))
                if margin_used + ep_est * qty / lev > cfg.engine_capital_inr * 1.05:
                    logger.info("Portfolio margin cap reached -- skip %s (would be %.0f > %.0f)",
                                sym, margin_used + ep_est * qty / lev, cfg.engine_capital_inr * 1.05)
                    continue

                cand["traded"] = True

                try:
                    fill = broker.place_entry_order(
                        symbol=sym, side=("BUY" if direction == 1 else "SELL"),
                        quantity=qty, trigger_price=entry_level,
                    )
                    if fill.filled_quantity < 1:
                        logger.warning("%s entry zero fill for %s", dir_str, sym)
                        cand["traded"] = False
                        continue

                    # Paper: backtest-identical fill = OR level + fixed slippage.
                    # Live: use the broker's realistic orderbook fill.
                    if cfg.mode == "paper":
                        entry_price = round(slip_price(entry_level, direction, "entry", E_BPS), 2)
                    else:
                        entry_price = fill.average_price or entry_level
                    filled_qty = fill.filled_quantity

                    # Initial SL = OR boundary ± r_val buffer (backtest)
                    r_val = or_width * cfg.r_factor
                    sl_price = round((or_low - r_val) if direction == 1 else (or_high + r_val), 2)

                    software_sl = False
                    sl_oid = ""
                    if cfg.mode != "paper":
                        try:
                            sl_oid = broker.place_sl_order(
                                symbol=sym, side=("SELL" if direction == 1 else "BUY"),
                                quantity=filled_qty, trigger_price=sl_price,
                            )
                        except Exception as exc:
                            logger.error("SL-M placement failed for %s: %s -- software SL active", sym, exc)
                            software_sl = True

                    pos = {
                        "symbol": sym, "direction": dir_str, "dir_i": direction,
                        "quantity": filled_qty, "entry_price": entry_price,
                        "or_high": or_high, "or_low": or_low, "or_width": or_width,
                        "atr_orb": atr_orb, "sl_price": sl_price,
                        "peak": entry_price,          # trail peak seed
                        "sl_order_id": sl_oid, "software_sl_active": software_sl,
                        "entry_date": today_str, "exec_type": fill.exec_type,
                        "product": "MIS", "token": tok,
                    }
                    state["positions"].append(pos)
                    save_engine_state(cfg.state_dir, cfg.mode, state)
                    append_jsonl(cfg.journal_path, {
                        "ts_utc": utc_now(), "event": "entry_execution", **pos
                    })
                    # Subscribe; register tick-based stop ONLY in paper mode.
                    # (Live mode's exchange SL-M is the executor — see modify_sl_order.)
                    if tok:
                        ticker.subscribe(tok)
                        if cfg.mode == "paper":
                            ticker.register_stop(tok, sym, direction, sl_price)
                    notify(cfg.discord_webhook_url,
                           f"ORB ENTRY [{cfg.mode}] {dir_str} {sym} qty={filled_qty} @ {entry_price:.2f} "
                           f"[{fill.exec_type}] SL={sl_price:.2f} (ATR trail)")
                    logger.info("%s ENTRY %s qty=%d @ %.2f SL=%.2f", dir_str, sym, filled_qty, entry_price, sl_price)

                except Exception as exc:
                    logger.error("%s entry error for %s: %s", dir_str, sym, exc)
                    notify_error(cfg.discord_webhook_url, f"{dir_str} entry error {sym}: {exc}")
                    cand["traded"] = False

        # ── Reap tick-based stop exits (WS thread placed the order + queued symbol) ──
        exited_syms = ticker.drain_exited()
        if exited_syms:
            state = load_engine_state(cfg.state_dir, cfg.mode)
            state["positions"] = [p for p in state["positions"] if p["symbol"] not in exited_syms]
            save_engine_state(cfg.state_dir, cfg.mode, state)
            logger.info("Tick SL: removed %s from state", exited_syms)

        # ── ATR trailing-stop update (per 1-min bar, matches backtest schedule) ──
        # peak & sl_cur recomputed each bar; new trigger pushed to the tick monitor
        # (exchange SL-M modify in live). Breach detection is tick-driven in
        # TickerManager; a bar-based fallback runs here only if WS is unavailable.
        state = load_engine_state(cfg.state_dir, cfg.mode)
        bar_exited: set[str] = set()
        dirty = False
        for pos in state.get("positions", []):
            sym = pos["symbol"]
            tok = pos.get("token") or token_cache.get(sym)
            direction = pos.get("dir_i", 1 if pos["direction"] == "LONG" else -1)
            atr_orb = float(pos.get("atr_orb") or 0)
            bar = fetch_latest_bar(broker, sym, trade_date, instrument_token=tok)
            if bar is None or atr_orb <= 0:
                continue
            peak = float(pos.get("peak", pos["entry_price"]))
            sl_cur = float(pos["sl_price"])
            if direction == 1:
                peak = max(peak, bar["high"])
                sl_cur = max(sl_cur, peak - atr_orb * cfg.atr_mult)
            else:
                peak = min(peak, bar["low"])
                sl_cur = min(sl_cur, peak + atr_orb * cfg.atr_mult)
            pos["peak"] = round(peak, 2)
            pos["sl_price"] = round(sl_cur, 2)
            dirty = True
            if tok:
                ticker.update_stop(tok, pos["sl_price"])
            if cfg.mode != "paper" and pos.get("sl_order_id") and not pos.get("software_sl_active"):
                try:
                    broker.modify_sl_order(order_id=pos["sl_order_id"], trigger_price=pos["sl_price"], symbol=sym)
                except Exception as exc:
                    logger.warning("SL modify failed %s: %s", sym, exc)
            # (b) Bar-based fallback breach: paper mode, per-position. A position is
            # tick-eligible only if the WS is up AND it has a valid token (so it was
            # subscribed). If not eligible (WS down, or token is None -> never subscribed),
            # the tick monitor can't see it, so enforce the stop at bar level here.
            # Gating on eligibility (not live watch-state) avoids a double-exit race with
            # a tick that fires mid-loop. Live mode relies on the exchange SL-M.
            tick_eligible = cfg.use_websocket and ticker.is_available and tok is not None
            if cfg.mode == "paper" and not tick_eligible:
                breached = (direction == 1 and bar["low"] <= sl_cur) or \
                           (direction == -1 and bar["high"] >= sl_cur)
                if breached:
                    logger.warning("BAR-FALLBACK SL %s (not tick-watched, tok=%s) sl=%.2f",
                                   sym, tok, sl_cur)
                    _do_stop_exit(broker, cfg, pos, sl_cur, exit_label="bar_sl")
                    if tok:
                        ticker.unregister_stop(tok)
                    bar_exited.add(sym)
        if bar_exited:
            state["positions"] = [p for p in state["positions"] if p["symbol"] not in bar_exited]
            dirty = True
        if dirty:
            save_engine_state(cfg.state_dir, cfg.mode, state)

        # Periodic reconcile (every 5 iterations ~= 5 min)
        reconcile_counter += 1
        if reconcile_counter % 5 == 0:
            state = reconcile_positions(broker, state, cfg, cfg.journal_path)
            save_engine_state(cfg.state_dir, cfg.mode, state)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE C: Hard exit at 15:20
    # ════════════════════════════════════════════════════════════════════════
    logger.info("PHASE C: Hard exit at %s", cfg.hard_exit_ist)

    # Stop WebSocket -- no more software SL checks needed during hard exit
    if cfg.use_websocket:
        ticker.stop()

    state = load_engine_state(cfg.state_dir, cfg.mode)
    hard_exit_all(broker, state, cfg, cfg.journal_path)
    save_engine_state(cfg.state_dir, cfg.mode, state)

    append_jsonl(cfg.journal_path, {
        "ts_utc": utc_now(), "event": "session_complete", "trade_date": today_str,
    })
    logger.info("ORB SESSION complete.")


if __name__ == "__main__":
    main()
