from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src import kite_client

logger = logging.getLogger(__name__)


@dataclass
class FillResult:
    order_id: str
    status: str
    filled_quantity: int
    average_price: float | None
    pending_quantity: int
    timed_out: bool
    exec_type: str = "UNKNOWN"   # "LIMIT", "MARKET", "LIMIT+MARKET"


class BrokerAdapter:
    def __init__(self, paper: bool, cfg=None):
        self.paper = paper
        self.cfg = cfg
        self.kite = None
        self._tick_map = None   # symbol -> tick_size, lazy-loaded from instruments.csv

    def login(self):
        self.kite = kite_client.login_or_reuse(paper=self.paper)
        return self.kite

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _snap_to_fill(self, oid: str, qty: int, timeout: int, poll: float = 2.0) -> dict:
        return kite_client.wait_for_order_fill_state(
            self.kite,
            self.paper,
            str(oid or ""),
            requested_quantity=int(qty),
            timeout_seconds=int(timeout),
            poll_interval=poll,
            logger=None,
        )

    def _to_fill_result(self, oid: str, snap: dict, exec_type: str) -> FillResult:
        return FillResult(
            order_id=str(oid or ""),
            status=str(snap.get("status") or ""),
            filled_quantity=int(snap.get("filled_quantity") or 0),
            average_price=(float(snap["average_price"]) if snap.get("average_price") is not None else None),
            pending_quantity=int(snap.get("pending_quantity") or 0),
            timed_out=bool(snap.get("timed_out")),
            exec_type=exec_type,
        )

    def _fetch_best_bid_ask(self, symbol: str) -> tuple[float | None, float | None]:
        """Return (best_bid, best_ask) from live quote. Returns (None, None) on failure."""
        try:
            q = self.kite.quote([f"NSE:{symbol}"])
            depth = q.get(f"NSE:{symbol}", {}).get("depth", {})
            bids = depth.get("buy", [])
            asks = depth.get("sell", [])
            best_bid = float(bids[0]["price"]) if bids and bids[0].get("price") else None
            best_ask = float(asks[0]["price"]) if asks and asks[0].get("price") else None
            return best_bid, best_ask
        except Exception as exc:
            logger.warning("Orderbook fetch failed for %s: %s", symbol, exc)
            return None, None

    def _load_tick_map(self) -> dict:
        """symbol -> tick_size from data/instruments.csv (NSE EQ rows). Cached."""
        if self._tick_map is not None:
            return self._tick_map
        m = {}
        try:
            import csv
            from pathlib import Path
            data_dir = getattr(self.cfg, "data_1m_dir", None) if self.cfg else None
            path = Path(data_dir) / "instruments.csv" if data_dir else Path("data/instruments.csv")
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("exchange") == "NSE" and row.get("segment") == "NSE" and str(row.get("instrument_type","")).upper() == "EQ":
                        try:
                            ts = float(row.get("tick_size") or 0)
                            if ts > 0:
                                m[str(row.get("tradingsymbol","")).strip().upper()] = ts
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:
            logger.warning("tick_size map load failed: %s -- defaulting to 0.05", exc)
        self._tick_map = m
        return m

    def _tick_size(self, symbol: str) -> float:
        """Real NSE-EQ tick size for this symbol (₹0.05 / 0.10 / 1.00 …). Default 0.05."""
        return self._load_tick_map().get(str(symbol).strip().upper(), 0.05)

    def _round_tick(self, price: float, symbol: str) -> float:
        """Round a price to the symbol's tick grid so Kite accepts it."""
        tick = self._tick_size(symbol)
        if tick <= 0:
            return round(price, 2)
        return round(round(price / tick) * tick, 2)

    # ── Sizing helpers ────────────────────────────────────────────────────────

    def fetch_ltp_map(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        return kite_client.fetch_ltp_map_with_retry(self.kite, symbols)

    def estimate_mis_margin(self, symbol: str, quantity: int, order_type: str, price: float | None) -> float | None:
        return kite_client.estimate_order_margin_inr(
            self.kite,
            self.paper,
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="BUY",
            quantity=int(quantity),
            order_type=order_type,
            product="MIS",
            price=price,
        )

    # ── Entry order: LIMIT with MARKET fallback ───────────────────────────────

    def place_entry_order(
        self,
        *,
        symbol: str,
        side: str,          # "BUY" for LONG, "SELL" for SHORT
        quantity: int,
        trigger_price: float,
    ) -> FillResult:
        """
        LIMIT with MARKET fallback.
        1. Fetch orderbook best bid/ask.
        2. Compute limit price:
           BUY:  max(best_ask, trigger_price) + 1 tick
           SELL: min(best_bid, trigger_price) - 1 tick
           Fallback (orderbook failure): trigger +/- N ticks
        3. Poll every entry_limit_poll_ms for up to entry_limit_timeout_seconds.
        4. If unfilled: cancel -> MARKET for remainder.
        5. If partially filled: MARKET for remainder.
        Returns blended FillResult with exec_type.
        """
        tick = self._tick_size(symbol)
        fallback_ticks = getattr(self.cfg, "entry_limit_ticks_from_trigger", 2) if self.cfg else 2
        poll_s = (getattr(self.cfg, "entry_limit_poll_ms", 200) if self.cfg else 200) / 1000.0
        timeout_s = getattr(self.cfg, "entry_limit_timeout_seconds", 3) if self.cfg else 3
        mkt_protection = getattr(self.cfg, "market_protection", -1) if self.cfg else -1

        side_up = side.upper()
        best_bid, best_ask = self._fetch_best_bid_ask(symbol)

        if side_up == "BUY":
            if best_ask is not None:
                limit_px = round(max(best_ask, trigger_price) + tick, 2)
            else:
                limit_px = round(trigger_price + fallback_ticks * tick, 2)
        else:  # SELL
            if best_bid is not None:
                limit_px = round(min(best_bid, trigger_price) - tick, 2)
            else:
                limit_px = round(trigger_price - fallback_ticks * tick, 2)

        limit_px = self._round_tick(limit_px, symbol)   # align to the symbol's tick grid

        # Paper mode: simulate fill at the computed orderbook-aware limit price.
        # In paper mode wait_for_order_fill_state returns average_price=None, which would
        # cause main.py to fall back to last_close.  Return here instead so the recorded
        # entry_price matches the limit price we would have placed in live mode.
        if self.paper:
            logger.info(
                "PAPER ENTRY %s %s qty=%d @ %.2f (orderbook-aware limit)",
                side_up, symbol, quantity, limit_px,
            )
            return FillResult(
                order_id=f"PAPER_{symbol}_{int(time.time())}",
                status="COMPLETE",
                filled_quantity=quantity,
                average_price=limit_px,
                pending_quantity=0,
                timed_out=False,
                exec_type="LIMIT",
            )

        logger.info("ENTRY LIMIT %s %s qty=%d lim=%.2f trig=%.2f", side_up, symbol, quantity, limit_px, trigger_price)

        limit_oid = kite_client.place_order_with_retry(
            self.kite,
            paper=self.paper,
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type=side_up,
            quantity=int(quantity),
            order_type="LIMIT",
            product="MIS",
            price=limit_px,
            market_protection=mkt_protection,
            logger=None,
        )

        # Poll until filled or timeout
        deadline = time.monotonic() + timeout_s
        filled_qty = 0
        avg_price = None
        status = ""
        while time.monotonic() < deadline:
            time.sleep(poll_s)
            snap = self._snap_to_fill(limit_oid, quantity, timeout=2, poll=poll_s)
            filled_qty = int(snap.get("filled_quantity") or 0)
            avg_price = float(snap["average_price"]) if snap.get("average_price") else None
            status = str(snap.get("status") or "")
            if status in ("COMPLETE", "CANCELLED", "REJECTED"):
                break

        remainder = quantity - filled_qty
        if remainder <= 0:
            # Fully filled by limit
            snap = self._snap_to_fill(limit_oid, quantity, timeout=5, poll=1.0)
            return self._to_fill_result(limit_oid, snap, "LIMIT")

        # Cancel the limit order
        try:
            self.cancel_order(limit_oid)
        except Exception:
            pass

        # Fill remainder via MARKET
        logger.info("ENTRY MARKET fallback %s %s qty=%d", side_up, symbol, remainder)
        mkt_oid = kite_client.place_order_with_retry(
            self.kite,
            paper=self.paper,
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type=side_up,
            quantity=int(remainder),
            order_type="MARKET",
            product="MIS",
            price=None,
            market_protection=mkt_protection,
            logger=None,
        )
        mkt_snap = self._snap_to_fill(mkt_oid, remainder, timeout=30, poll=1.0)
        mkt_filled = int(mkt_snap.get("filled_quantity") or 0)
        mkt_price = float(mkt_snap["average_price"]) if mkt_snap.get("average_price") else None

        total_filled = filled_qty + mkt_filled
        if total_filled > 0:
            lim_val = (avg_price or 0.0) * filled_qty
            mkt_val = (mkt_price or 0.0) * mkt_filled
            blended = (lim_val + mkt_val) / total_filled
        else:
            blended = None

        exec_type = "MARKET" if filled_qty == 0 else "LIMIT+MARKET"
        return FillResult(
            order_id=mkt_oid,
            status=str(mkt_snap.get("status") or ""),
            filled_quantity=total_filled,
            average_price=blended,
            pending_quantity=quantity - total_filled,
            timed_out=bool(mkt_snap.get("timed_out")),
            exec_type=exec_type,
        )

    # ── SL order: exchange-native SL-M ────────────────────────────────────────

    def place_sl_order(
        self,
        *,
        symbol: str,
        side: str,          # "SELL" for LONG position, "BUY" for SHORT position
        quantity: int,
        trigger_price: float,
    ) -> str:
        """Place exchange-native SL-M. Returns order_id. No fill wait -- exchange fires it."""
        side_up = side.upper()
        trigger_price = self._round_tick(trigger_price, symbol)   # align to tick grid
        logger.info("SL-M %s %s qty=%d trig=%.2f", side_up, symbol, quantity, trigger_price)
        oid = kite_client.place_order_with_retry(
            self.kite,
            paper=self.paper,
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type=side_up,
            quantity=int(quantity),
            order_type="SL-M",
            product="MIS",
            price=None,
            trigger_price=trigger_price,
            market_protection=getattr(self.cfg, "market_protection", -1) if self.cfg else -1,
            logger=None,
        )
        return str(oid or "")

    def modify_sl_order(self, *, order_id: str, trigger_price: float, symbol: str = "") -> None:
        """Modify an open SL-M order's trigger price (live-mode trailing stop)."""
        if not order_id:
            return
        if self.paper:
            return  # paper trailing is tracked in-process, no broker order to modify
        trig = self._round_tick(trigger_price, symbol) if symbol else round(float(trigger_price), 2)
        self.kite.modify_order(
            variety=self.kite.VARIETY_REGULAR,
            order_id=order_id,
            trigger_price=trig,
        )

    # ── Target order: exchange-held LIMIT ─────────────────────────────────────

    def place_target_order(
        self,
        *,
        symbol: str,
        side: str,          # "SELL" for LONG target, "BUY" for SHORT target
        quantity: int,
        price: float,
    ) -> str:
        """Place LIMIT target order. Returns order_id. Exchange manages it."""
        side_up = side.upper()
        price = self._round_tick(price, symbol)   # align to tick grid
        logger.info("TARGET LIMIT %s %s qty=%d price=%.2f", side_up, symbol, quantity, price)
        oid = kite_client.place_order_with_retry(
            self.kite,
            paper=self.paper,
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type=side_up,
            quantity=int(quantity),
            order_type="LIMIT",
            product="MIS",
            price=price,
            market_protection=getattr(self.cfg, "market_protection", -1) if self.cfg else -1,
            logger=None,
        )
        return str(oid or "")

    # ── EOD exit: LIMIT with MARKET fallback ──────────────────────────────────

    def _ensure_sl_order_cleared(self, sl_order_id: str) -> None:
        """Cancel SL-M before placing any non-SL exit. Swallows errors if already done."""
        if not sl_order_id:
            return
        try:
            self.cancel_order(sl_order_id)
            logger.info("SL-M %s cancelled before exit", sl_order_id)
        except Exception as exc:
            logger.info("SL-M cancel %s: %s (may already be executed/cancelled)", sl_order_id, exc)

    def place_exit_order(
        self,
        *,
        symbol: str,
        side: str,          # "SELL" for closing LONG, "BUY" for covering SHORT
        quantity: int,
        sl_order_id: str = "",
    ) -> FillResult:
        """
        EOD exit: LIMIT with MARKET fallback.
        1. Cancels SL-M first (_ensure_sl_order_cleared).
        2. Fetches orderbook; limit price:
           SELL (close LONG):  best_bid * exit_limit_bid_fraction
           BUY  (cover SHORT): best_ask * exit_limit_ask_fraction
           Orderbook failure -> skip limit, go straight to MARKET.
        3. Polls exit_limit_poll_ms * exit_limit_timeout_seconds.
        4. Unfilled -> cancel -> MARKET.
        """
        self._ensure_sl_order_cleared(sl_order_id)

        side_up = side.upper()
        poll_s = (getattr(self.cfg, "exit_limit_poll_ms", 200) if self.cfg else 200) / 1000.0
        timeout_s = getattr(self.cfg, "exit_limit_timeout_seconds", 2) if self.cfg else 2
        bid_frac = getattr(self.cfg, "exit_limit_bid_fraction", 0.9995) if self.cfg else 0.9995
        ask_frac = getattr(self.cfg, "exit_limit_ask_fraction", 1.0005) if self.cfg else 1.0005
        mkt_protection = getattr(self.cfg, "market_protection", -1) if self.cfg else -1

        best_bid, best_ask = self._fetch_best_bid_ask(symbol)
        skip_limit = False
        limit_px = None

        if side_up == "SELL":
            if best_bid is None or best_bid <= 0:
                skip_limit = True
            else:
                limit_px = round(best_bid * bid_frac, 2)
        else:  # BUY
            if best_ask is None or best_ask <= 0:
                skip_limit = True
            else:
                limit_px = round(best_ask * ask_frac, 2)

        if limit_px is not None:
            limit_px = self._round_tick(limit_px, symbol)   # align to tick grid

        filled_qty = 0
        avg_price = None

        if not skip_limit and limit_px is not None:
            logger.info("EXIT LIMIT %s %s qty=%d lim=%.2f", side_up, symbol, quantity, limit_px)
            limit_oid = kite_client.place_order_with_retry(
                self.kite, paper=self.paper, tradingsymbol=symbol, exchange="NSE",
                transaction_type=side_up, quantity=int(quantity), order_type="LIMIT",
                product="MIS", price=limit_px, market_protection=mkt_protection, logger=None,
            )

            deadline = time.monotonic() + timeout_s
            status = ""
            while time.monotonic() < deadline:
                time.sleep(poll_s)
                snap = self._snap_to_fill(limit_oid, quantity, timeout=2, poll=poll_s)
                filled_qty = int(snap.get("filled_quantity") or 0)
                avg_price = float(snap["average_price"]) if snap.get("average_price") else None
                status = str(snap.get("status") or "")
                if status in ("COMPLETE", "CANCELLED", "REJECTED"):
                    break

            remainder = quantity - filled_qty
            if remainder <= 0:
                snap = self._snap_to_fill(limit_oid, quantity, timeout=5, poll=1.0)
                return self._to_fill_result(limit_oid, snap, "LIMIT")

            try:
                self.cancel_order(limit_oid)
            except Exception:
                pass

            if filled_qty > 0:
                # Partial limit fill -- MARKET for remainder
                mkt_oid = kite_client.place_order_with_retry(
                    self.kite, paper=self.paper, tradingsymbol=symbol, exchange="NSE",
                    transaction_type=side_up, quantity=int(remainder), order_type="MARKET",
                    product="MIS", price=None, market_protection=mkt_protection, logger=None,
                )
                mkt_snap = self._snap_to_fill(mkt_oid, remainder, timeout=30, poll=1.0)
                mkt_filled = int(mkt_snap.get("filled_quantity") or 0)
                mkt_price = float(mkt_snap["average_price"]) if mkt_snap.get("average_price") else None
                total = filled_qty + mkt_filled
                blended = ((avg_price or 0) * filled_qty + (mkt_price or 0) * mkt_filled) / total if total else None
                return FillResult(
                    order_id=mkt_oid, status=str(mkt_snap.get("status") or ""),
                    filled_quantity=total, average_price=blended,
                    pending_quantity=quantity - total,
                    timed_out=bool(mkt_snap.get("timed_out")), exec_type="LIMIT+MARKET",
                )
            # Fully unfilled limit -- fall through to full MARKET

        # Full MARKET exit
        logger.info("EXIT MARKET %s %s qty=%d", side_up, symbol, quantity)
        mkt_oid = kite_client.place_order_with_retry(
            self.kite, paper=self.paper, tradingsymbol=symbol, exchange="NSE",
            transaction_type=side_up, quantity=int(quantity), order_type="MARKET",
            product="MIS", price=None, market_protection=mkt_protection, logger=None,
        )
        mkt_snap = self._snap_to_fill(mkt_oid, quantity, timeout=30, poll=1.0)
        return self._to_fill_result(mkt_oid, mkt_snap, "MARKET")

    # ── Positions + cancel + history ──────────────────────────────────────────

    def get_positions(self) -> list[dict[str, Any]]:
        return kite_client.get_positions_with_retry(self.kite, self.paper)

    def cancel_order(self, order_id: str) -> bool:
        return kite_client.cancel_order(
            self.kite,
            self.paper,
            order_id=str(order_id or ""),
            variety="regular",
            logger=None,
        )

    def get_order_history(self, order_id: str) -> list[dict]:
        try:
            return self.kite.order_history(order_id=str(order_id)) or []
        except Exception:
            return []
