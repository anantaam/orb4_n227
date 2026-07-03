from __future__ import annotations

from dataclasses import dataclass

from orb_config import OrbConfig
from orb_broker_adapter import BrokerAdapter


@dataclass
class SizedIntent:
    symbol: str
    score: float
    ltp: float
    quantity: int
    required_margin_inr: float | None
    effective_leverage: float | None
    rejected_reason: str | None


def _safe_effective_leverage(notional: float, margin: float | None) -> float | None:
    if margin is None or margin <= 0:
        return None
    return notional / margin


def _max_affordable_qty(
    broker: BrokerAdapter,
    symbol: str,
    ltp: float,
    qty_hint: int,
    margin_budget: float,
    probe_order_type: str,
) -> tuple[int, float | None]:
    lo = 1
    hi = max(1, int(qty_hint))
    best_q = 0
    best_m = None

    # Expand upper bound until clearly unaffordable
    for _ in range(10):
        if hi >= max(2 * qty_hint, 1):
            break
        m = broker.estimate_mis_margin(symbol, hi, probe_order_type, None if probe_order_type == "MARKET" else ltp)
        if m is None or m <= margin_budget:
            best_q = hi
            best_m = m
            hi *= 2
        else:
            break
    hi = max(hi, 1)

    # Binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        m = broker.estimate_mis_margin(symbol, mid, probe_order_type, None if probe_order_type == "MARKET" else ltp)
        if m is None:
            hi = mid - 1
            continue
        if m <= margin_budget:
            best_q = mid
            best_m = m
            lo = mid + 1
        else:
            hi = mid - 1

    return best_q, best_m


def _size_one(
    cfg: OrbConfig,
    broker: BrokerAdapter,
    symbol: str,
    score: float,
    ltp: float,
    sleeve: float,
    margin_budget: float,
) -> SizedIntent:
    """Per-symbol sizing given a price (`ltp`). Sizing logic is unchanged — only the
    price source is parameterized so callers can pass a live LTP (Phase A) or the
    actual entry price / orb_close (entry-time re-size)."""
    ltp = float(ltp or 0)
    if ltp <= 0:
        return SizedIntent(symbol, score, 0.0, 0, None, None, "no_ltp")
    qty_hint = int(sleeve // ltp)
    if qty_hint < 1:
        return SizedIntent(symbol, score, ltp, 0, None, None, "qty_lt_1")
    qty, margin = _max_affordable_qty(
        broker=broker,
        symbol=symbol,
        ltp=ltp,
        qty_hint=qty_hint,
        margin_budget=margin_budget,
        probe_order_type=cfg.margin_probe_order_type,
    )
    if qty < 1:
        return SizedIntent(symbol, score, ltp, 0, margin, None, "margin_unaffordable")
    eff_lev = _safe_effective_leverage(qty * ltp, margin)
    if eff_lev is not None and eff_lev > cfg.max_effective_leverage_cap:
        cap_qty = int((cfg.max_effective_leverage_cap * float(margin)) // ltp)
        qty = max(0, cap_qty)
        if qty < 1:
            return SizedIntent(symbol, score, ltp, 0, margin, eff_lev, "leverage_cap")
        eff_lev = _safe_effective_leverage(qty * ltp, margin)
    return SizedIntent(symbol, score, ltp, qty, margin, eff_lev, None)


def _sleeve_budget(cfg: OrbConfig) -> tuple[float, float]:
    sleeve = cfg.engine_capital_inr / float(cfg.max_positions)
    return sleeve, sleeve * cfg.equity_mis_margin_buffer_fraction


def size_candidates(
    cfg: OrbConfig,
    broker: BrokerAdapter,
    ranked_candidates: list[tuple[str, float]],  # (symbol, composite_score)
) -> list[SizedIntent]:
    symbols = [s for s, _ in ranked_candidates]
    ltp_map = broker.fetch_ltp_map(symbols)
    sleeve, margin_budget = _sleeve_budget(cfg)
    return [
        _size_one(cfg, broker, symbol, score, ltp_map.get(symbol, 0), sleeve, margin_budget)
        for symbol, score in ranked_candidates
    ]


def size_at_entry(
    cfg: OrbConfig,
    broker: BrokerAdapter,
    symbol: str,
    score: float,
    price: float,
) -> int:
    """Entry-time sizing using the actual entry price (never LTP-dependent).
    `price` is the breakout/entry level or orb_close — always available in memory,
    so sizing can never fail for lack of a price. Returns qty (0 = unaffordable)."""
    sleeve, margin_budget = _sleeve_budget(cfg)
    return _size_one(cfg, broker, symbol, score, price, sleeve, margin_budget).quantity
