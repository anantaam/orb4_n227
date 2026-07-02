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


def size_candidates(
    cfg: OrbConfig,
    broker: BrokerAdapter,
    ranked_candidates: list[tuple[str, float]],  # (symbol, composite_score)
) -> list[SizedIntent]:
    symbols = [s for s, _ in ranked_candidates]
    ltp_map = broker.fetch_ltp_map(symbols)
    sleeve = cfg.engine_capital_inr / float(cfg.max_positions)
    margin_budget = sleeve * cfg.equity_mis_margin_buffer_fraction

    sized: list[SizedIntent] = []
    for symbol, score in ranked_candidates:
        ltp = float(ltp_map.get(symbol, 0) or 0)
        if ltp <= 0:
            sized.append(SizedIntent(symbol, score, 0.0, 0, None, None, "no_ltp"))
            continue
        qty_hint = int(sleeve // ltp)
        if qty_hint < 1:
            sized.append(SizedIntent(symbol, score, ltp, 0, None, None, "qty_lt_1"))
            continue
        qty, margin = _max_affordable_qty(
            broker=broker,
            symbol=symbol,
            ltp=ltp,
            qty_hint=qty_hint,
            margin_budget=margin_budget,
            probe_order_type=cfg.margin_probe_order_type,
        )
        if qty < 1:
            sized.append(SizedIntent(symbol, score, ltp, 0, margin, None, "margin_unaffordable"))
            continue
        eff_lev = _safe_effective_leverage(qty * ltp, margin)
        if eff_lev is not None and eff_lev > cfg.max_effective_leverage_cap:
            cap_qty = int((cfg.max_effective_leverage_cap * float(margin)) // ltp)
            qty = max(0, cap_qty)
            if qty < 1:
                sized.append(SizedIntent(symbol, score, ltp, 0, margin, eff_lev, "leverage_cap"))
                continue
            eff_lev = _safe_effective_leverage(qty * ltp, margin)

        sized.append(SizedIntent(symbol, score, ltp, qty, margin, eff_lev, None))
    return sized
