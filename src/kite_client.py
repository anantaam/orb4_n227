"""
Kite client: login, instruments download, positions, order place + status poll.
In paper mode, order methods only log and return simulated order_id.
"""
import os
import threading
import time
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyotp
import requests
from kiteconnect import KiteConnect

try:
    from kiteconnect import KiteTicker
except Exception:  # pragma: no cover
    KiteTicker = None

try:
    from . import discord_notifier
except ImportError:
    discord_notifier = None


def _get_creds():
    return {
        "api_key": os.getenv("KITE_API_KEY"),
        "api_secret": os.getenv("KITE_API_SECRET"),
        "user_id": os.getenv("KITE_USER_ID"),
        "password": os.getenv("KITE_PASSWORD"),
        "totp_secret": os.getenv("KITE_TOTP_SECRET"),
    }


def login(paper: bool = False) -> KiteConnect | None:
    """Login to Kite. Returns session in both paper and live (paper only affects order placement)."""
    creds = _get_creds()
    for k, v in creds.items():
        if not v:
            raise ValueError(f"Missing env var for Kite: KITE_{k.upper() if k != 'api_key' else 'API_KEY'} etc.")
    api_key = creds["api_key"]
    api_secret = creds["api_secret"]
    user_id = creds["user_id"]
    password = creds["password"]
    totp_secret = creds["totp_secret"]

    try:
        pin = pyotp.TOTP(totp_secret).now()
        twofa = f"{int(pin):06d}" if len(pin) <= 5 else pin
        kite = KiteConnect(api_key=api_key)
        s = requests.Session()

        r = s.get(kite.login_url(), allow_redirects=False)
        loc = r.headers["location"]
        sess_id = parse_qs(urlparse(loc).query)["sess_id"][0]
        s.get(loc)
        s.get(
            "https://kite.zerodha.com/api/connect/session",
            params={"sess_id": sess_id, "api_key": api_key},
        ).json()
        r = s.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": user_id, "password": password, "type": "user_id"},
        )
        request_id = r.json()["data"]["request_id"]
        s.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": user_id,
                "request_id": request_id,
                "twofa_value": twofa,
                "twofa_type": "totp",
                "skip_session": "true",
            },
        )
        r = s.get(
            "https://kite.zerodha.com/connect/finish",
            params={"api_key": api_key, "sess_id": sess_id},
            allow_redirects=False,
        )
        request_token = parse_qs(urlparse(r.headers["location"]).query)["request_token"][0]
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        return kite
    except Exception as e:
        if discord_notifier:
            discord_notifier.notify_error(f"Kite login failed: {e}")
        raise RuntimeError(f"Kite login failed: {e}") from e


def _parse_expiry(expiry) -> date | None:
    """Parse instrument expiry to date. Handles str, datetime, date, or None."""
    if expiry is None or (isinstance(expiry, float) and pd.isna(expiry)):
        return None
    if isinstance(expiry, date) and not isinstance(expiry, datetime):
        return expiry
    if isinstance(expiry, datetime):
        return expiry.date()
    try:
        s = str(expiry).strip()
        if not s:
            return None
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _build_futures_contracts(instruments_df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Build underlying -> list of {tradingsymbol, instrument_token, expiry, lot_size} for NFO FUT.
    Sorted by expiry ascending. Used to pick a contract that does not expire before exit_date.
    """
    nfo = instruments_df[
        (instruments_df["exchange"].astype(str) == "NFO")
        & (instruments_df["instrument_type"].astype(str) == "FUT")
    ].copy()
    if nfo.empty or "expiry" not in nfo.columns or "name" not in nfo.columns:
        return {}
    nfo["_expiry_d"] = nfo["expiry"].apply(_parse_expiry)
    nfo = nfo.dropna(subset=["_expiry_d"])
    nfo["_name"] = nfo["name"].astype(str).str.upper().str.strip()
    out: dict[str, list[dict]] = {}
    for _, row in nfo.iterrows():
        name = row["_name"]
        if not name:
            continue
        out.setdefault(name, []).append({
            "tradingsymbol": str(row.get("tradingsymbol", "")).strip(),
            "instrument_token": int(row["instrument_token"]),
            "expiry": row["_expiry_d"],
            "lot_size": int(row.get("lot_size", 1)),
        })
    for name in out:
        out[name].sort(key=lambda x: x["expiry"])
    return out


# Require FUT expiry at least this many calendar days after planned exit (roll buffer).
FUTURES_EXPIRY_BUFFER_DAYS = 7


def resolve_futures_contract(
    underlying: str,
    exit_date: date,
    futures_contracts: dict[str, list[dict]] | None,
) -> tuple[str, int, int] | None:
    """
    Pick an NFO FUT contract for the given underlying whose expiry is on or after
    exit_date + FUTURES_EXPIRY_BUFFER_DAYS (calendar days).
    Returns (tradingsymbol, instrument_token, lot_size) or None if no suitable contract.
    Chooses the nearest such expiry.
    """
    if not futures_contracts or not underlying:
        return None
    key = str(underlying).strip().upper()
    contracts = futures_contracts.get(key)
    if not contracts:
        return None
    min_expiry_ok = exit_date + timedelta(days=FUTURES_EXPIRY_BUFFER_DAYS)
    for c in contracts:
        if c["expiry"] >= min_expiry_ok:
            return (c["tradingsymbol"], c["instrument_token"], c["lot_size"])
    return None


def tick_size_maps_from_instruments_df(
    instruments_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[int, float]]:
    """
    Build tick-size maps from a Kite instruments dataframe (same row set as written to instruments.csv).
    Used immediately after each kite.instruments() fetch so LIMIT rounding uses that snapshot, not a stale read.
    """
    if instruments_df is None or instruments_df.empty:
        return {}, {}
    if "tick_size" not in instruments_df.columns or "tradingsymbol" not in instruments_df.columns:
        return {}, {}
    if "exchange" not in instruments_df.columns:
        return {}, {}
    by_sym: dict[tuple[str, str], float] = {}
    by_token: dict[int, float] = {}
    has_tok_col = "instrument_token" in instruments_df.columns
    for _, row in instruments_df.iterrows():
        ex = str(row.get("exchange", "")).strip().upper()
        sym = str(row.get("tradingsymbol", "")).strip().upper()
        try:
            t = float(row.get("tick_size"))
        except (TypeError, ValueError):
            continue
        if t <= 0 or not ex or not sym:
            continue
        by_sym[(ex, sym)] = t
        if has_tok_col:
            try:
                raw_tok = row.get("instrument_token")
                if raw_tok is None or pd.isna(raw_tok):
                    continue
                tok = int(float(raw_tok))
                if tok > 0:
                    by_token[tok] = t
            except (TypeError, ValueError):
                pass
    return by_sym, by_token


def download_instruments(
    kite: KiteConnect | None,
    base_dir: str,
    paper: bool,
    logger=None,
) -> tuple[
    dict[str, int],
    dict[str, int] | None,
    dict[str, list[dict]] | None,
    dict[tuple[str, str], float],
    dict[int, float],
]:
    """
    **Call once per engine run** after Kite login: fetches `kite.instruments()`, writes `instruments.csv`,
    and derives tick-size maps from that same in-memory snapshot (live/paper both use a real session).

    `paper` does not skip the download; it only affects order placement elsewhere.

    Returns:
        symbol_to_token, symbol_to_lot (NFO) or None, futures_contracts or None,
        tick_by_exchange_symbol, tick_by_instrument_token (for LIMIT price rounding).

    If `kite` is None (offline), reads existing `instruments.csv` for EQ tokens and tick maps from disk.
    """
    _ = paper  # API download runs whenever kite is set; paper vs live does not skip instruments.
    instruments_path = os.path.join(base_dir, "instruments.csv")
    os.makedirs(base_dir, exist_ok=True)

    if kite is None:
        tick_ex, tick_tok = load_tick_size_maps(base_dir)
        if os.path.isfile(instruments_path):
            try:
                df = pd.read_csv(instruments_path)
                eq = df[
                    (df.get("exchange", "") == "NSE")
                    & (df.get("segment", "") == "NSE")
                    & (df.get("instrument_type", "") == "EQ")
                ]
                if "tradingsymbol" in eq.columns and "instrument_token" in eq.columns:
                    eq = eq.copy()
                    eq["tradingsymbol"] = eq["tradingsymbol"].astype(str).str.upper().str.strip()
                    symbol_to_token = dict(zip(eq["tradingsymbol"], eq["instrument_token"].astype(int)))
                    if logger:
                        logger.info(
                            "No Kite session: using existing %s; NSE EQ: %s symbols; tick keys=%s/%s",
                            instruments_path,
                            len(symbol_to_token),
                            len(tick_ex),
                            len(tick_tok),
                        )
                    return symbol_to_token, None, None, tick_ex, tick_tok
            except Exception:
                pass
        if logger:
            logger.warning("No kite and no/invalid instruments.csv; symbol_to_token will be empty")
        return {}, None, None, tick_ex, tick_tok

    instruments_df = pd.DataFrame(kite.instruments())
    instruments_df = instruments_df[instruments_df["exchange"].isin(["NSE", "NFO"])].copy()
    eq_df = instruments_df[
        (instruments_df["exchange"] == "NSE")
        & (instruments_df["segment"] == "NSE")
        & (instruments_df["instrument_type"] == "EQ")
    ][["tradingsymbol", "instrument_token"]].copy()
    eq_df["tradingsymbol"] = eq_df["tradingsymbol"].astype(str).str.upper().str.strip()
    instruments_df.to_csv(instruments_path, index=False)
    tick_ex, tick_tok = tick_size_maps_from_instruments_df(instruments_df)
    if logger:
        logger.info(
            "Saved %s from kite.instruments(); NSE EQ: %s symbols; tick maps: %s (ex,sym) / %s tokens",
            instruments_path,
            len(eq_df),
            len(tick_ex),
            len(tick_tok),
        )

    symbol_to_token = dict(zip(eq_df["tradingsymbol"], eq_df["instrument_token"].astype(int)))

    symbol_to_lot = None
    nfo = instruments_df[instruments_df["segment"].astype(str).str.contains("NFO", na=False)]
    if not nfo.empty and "lot_size" in nfo.columns and "tradingsymbol" in nfo.columns:
        symbol_to_lot = {}
        for _, row in nfo.iterrows():
            sym = str(row.get("tradingsymbol", "")).strip().upper()
            if sym:
                symbol_to_lot[sym] = int(row.get("lot_size", 1))

    futures_contracts = _build_futures_contracts(instruments_df)

    return symbol_to_token, symbol_to_lot, futures_contracts, tick_ex, tick_tok


def round_price_to_tick_size(price: float, tick: float | None) -> float:
    """
    Round a limit price to a valid multiple of Kite's tick_size (avoids InputException).
    Uses Decimal so 0.05 steps are exact (float noise can otherwise yield invalid prices).
    """
    if tick is None or float(tick) <= 0:
        return round(float(price), 2)
    dt = Decimal(str(float(tick)))
    dp = Decimal(str(float(price)))
    n = (dp / dt).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(n * dt)


def load_tick_size_maps(
    base_dir: str,
) -> tuple[dict[tuple[str, str], float], dict[int, float]]:
    """
    Read instruments.csv from disk and build tick maps (e.g. offline tests or kite=None path).
    On a normal engine run, prefer tick maps returned by download_instruments() from the live API snapshot.
    """
    path = os.path.join(base_dir, "instruments.csv")
    if not os.path.isfile(path):
        return {}, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}
    return tick_size_maps_from_instruments_df(df)


def exchange_symbol_tick_map(base_dir: str) -> dict[tuple[str, str], float]:
    """
    Load (exchange, tradingsymbol) -> tick_size from instruments.csv (Kite master).
    Used to round LIMIT prices so Kite accepts them (e.g. NSE EQ tick 0.05 vs 0.01).
    """
    return load_tick_size_maps(base_dir)[0]


def estimate_order_margin_inr(
    kite: KiteConnect | None,
    paper: bool,
    *,
    exchange: str,
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    order_type: str,
    product: str,
    price: float | None,
    logger=None,
) -> float | None:
    """
    Kite order_margins (single order). Returns total margin in INR or None on failure/paper.
    """
    if paper or kite is None or quantity <= 0:
        return None
    ot = (order_type or "MARKET").upper()
    params = [
        {
            "exchange": exchange,
            "tradingsymbol": tradingsymbol,
            "transaction_type": transaction_type,
            "variety": KiteConnect.VARIETY_REGULAR,
            "product": product,
            "order_type": ot,
            "quantity": int(quantity),
            "price": float(price or 0),
            "trigger_price": 0,
        }
    ]
    try:
        data = kite.order_margins(params)
    except Exception as e:
        if logger:
            logger.warning("order_margins failed for %s %s: %s", exchange, tradingsymbol, e)
        return None
    row = None
    if isinstance(data, list) and data:
        row = data[0]
    elif isinstance(data, dict):
        row = data.get("data", data)
        if isinstance(row, list) and row:
            row = row[0]
    if not isinstance(row, dict):
        return None
    total = row.get("total")
    if total is None:
        total = row.get("final_margin")
    if total is None and isinstance(row.get("margins"), list) and row["margins"]:
        m0 = row["margins"][0]
        if isinstance(m0, dict):
            total = m0.get("total") or m0.get("margin")
    try:
        return float(total) if total is not None else None
    except (TypeError, ValueError):
        return None


def fetch_ltp_mixed(
    kite: KiteConnect | None,
    specs: list[tuple[str, str]],
    logger=None,
) -> dict[tuple[str, str], float]:
    """
    Last traded price for (exchange, tradingsymbol) pairs, exchange NSE|NFO upper.
    """
    if kite is None or not specs:
        return {}
    keys = []
    seen: set[tuple[str, str]] = set()
    for ex, sym in specs:
        exu = (ex or "NSE").strip().upper()
        syu = (sym or "").strip().upper()
        if not syu:
            continue
        t = (exu, syu)
        if t in seen:
            continue
        seen.add(t)
        keys.append(f"{exu}:{syu}")
    out: dict[tuple[str, str], float] = {}
    chunk = 400
    try:
        for i in range(0, len(keys), chunk):
            batch = keys[i : i + chunk]
            q = kite.quote(batch)
            for key, row in (q or {}).items():
                parts = str(key).split(":")
                exu = parts[-2].strip().upper() if len(parts) > 1 else "NSE"
                syu = parts[-1].strip().upper()
                lp = row.get("last_price")
                if lp is None and isinstance(row.get("ohlc"), dict):
                    lp = row["ohlc"].get("close")
                if lp is not None:
                    try:
                        v = float(lp)
                        if v > 0:
                            out[(exu, syu)] = v
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        if logger:
            logger.warning("fetch_ltp_mixed failed: %s", e)
    return out


def cancel_order(
    kite: KiteConnect | None,
    paper: bool,
    order_id: str,
    variety: str = "regular",
    logger=None,
) -> bool:
    if paper or kite is None or not order_id or str(order_id).startswith("PAPER"):
        return True
    try:
        kite.cancel_order(variety=variety, order_id=str(order_id))
        if logger:
            logger.info("Cancelled order %s", order_id)
        return True
    except Exception as e:
        if logger:
            logger.warning("cancel_order %s failed: %s", order_id, e)
        return False


def fetch_ltp_map(
    kite: KiteConnect | None,
    symbols: list[str],
    logger=None,
) -> dict[str, float]:
    """
    Batch last traded price for NSE equity symbols (for cash sleeve sizing).
    Returns upper-case symbol -> last_price (>0 only).
    """
    if kite is None or not symbols:
        return {}
    instruments = []
    seen = set()
    for s in symbols:
        sym = (s or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            instruments.append(f"NSE:{sym}")
    if not instruments:
        return {}
    try:
        # Kite allows up to ~500 instruments per quote call; chunk if needed
        out: dict[str, float] = {}
        chunk = 400
        for i in range(0, len(instruments), chunk):
            batch = instruments[i : i + chunk]
            q = kite.quote(batch)
            for key, row in q.items():
                sym = key.split(":")[-1].strip().upper() if ":" in str(key) else str(key).upper()
                lp = row.get("last_price")
                if lp is None and isinstance(row.get("ohlc"), dict):
                    lp = row["ohlc"].get("close")
                if lp is not None:
                    try:
                        v = float(lp)
                        if v > 0:
                            out[sym] = v
                    except (TypeError, ValueError):
                        pass
        return out
    except Exception as e:
        if logger:
            logger.warning("fetch_ltp_map failed: %s", e)
        return {}


def get_positions(kite: KiteConnect | None, paper: bool) -> list[dict]:
    """Fetch current positions from Kite. In paper mode return []."""
    if paper or kite is None:
        return []
    try:
        data = kite.positions()
        return data.get("net", []) or []
    except Exception:
        return []


def get_positions_with_retry(
    kite: KiteConnect | None,
    paper: bool,
    logger=None,
    max_attempts: int = 3,
) -> list[dict]:
    if paper or kite is None:
        return []
    from .retry_util import call_with_retry

    return call_with_retry(
        lambda: (kite.positions().get("net", []) or []),
        max_attempts=max_attempts,
        base_delay_seconds=0.8,
        logger=logger,
        operation_name="kite.positions",
    )


def place_order(
    kite: KiteConnect | None,
    paper: bool,
    *,
    tradingsymbol: str,
    exchange: str,
    transaction_type: str,
    quantity: int,
    order_type: str,
    product: str = "CNC",
    price: float | None = None,
    tick_size: float | None = None,
    market_protection: int = -1,
    trigger_price: float | None = None,
    tag: str | None = None,
    logger=None,
) -> str | None:
    """
    Place order. Returns order_id. In paper mode logs and returns a fake order_id.
    For LIMIT orders, pass tick_size from instruments.csv so the price is a valid tick multiple.
    """
    if paper or kite is None:
        fake_id = f"PAPER_{tradingsymbol}_{int(time.time())}"
        if logger:
            logger.info(
                "PAPER order: %s %s %s qty=%s type=%s product=%s -> %s",
                transaction_type,
                tradingsymbol,
                exchange,
                quantity,
                order_type,
                product,
                fake_id,
            )
        return fake_id

    try:
        params = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product": product,
        }
        order_type_u = str(order_type or "").upper()
        if price is not None and order_type_u == "LIMIT":
            p = float(price)
            if tick_size is not None and float(tick_size) > 0:
                p = round_price_to_tick_size(p, float(tick_size))
            params["price"] = p
        elif order_type_u in ("MARKET", "SL-M"):
            # Zerodha rejects MARKET/SL-M via API when market_protection is missing/0.
            # Allowed values: -1 (broker default) or 1..100.
            mp = int(market_protection)
            if mp == 0 or mp < -1 or mp > 100:
                mp = -1
            params["market_protection"] = mp
            if trigger_price is not None:
                params["trigger_price"] = float(trigger_price)
        if tag:
            params["tag"] = str(tag)[:20]
        order_id = kite.place_order(variety="regular", **params)
        if logger:
            logger.info("Placed order %s: %s", order_id, params)
        return str(order_id)
    except Exception as e:
        if discord_notifier:
            discord_notifier.notify_error(f"Place order failed: {e}")
        if logger:
            logger.exception("Place order failed: %s", e)
        raise


def wait_for_order_confirmation(
    kite: KiteConnect | None,
    paper: bool,
    order_id: str,
    timeout_seconds: int = 120,
    poll_interval: float = 2.0,
    logger=None,
) -> bool:
    """
    Poll order status until COMPLETE, CANCELLED, or REJECTED, or timeout.
    Returns True if status is COMPLETE. In paper mode returns True immediately.
    """
    snap = wait_for_order_fill_state(
        kite,
        paper,
        order_id,
        requested_quantity=0,
        timeout_seconds=timeout_seconds,
        poll_interval=poll_interval,
        logger=logger,
    )
    return bool(snap.get("fully_filled")) or (
        (snap.get("status") or "").upper() in ("COMPLETE", "TRADED") and int(snap.get("filled_quantity") or 0) > 0
    )


def _snapshot_from_order_row(
    row: dict,
    requested_quantity: int,
    *,
    timed_out: bool = False,
) -> dict:
    """Normalize Kite order_history row into fill snapshot."""
    st = str(row.get("status") or "").upper()
    try:
        filled = int(row.get("filled_quantity") or 0)
    except (TypeError, ValueError):
        filled = 0
    try:
        pending = int(row.get("pending_quantity") or 0)
    except (TypeError, ValueError):
        pending = 0
    avg = row.get("average_price")
    try:
        avg_f = float(avg) if avg is not None and str(avg).strip() != "" else None
    except (TypeError, ValueError):
        avg_f = None
    req = max(int(requested_quantity or 0), 0)
    terminal_ok = st in ("COMPLETE", "TRADED", "CANCELLED", "REJECTED")
    fully_filled = req > 0 and filled >= req and pending == 0 and st in ("COMPLETE", "TRADED")
    partial_fill = filled > 0 and filled < req
    zero_fill = filled == 0
    open_partial = st == "OPEN" and filled > 0 and pending > 0
    return {
        "status": st,
        "filled_quantity": filled,
        "pending_quantity": pending,
        "requested_quantity": req,
        "average_price": avg_f,
        "fully_filled": fully_filled,
        "partial_fill": partial_fill or open_partial,
        "zero_fill": zero_fill and terminal_ok,
        "timed_out": timed_out,
        "terminal": terminal_ok,
        "raw_row": row,
    }


def _parse_positive_price(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        p = float(val)
        return p if p > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_live_average_fill_price(
    kite: KiteConnect,
    order_id: str,
    snapshot: dict | None = None,
    logger=None,
) -> float | None:
    """
    Best-effort execution price for a live order: order_history last row, then order_trades VWAP.
    """
    oid = str(order_id or "").strip()
    if not oid:
        return None
    snap = snapshot or {}

    p = _parse_positive_price(snap.get("average_price"))
    if p is not None:
        return p
    raw = snap.get("raw_row") or {}
    p = _parse_positive_price(raw.get("average_price"))
    if p is not None:
        return p

    try:
        history = kite.order_history(oid)
        if history:
            last = history[-1] or {}
            p = _parse_positive_price(last.get("average_price"))
            if p is not None:
                return p
    except Exception as e:
        if logger:
            logger.debug("order_history refresh for fill price %s: %s", oid, e)

    try:
        trades = kite.order_trades(oid)
    except Exception as e:
        if logger:
            logger.warning("order_trades failed for %s: %s", oid, e)
        trades = []
    if not trades:
        return None
    total_q = 0
    total_pv = 0.0
    for t in trades:
        q = int(t.get("quantity") or 0)
        if q <= 0:
            continue
        tp = _parse_positive_price(t.get("average_price") or t.get("price"))
        if tp is None:
            continue
        total_q += q
        total_pv += tp * q
    if total_q <= 0:
        return None
    return total_pv / total_q


def resolve_fill_price_for_order(
    paper: bool,
    order: dict,
    snap: dict,
    kite: KiteConnect | None,
    order_id: str,
    logger=None,
) -> float | None:
    """
    Paper: fill_price is always sizing_price (LTP used for sizing) when present; sells have no sizing_price.
    Live: average fill from order history / order_trades (never sizing_price as fill).
    """
    sizing = order.get("sizing_price")
    oid = str(order_id or "").strip()

    if paper or (oid.startswith("PAPER")):
        if sizing is not None:
            try:
                return float(sizing)
            except (TypeError, ValueError):
                return None
        return None

    if kite is None or not oid:
        return None

    p = fetch_live_average_fill_price(kite, oid, snap, logger=logger)
    if p is not None:
        return p
    if logger:
        logger.warning(
            "Live fill price could not be resolved from history/trades for order %s",
            oid,
        )
    return None


def resolve_paper_sell_mark_price(
    order: dict,
    ltp_by_symbol: dict[str, float] | None,
    ltp_mixed: dict[tuple[str, str], float] | None,
    kite: KiteConnect | None,
    logger=None,
) -> float | None:
    """
    Paper-mode fallback when resolve_fill_price_for_order returns None on sells
    (no sizing_price). Uses cached LTP maps, then Kite quotes for (exchange, symbol)
    or underlying equity symbol. Must only be called when paper=True.
    """
    ex = (order.get("exchange") or "NSE").strip().upper()
    tsym = (order.get("symbol") or "").strip().upper()
    und = (order.get("underlying") or "").strip().upper()
    if not tsym:
        return None

    lm = ltp_mixed or {}
    if lm:
        p = lm.get((ex, tsym))
        pv = _parse_positive_price(p)
        if pv is not None:
            return pv

    lb = ltp_by_symbol or {}
    for sym in (tsym, und):
        if not sym:
            continue
        pv = _parse_positive_price(lb.get(sym))
        if pv is not None:
            return pv

    if kite is None:
        return None

    if ex == "NFO":
        M = fetch_ltp_mixed(kite, [(ex, tsym)], logger=logger)
        pv = _parse_positive_price(M.get((ex, tsym)))
        if pv is not None:
            return pv

    fetch_sym = und or tsym
    if not fetch_sym:
        return None
    M = fetch_ltp_map_with_retry(kite, [fetch_sym], logger=logger)
    pv = _parse_positive_price(M.get(fetch_sym.upper()))
    if pv is not None:
        return pv
    if logger:
        logger.warning(
            "Paper sell LTP fallback: no price for exchange=%s symbol=%s underlying=%s",
            ex,
            tsym,
            und or "-",
        )
    return None


def wait_for_order_fill_state(
    kite: KiteConnect | None,
    paper: bool,
    order_id: str,
    requested_quantity: int,
    timeout_seconds: int = 120,
    poll_interval: float = 2.0,
    logger=None,
) -> dict:
    """
    Poll until terminal status or timeout. Returns snapshot with filled/pending/avg price.
    Paper / PAPER_* ids: assumed full fill at requested qty (no broker price — use order sizing).
    """
    if paper or kite is None or (order_id and str(order_id).startswith("PAPER")):
        rq = int(requested_quantity or 0)
        return {
            "status": "COMPLETE",
            "filled_quantity": rq,
            "pending_quantity": 0,
            "requested_quantity": rq,
            "average_price": None,
            "fully_filled": rq > 0,
            "partial_fill": False,
            "zero_fill": rq == 0,
            "timed_out": False,
            "terminal": True,
            "raw_row": {},
        }

    deadline = time.time() + timeout_seconds
    last_row: dict = {}
    while time.time() < deadline:
        try:
            history = kite.order_history(order_id)
            if history:
                last_row = history[-1] or {}
                st = str(last_row.get("status") or "").upper()
                if st in ("COMPLETE", "TRADED", "CANCELLED", "REJECTED"):
                    snap = _snapshot_from_order_row(last_row, requested_quantity, timed_out=False)
                    if logger:
                        logger.info(
                            "Order %s terminal %s filled=%s pending=%s avg=%s",
                            order_id,
                            snap["status"],
                            snap["filled_quantity"],
                            snap["pending_quantity"],
                            snap["average_price"],
                        )
                    return snap
        except Exception as e:
            if logger:
                logger.debug("Order status poll error: %s", e)
        time.sleep(poll_interval)

    try:
        history = kite.order_history(order_id)
        if history:
            last_row = history[-1] or {}
    except Exception as e:
        if logger:
            logger.debug("Final order read failed: %s", e)
        last_row = {}
    if not last_row:
        last_row = {
            "status": "TIMEOUT_NO_HISTORY",
            "filled_quantity": 0,
            "pending_quantity": 0,
        }

    snap = _snapshot_from_order_row(last_row, requested_quantity, timed_out=True)
    snap["timed_out"] = True
    if snap["partial_fill"] or (snap["filled_quantity"] > 0 and snap["filled_quantity"] < requested_quantity):
        if logger:
            logger.warning(
                "Order %s TIMEOUT/partial: status=%s filled=%s requested=%s pending=%s",
                order_id,
                snap["status"],
                snap["filled_quantity"],
                requested_quantity,
                snap["pending_quantity"],
            )
        if discord_notifier:
            discord_notifier.notify_error(
                f"Order {order_id} not fully filled: status={snap['status']} "
                f"filled={snap['filled_quantity']}/{requested_quantity} pending={snap['pending_quantity']}"
            )
    elif not snap["fully_filled"] and snap["filled_quantity"] == 0:
        if logger:
            logger.warning("Order %s timeout with zero fill status=%s", order_id, snap["status"])
        if discord_notifier:
            discord_notifier.notify_error(
                f"Order {order_id} timeout / zero fill: status={snap['status']}"
            )
    else:
        if logger:
            logger.warning("Order %s confirmation timeout after %s s", order_id, timeout_seconds)
        if discord_notifier:
            discord_notifier.notify_error(
                f"Order {order_id} confirmation timeout after {timeout_seconds}s"
            )
    return snap


def place_order_with_retry(
    kite: KiteConnect | None,
    paper: bool,
    *,
    tradingsymbol: str,
    exchange: str,
    transaction_type: str,
    quantity: int,
    order_type: str,
    product: str = "CNC",
    price: float | None = None,
    tick_size: float | None = None,
    market_protection: int = -1,
    trigger_price: float | None = None,
    tag: str | None = None,
    logger=None,
    max_attempts: int = 3,
) -> str | None:
    from .retry_util import call_with_retry

    def _go() -> str | None:
        return place_order(
            kite,
            paper,
            tradingsymbol=tradingsymbol,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
            tick_size=tick_size,
            market_protection=market_protection,
            trigger_price=trigger_price,
            tag=tag,
            logger=logger,
        )

    return call_with_retry(
        _go,
        max_attempts=max_attempts,
        base_delay_seconds=1.0,
        logger=logger,
        operation_name=f"kite.place_order {transaction_type} {tradingsymbol} qty={quantity}",
    )


def fetch_ltp_map_with_retry(
    kite: KiteConnect | None,
    symbols: list[str],
    logger=None,
    max_attempts: int = 3,
) -> dict[str, float]:
    from .retry_util import call_with_retry

    return call_with_retry(
        lambda: fetch_ltp_map(kite, symbols, logger=logger),
        max_attempts=max_attempts,
        base_delay_seconds=0.8,
        logger=logger,
        operation_name="kite.fetch_ltp_map",
    )


def is_market_data_streaming_live(
    kite: KiteConnect | None,
    base_dir: str,
    logger=None,
    wait_seconds: float = 10.0,
) -> bool:
    """
    Probe live market-data availability using Kite websocket (ticker).
    Returns True if at least one tick is received within wait_seconds.
    """
    if kite is None or KiteTicker is None:
        if logger:
            logger.warning("Kite websocket probe unavailable (kite session or KiteTicker missing)")
        return False

    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = getattr(kite, "access_token", None)
    if not api_key or not access_token:
        if logger:
            logger.warning("Kite websocket probe unavailable (missing api_key/access_token)")
        return False

    token = None
    instruments_path = os.path.join(base_dir, "instruments.csv")
    if os.path.isfile(instruments_path):
        try:
            df = pd.read_csv(instruments_path)
            eq = df[
                (df.get("exchange", "") == "NSE")
                & (df.get("segment", "") == "NSE")
                & (df.get("instrument_type", "") == "EQ")
            ]
            if not eq.empty and "instrument_token" in eq.columns:
                token = int(eq.iloc[0]["instrument_token"])
        except Exception:
            token = None
    if token is None:
        try:
            inst = pd.DataFrame(kite.instruments("NSE"))
            eq = inst[inst.get("instrument_type", "") == "EQ"]
            if not eq.empty and "instrument_token" in eq.columns:
                token = int(eq.iloc[0]["instrument_token"])
        except Exception as e:
            if logger:
                logger.warning("Kite websocket probe failed selecting token: %s", e)
            return False
    if token is None:
        return False

    got_tick = threading.Event()
    done = threading.Event()

    kws = KiteTicker(api_key, access_token)

    def _on_connect(ws, _response):
        try:
            ws.subscribe([token])
            ws.set_mode(ws.MODE_LTP, [token])
        except Exception:
            done.set()

    def _on_ticks(_ws, ticks):
        if ticks:
            got_tick.set()
            done.set()

    def _on_error(_ws, code, reason):
        if logger:
            logger.warning("Kite websocket probe error code=%s reason=%s", code, reason)
        done.set()

    def _on_close(_ws, code, reason):
        if logger:
            logger.info("Kite websocket probe closed code=%s reason=%s", code, reason)
        done.set()

    kws.on_connect = _on_connect
    kws.on_ticks = _on_ticks
    kws.on_error = _on_error
    kws.on_close = _on_close

    try:
        kws.connect(threaded=True)
        deadline = time.time() + max(wait_seconds, 2.0)
        while time.time() < deadline and not done.is_set():
            time.sleep(0.2)
    except Exception as e:
        if logger:
            logger.warning("Kite websocket probe exception: %s", e)
        return False
    finally:
        try:
            kws.close()
        except Exception:
            pass

    return got_tick.is_set()
