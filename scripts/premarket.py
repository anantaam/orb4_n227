"""
ORB Pre-market script (oneshot, ~08:30 IST)
==========================================
1. Holiday gate -- exit if non-trading day
2. Kite login (no orders placed)
3. Download 1m data for FnO universe -> /home/ubuntu/orb/data/
4. Compute ATR(14) and RelVol(20) per stock using prior-day values
5. Save state/{mode}/preranking_{date}.json
6. Discord notification
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Allow running from /home/ubuntu/orb
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from orb_config import load_config
from orb_notifier import notify, notify_error
from orb_session_idle import handle_non_session_day, ist_today_date
from orb_state import append_jsonl, ensure_dirs, preranking_file, save_json, utc_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── CLI args ──────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Skip holiday gate")
    p.add_argument("--paper", action="store_true", help="Force paper mode")
    p.add_argument("--days", type=int, default=55, help="Days of 1m data to download")
    return p.parse_args()


# ── Data download ─────────────────────────────────────────────────────────────


def download_1m_data(days: int, data_dir: Path) -> None:
    """Download full FnO-underlying 1m universe (vendored from e3, uses orb's own venv/kite_client)."""
    dl_script = Path(__file__).resolve().parent / "download_kite_1m_fno.py"
    if not dl_script.exists():
        logger.warning("Download script not found at %s -- skipping download", dl_script)
        return
    # --force skips the session-day gate (premarket.py already does its own gate)
    cmd = [sys.executable, str(dl_script), "--days", str(days), "--out", str(data_dir), "--force"]
    logger.info("Running download: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.warning("Download script exited with code %d", result.returncode)


# ── Feature computation ───────────────────────────────────────────────────────


def compute_atr(df_daily: pd.DataFrame, lookback: int) -> float | None:
    """ATR(lookback) using prior sessions (shift(1) applied)."""
    if len(df_daily) < lookback + 1:
        return None
    tr = pd.concat([
        df_daily["high"] - df_daily["low"],
        (df_daily["high"] - df_daily["close"].shift(1)).abs(),
        (df_daily["low"] - df_daily["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(lookback).mean().shift(1)
    val = atr_series.iloc[-1]
    return float(val) if not pd.isna(val) else None


def compute_rvol_baseline(df_1m: pd.DataFrame, or_bars: int, rvol_lookback: int, today: date) -> float | None:
    """
    Compute the 20-day OR-volume baseline (raw mean, NOT the ratio).

    OR volume per day = sum of the first `or_bars` 1-min bars (09:15–10:14).
    Baseline = rolling(rvol_lookback).mean().shift(1)  -- prior sessions only, no lookahead.

    Returns the baseline value for today's session.
    main.py Phase A computes the live rvol = today_or_vol / baseline at 10:14.
    Expects timezone-naive index.
    """
    if df_1m.empty:
        return None

    df_1m = df_1m.copy()
    # Ensure timezone-naive DatetimeIndex with neutral index name
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        df_1m.index = pd.to_datetime(df_1m.index)
    if df_1m.index.tz is not None:
        df_1m.index = df_1m.index.tz_localize(None)
    df_1m.index.name = "ts"
    # .date returns plain date objects -- safe to compare with today (also plain date)
    df_1m["date"] = df_1m.index.date

    # Only use bars before today (prior sessions)
    df_prior = df_1m[df_1m["date"] < today]
    if df_prior.empty:
        return None

    # OR = 09:15 to 09:15 + or_bars minutes
    or_start = datetime.strptime("09:20", "%H:%M").time()
    or_end_min = or_start.hour * 60 + or_start.minute + or_bars
    or_end_h, or_end_m = divmod(or_end_min, 60)
    from datetime import time as dtime
    or_end = dtime(or_end_h, or_end_m)

    t = df_prior.index.time
    or_mask = (t >= or_start) & (t <= or_end)  # inclusive 09:20–09:35, matches backtest
    or_vols = df_prior[or_mask].groupby("date")["volume"].sum()

    if len(or_vols) < rvol_lookback + 1:
        return None

    # Return the baseline (20-day mean of OR vol, shifted 1 -- no lookahead)
    baseline_series = or_vols.rolling(rvol_lookback).mean().shift(1)
    val = baseline_series.iloc[-1]
    return float(val) if not pd.isna(val) else None


def compute_turn3(df_1m: pd.DataFrame, today: date) -> float | None:
    """3-day mean of daily turnover (close*volume summed intraday), shift(1) — no lookahead.
    Matches backtest turn3 = daily_turn.rolling(3).mean().shift(1)."""
    if df_1m.empty:
        return None
    df = df_1m.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["date"] = df.index.date
    df = df[df["date"] < today]  # prior sessions only
    if df.empty:
        return None
    daily_turn = (df["close"] * df["volume"]).groupby(df["date"]).sum()
    if len(daily_turn) < 4:
        return None
    val = daily_turn.rolling(3).mean().shift(1).iloc[-1]
    return float(val) if not pd.isna(val) else None


def aggregate_daily(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m bars to daily OHLC. Expects timezone-naive DatetimeIndex."""
    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Drop any date helper column before resample to avoid aggregation errors
    df = df.drop(columns=["date"], errors="ignore")
    result = df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    # Drop non-trading days: volume sum returns 0 (not NaN) for empty buckets,
    # but open is NaN for days with no bars -- use open as the trading-day indicator.
    return result.dropna(subset=["open"])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    cfg = load_config(BASE_DIR)
    if args.paper:
        cfg = dataclasses.replace(cfg, mode="paper")

    ensure_dirs(cfg.data_1m_dir, cfg.state_dir / cfg.mode)

    trade_date = ist_today_date()
    today_str = trade_date.isoformat()

    if not args.force:
        if handle_non_session_day(cfg, phase="premarket"):
            logger.info("Non-trading day -- exiting.")
            sys.exit(0)

    logger.info("ORB PREMARKET start: %s mode=%s", today_str, cfg.mode)

    # Step 2: Kite login (needed for download, no orders placed)
    try:
        from orb_broker_adapter import BrokerAdapter
        broker = BrokerAdapter(paper=(cfg.mode == "paper"), cfg=cfg)
        broker.login()
        logger.info("Kite login OK")
    except Exception as exc:
        logger.error("Kite login failed: %s", exc)
        notify_error(cfg.discord_webhook_url, f"Premarket Kite login failed: {exc}")
        sys.exit(1)

    # Step 3: Download 1m data
    download_1m_data(days=args.days, data_dir=cfg.data_1m_dir)

    # Step 4: Compute ATR14 + RelVol20 for each stock
    records = []
    # CSVs are named SYMBOL_1min_kite.csv
    csv_files = sorted(cfg.data_1m_dir.glob("*_1min_kite.csv"))
    logger.info("Computing features for %d CSV files", len(csv_files))

    for csv_path in csv_files:
        # Strip _1min_kite suffix to get plain symbol
        symbol = csv_path.stem.replace("_1min_kite", "")
        try:
            df_1m = pd.read_csv(csv_path, index_col=0)
            if df_1m.empty:
                continue
            # Standardise column names to lowercase
            df_1m.columns = [c.lower() for c in df_1m.columns]
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df_1m.columns)):
                continue
            # Parse index, strip tz, rename index to avoid column-name conflicts
            df_1m.index = pd.to_datetime(df_1m.index, utc=False)
            if df_1m.index.tz is not None:
                df_1m.index = df_1m.index.tz_localize(None)
            df_1m.index.name = "ts"

            df_daily = aggregate_daily(df_1m)
            atr14 = compute_atr(df_daily, cfg.or_atr_lookback)
            # Baseline = 20-day rolling mean of OR volume (shift(1), no today data).
            # main.py Phase A will compute live rvol = today_or_vol / or_vol_baseline.
            or_vol_baseline = compute_rvol_baseline(df_1m, cfg.or_bars, cfg.rvol_lookback, trade_date)
            turn3 = compute_turn3(df_1m, trade_date)

            if atr14 is None or atr14 <= 0:
                continue

            records.append({
                "symbol": symbol,
                "atr14": round(atr14, 4),
                "or_vol_baseline": round(or_vol_baseline, 2) if or_vol_baseline is not None else None,
                "turn3": round(turn3, 2) if turn3 is not None else None,
            })
        except Exception as exc:
            logger.debug("Skip %s: %s", symbol, exc)
            continue

    logger.info("Pre-ranked %d stocks", len(records))

    # Step 5: Save preranking JSON
    preranking = {
        "date": today_str,
        "mode": cfg.mode,
        "stocks": records,
        "computed_at": utc_now(),
    }
    out_path = preranking_file(cfg.state_dir, cfg.mode, today_str)
    save_json(out_path, preranking)
    logger.info("Saved preranking -> %s", out_path)

    append_jsonl(cfg.journal_path, {
        "ts_utc": utc_now(),
        "event": "premarket_complete",
        "trade_date": today_str,
        "n_stocks": len(records),
    })

    # Step 6: Discord
    notify(
        cfg.discord_webhook_url,
        f"ORB PREMARKET complete [{cfg.mode}] {today_str} -- {len(records)} stocks pre-ranked, session confirmed.",
    )
    logger.info("ORB PREMARKET done.")


if __name__ == "__main__":
    main()
