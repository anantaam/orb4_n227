#!/usr/bin/env python3
"""
DB data loader for ORB backtest — psycopg2 interface to nse.intraday_1min.

Processes one symbol at a time to keep peak RAM under ~50 MB regardless of
universe size (vs ~12 GB if all 102M rows are fetched into Python at once).
"""
import sys, time as _time
from pathlib import Path
from datetime import time as dtime

import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).parent))
from paths import DB_HOST, DB_PORT, DB_NAME, DB_USER

# ── Constants (must match sensitivity_backtest.py) ─────────────────────────
ORB_START         = dtime(9, 20)
ORB_END           = dtime(9, 35)
ATR_PER           = 14
VOL_PER           = 14
SESSION_START_SEC = 32400   # 09:00:00
SESSION_END_SEC   = 56160   # 15:36:00

SQL = """
    SELECT (ts AT TIME ZONE 'Asia/Kolkata')::timestamp AS ts_ist,
           open, high, low, close, volume
    FROM   intraday_1min
    WHERE  ticker = %s
      AND  EXTRACT(HOUR FROM (ts AT TIME ZONE 'Asia/Kolkata')) BETWEEN 9 AND 15
    ORDER  BY ts
"""
COLS = ["ts", "open", "high", "low", "close", "volume"]


def get_conn():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)


def load_universe_from_db():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("""
        SELECT f.symbol FROM fno_constituents f
        WHERE EXISTS (SELECT 1 FROM intraday_1min i WHERE i.ticker = f.symbol LIMIT 1)
        ORDER BY f.symbol
    """)
    symbols = [r[0] for r in cur.fetchall()]
    conn.close()
    print(f"Universe: {len(symbols)} symbols (nse.fno_constituents)")
    return symbols


def _fetch_one(conn, sym):
    """Fetch all session bars for a single symbol. Returns DataFrame or None."""
    cur = conn.cursor()
    cur.execute(SQL, (sym,))
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COLS)
    df["ts"] = pd.to_datetime(df["ts"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = sym
    return df


def _sym_daily_metrics(df, sym):
    """Compute ORB daily metrics for one symbol's bar DataFrame."""
    df = df.sort_values("ts").reset_index(drop=True)
    df["date_only"] = df["ts"].dt.date
    df["t"]         = df["ts"].dt.time

    df["prev_c"] = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_c"]).abs(),
        (df["low"]  - df["prev_c"]).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PER, min_periods=ATR_PER).mean()
    df["vt"]  = df["close"] * df["volume"]

    daily_turn = df.groupby("date_only")["vt"].sum().rename("daily_turn")

    orb = df[(df["t"] >= ORB_START) & (df["t"] <= ORB_END)]
    if orb.empty:
        return None

    def _agg(g):
        return pd.Series({
            "orb_open":  float(g["open"].iloc[0]),
            "orb_high":  float(g["high"].max()),
            "orb_low":   float(g["low"].min()),
            "orb_close": float(g["close"].iloc[-1]),
            "orb_vol":   float(g["volume"].sum()),
            "orb_turn":  float(g["vt"].sum()),
            "atr_orb":   float(g["atr"].iloc[-1]),
        })

    agg = orb.groupby("date_only").apply(_agg).reset_index()
    agg = agg.join(daily_turn, on="date_only")
    agg["symbol"] = sym

    agg["orb_vol_ma"] = agg["orb_vol"].rolling(VOL_PER, min_periods=VOL_PER).mean().shift(1)
    agg["rel_vol"]    = agg["orb_vol"] / agg["orb_vol_ma"]
    agg["turn3"]      = agg["daily_turn"].rolling(3, min_periods=3).mean().shift(1)
    agg["turn_B"]     = agg["turn3"] + agg["orb_turn"]
    agg["direction"]  = np.sign(agg["orb_close"] - agg["orb_open"])
    agg["atr_pct"]    = agg["atr_orb"] / agg["orb_close"] * 100
    agg["score"]      = agg["rel_vol"] * agg["atr_pct"]

    return agg.dropna(subset=["atr_orb", "rel_vol", "turn_B"])


def _sym_bar_cache(df):
    """Build bar_cache dict for one symbol."""
    df = df.copy()
    df["date_only"] = df["ts"].dt.date
    df["tsec"] = (df["ts"].dt.hour * 3600
                  + df["ts"].dt.minute * 60
                  + df["ts"].dt.second).astype(np.int32)
    mask = (df["tsec"] >= SESSION_START_SEC) & (df["tsec"] < SESSION_END_SEC)
    df   = df[mask].sort_values("tsec")
    sym_dict = {}
    for date, grp in df.groupby("date_only"):
        sym_dict[date] = (
            grp["open"].values.astype(np.float32),
            grp["high"].values.astype(np.float32),
            grp["low"].values.astype(np.float32),
            grp["close"].values.astype(np.float32),
            grp["tsec"].values.astype(np.int32),
        )
    return sym_dict


def load_all_from_db(symbols):
    """
    Process one symbol at a time — peak RAM ~30 MB regardless of universe size.
    Returns (daily_metrics_df, bar_cache_dict).
    """
    t0 = _time.time()
    conn = get_conn()
    daily_parts = []; bar_cache = {}

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:3d}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        t1  = _time.time()
        df  = _fetch_one(conn, sym)
        if df is None:
            print("no data", flush=True); continue
        dm  = _sym_daily_metrics(df, sym)
        if dm is not None:
            daily_parts.append(dm)
        bar_cache[sym] = _sym_bar_cache(df)
        print(f"{len(df):,} bars  {_time.time()-t1:.1f}s", flush=True)
        del df

    conn.close()
    daily = pd.concat(daily_parts, ignore_index=True)
    print(f"\n  {len(daily):,} daily rows  |  {len(bar_cache)} symbols  "
          f"|  total: {(_time.time()-t0)/60:.1f} min", flush=True)
    return daily, bar_cache
