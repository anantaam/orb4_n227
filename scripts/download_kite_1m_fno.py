#!/usr/bin/env python3
"""
Download 1-minute NSE EQ bars for FnO-underlying symbols via Kite Connect.

Writes CSVs compatible with compute_market_profile_metrics.py glob:
  SYMBOL_1min_kite.csv

Uses in-repo `src.kite_client` for headless login (same Kite env vars as E3 runtime).

Examples:
  python scripts/download_kite_1m_fno.py --out-dir ~/e3/data

  # Long backfill (backtests / full history) — override default depth:
  DOWNLOAD_DAYS=400 python scripts/download_kite_1m_fno.py --out-dir ~/e3/data

Environment:
  DOWNLOAD_DAYS       Default calendar depth if --days omitted (default 50).

Runs only on IST NSE **session days** unless `--force` (same gate as live_engine via `live_session_idle`).
Daily **FnO list** built from ONE fresh `kite.instruments()` call per run. After download, deletes orphan
`*_1min_kite.csv` not in current universe unless `--no-prune-orphans`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# ~50 calendar days comfortably covers 20+ sessions (rolling warm-up in metrics)
# without multi-hour full-universe pulls. Use DOWNLOAD_DAYS or --days for deeper history.
_DEFAULT_DOWNLOAD_DAYS = 50


def _prune_orphan_1m_files(out_dir: Path, expected_symbols: list[str]) -> int:
    """Remove *_1min_kite.csv not in the current FnO/EQ universe after a full run."""
    keep = {f"{s}_1min_kite.csv" for s in expected_symbols}
    n = 0
    for p in sorted(out_dir.glob("*_1min_kite.csv")):
        if p.name not in keep:
            p.unlink(missing_ok=True)
            print(f"[PRUNE] removed orphan {p.name}")
            n += 1
    if n:
        print(f"=== PRUNE {n} orphan file(s) outside current universe ===")
    return n


def fno_underlying_names_from_snapshot(instruments: list) -> list[str]:
    """Current FnO-underlying symbols from ONE `kite.instruments()` snapshot (NFO+FUT rows)."""
    names: set[str] = set()
    for r in instruments:
        if r.get("exchange") != "NFO" or str(r.get("instrument_type")).upper() != "FUT":
            continue
        name = (r.get("name") or "").strip().upper()
        if name:
            names.add(name)
    return sorted(names)


def nse_eq_token_map_from_snapshot(instruments: list) -> dict[str, int]:
    """EQ token map from the SAME instruments snapshot — no stale second HTTP round-trip."""
    out: dict[str, int] = {}
    for r in instruments:
        if r.get("exchange") != "NSE" or r.get("segment") != "NSE":
            continue
        if str(r.get("instrument_type")).upper() != "EQ":
            continue
        sym = str(r.get("tradingsymbol", "")).strip().upper()
        tok = r.get("instrument_token")
        if sym and tok is not None:
            try:
                out[sym] = int(tok)
            except (TypeError, ValueError):
                pass
    return out


def fetch_minute_series(kite, token: int, start: datetime, end: datetime, *, chunk_days: int, pause_sec: float) -> pd.DataFrame | None:
    rows: list = []
    cur = start

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        for attempt in range(1, 5):
            try:
                chunk = kite.historical_data(int(token), cur, nxt, interval="minute", continuous=False)
                if chunk:
                    rows.extend(chunk)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"[FAIL] token={token} {cur}->{nxt} {e}")
                    return None
                time.sleep(0.5 * attempt)
        time.sleep(pause_sec)
        cur = nxt + timedelta(seconds=1)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Download 1m Kite equity data for FnO underlyings")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--days",
        type=int,
        default=int(os.environ.get("DOWNLOAD_DAYS", str(_DEFAULT_DOWNLOAD_DAYS))),
    )
    ap.add_argument("--chunk-days", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.35, help="Seconds between chunk requests")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only first N symbols (debug)")
    ap.add_argument("--paper-login", action="store_true", help="Use kite_client.login(paper=True)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Run even on non-trading days (adhoc backfill; default is IST session-day only).",
    )
    ap.add_argument(
        "--no-prune-orphans",
        action="store_true",
        help="Keep CSVs for symbols no longer in FnO universe.",
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if not args.force:
        from live_config import load_config
        from live_session_idle import handle_non_session_day, ist_today_date, notify_trading_session_day_once

        cfg = load_config(repo_root)
        if handle_non_session_day(cfg, "download_1m", ist_today_date()):
            print("=== SKIP non-session IST (Discord + optional shutdown); use --force to override ===")
            return 0
        notify_trading_session_day_once(cfg, "download_1m", ist_today_date())
    from src import kite_client  # noqa: E402

    kite = kite_client.login(paper=args.paper_login)
    raw_instruments = kite.instruments()
    print(f"=== FnO+EQ universe rebuilt from single kite.instruments() snapshot ({len(raw_instruments)} rows) ===")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    instruments_path = args.out_dir / "instruments.csv"
    pd.DataFrame(raw_instruments).to_csv(instruments_path, index=False)
    print(f"Saved instruments snapshot for tick sizes -> {instruments_path}")

    univ = fno_underlying_names_from_snapshot(raw_instruments)
    eq_map = nse_eq_token_map_from_snapshot(raw_instruments)

    symbols = [s for s in univ if s in eq_map]
    if args.limit > 0:
        symbols = symbols[: args.limit]
    missing = sorted(set(univ) - set(symbols))

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    end_dt = now_ist.replace(hour=16, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(args.days))

    print(f"Universe FnO-underlyings: {len(univ)} | downloadable EQ tokens: {len(symbols)} | missing tokens: {len(missing)}")
    print(f"Window: {start_dt.date()} -> {end_dt.date()} (~{args.days}d)")

    ok = 0
    for i, sym in enumerate(symbols, 1):
        tok = eq_map[sym]
        df = fetch_minute_series(kite, tok, start_dt, end_dt, chunk_days=args.chunk_days, pause_sec=args.pause)
        if df is None or df.empty:
            print(f"[{i}/{len(symbols)}] SKIP {sym} empty")
            continue
        dest = args.out_dir / f"{sym}_1min_kite.csv"
        df.to_csv(dest, index=False)
        ok += 1
        print(f"[{i}/{len(symbols)}] OK {sym} rows={len(df):,}")

    print(f"=== DONE {ok}/{len(symbols)} files -> {args.out_dir} ===")
    if ok and not args.no_prune_orphans and not args.limit:
        _prune_orphan_1m_files(args.out_dir, symbols)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
