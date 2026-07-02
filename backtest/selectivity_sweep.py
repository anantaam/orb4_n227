#!/usr/bin/env python3
"""
Selectivity sweep — 4 filter dimensions vs rf=0.15 baseline.

Filters tested (one at a time, others off):
  1. VIX threshold     : skip day if prev-day VIX < min_vix
  2. Nifty ORB width   : skip day if Nifty ORB range > max_nifty_pct of price
  3. Stock tightness   : skip stock-day if (orb_h - orb_l) / atr_orb > max_tight
  4. Score threshold   : skip stock-day if score < min_score

Baseline: rf=0.15 (atr_mult=1.5, r_factor=0.15, rel_vol=1.8, top_k=8)
IS : 2021-01-01 – 2022-12-31
OOS: 2023-01-01 – 2026-05-30

Outputs:
  results/selectivity_results.csv
  results/selectivity_sweep.png   (4×4: IS Sharpe | OOS Sharpe | OOS K | Idle%)
"""
import sys, pickle, warnings
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PICKLE_PATH, BAR_CACHE_PATH, RESULTS_DIR, DB_HOST, DB_PORT, DB_NAME, DB_USER
from sensitivity_backtest import run_once_np, calc_stats, START_CAP

# ── Constants ──────────────────────────────────────────────────────────────────
IS_END    = date(2022, 12, 31)
OOS_START = date(2023,  1,  1)

BASE = dict(atr_mult=1.5, rel_vol=1.8, r_factor=0.15, min_turn=25e7, top_k=8)

# Nifty ORB window in seconds-since-midnight
ORB_S = 9 * 3600 + 20 * 60   # 09:20
ORB_E = 9 * 3600 + 35 * 60   # 09:35

GRIDS = [
    ("VIX threshold",   "min_vix",        [10, 12, 14, 16, 18, 20, 22],
     "Skip day if prev-day VIX < X  (trade only on elevated-vol days)"),
    ("Nifty ORB width", "max_nifty_pct",  [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8],
     "Skip day if Nifty ORB range > X% of price  (skip chaotic opens)"),
    ("Stock tightness", "max_tight",      [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
     "Skip stock-day if (orb_h−orb_l)/atr > X  (only tight coiling setups)"),
    ("Score threshold", "min_score",      [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0],
     "Skip stock-day if rel_vol×atr_pct < X  (only high-quality candidates)"),
]

# ── Load caches ────────────────────────────────────────────────────────────────
print("Loading caches ...", flush=True)
with open(PICKLE_PATH,    "rb") as f: daily     = pickle.load(f)
with open(BAR_CACHE_PATH, "rb") as f: bar_cache = pickle.load(f)
print(f"  daily: {len(daily):,} rows  |  symbols: {len(bar_cache)}", flush=True)

# ── Load VIX (prev-day, no lookahead) ─────────────────────────────────────────
print("Loading India VIX ...", flush=True)
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
cur  = conn.cursor()
cur.execute("SELECT date, close FROM eod_prices WHERE ticker='INDIAVIX' ORDER BY date")
vix_df = pd.DataFrame(cur.fetchall(), columns=["date", "vix"])
conn.close()
vix_df["date"]     = pd.to_datetime(vix_df["date"]).dt.date
vix_df["prev_vix"] = vix_df["vix"].shift(1)          # use PREVIOUS day — no lookahead
vix_map = dict(zip(vix_df["date"], vix_df["prev_vix"]))
print(f"  VIX rows: {len(vix_df)}  ({vix_df['date'].min()} – {vix_df['date'].max()})", flush=True)

# ── Compute Nifty ORB range% from bar_cache ───────────────────────────────────
print("Computing Nifty ORB width ...", flush=True)
nifty_orb_map = {}
nifty_cache   = bar_cache.get("NIFTY", {})
for dt, (o_arr, h_arr, l_arr, c_arr, t_arr) in nifty_cache.items():
    mask = (t_arr >= ORB_S) & (t_arr <= ORB_E)
    if not mask.any():
        continue
    orb_h = float(h_arr[mask].max())
    orb_l = float(l_arr[mask].min())
    orb_c = float(c_arr[mask][-1])
    if orb_c > 0:
        nifty_orb_map[dt] = (orb_h - orb_l) / orb_c * 100.0
print(f"  Nifty ORB dates: {len(nifty_orb_map)}", flush=True)

# ── Enrich daily ───────────────────────────────────────────────────────────────
daily = daily.copy()
daily["tightness"]      = (daily["orb_high"] - daily["orb_low"]) / daily["atr_orb"]
daily["prev_vix"]       = daily["date_only"].map(vix_map)
daily["nifty_range_pct"]= daily["date_only"].map(nifty_orb_map)

# Total trading days in full window (for idle% denominator)
all_dates = sorted(daily["date_only"].unique())
N_TOTAL   = len(all_dates)

# ── IS / OOS split stats ────────────────────────────────────────────────────────
def split_stats(pnl_list, dates_used):
    """Return (is_stats, oos_stats) given pnl_list and the sorted dates used."""
    date_idx = {d: i for i, d in enumerate(dates_used)}
    is_pnl  = [p for p in pnl_list if p["date"] <= IS_END]
    oos_pnl = [p for p in pnl_list if p["date"] >= OOS_START]

    # Rebuild per-period equity lists (starting from START_CAP in each period)
    def _eq(pnl_sub, start_cap):
        if not pnl_sub: return []
        by_date = {}
        for p in pnl_sub:
            by_date.setdefault(p["date"], 0.0)
            by_date[p["date"]] += p["net_pnl"]
        eq = start_cap
        eq_list = []
        for d in sorted(by_date):
            eq += by_date[d]
            eq_list.append(eq)
        return eq_list

    is_eq  = _eq(is_pnl,  START_CAP)
    oos_eq = _eq(oos_pnl, START_CAP)

    return calc_stats(is_pnl, is_eq), calc_stats(oos_pnl, oos_eq)


# ── Filter + run ───────────────────────────────────────────────────────────────
def run_filtered(d, *, min_vix=None, max_nifty_pct=None, max_tight=None, min_score=None):
    # Day-level gates
    if min_vix is not None:
        good = set(d.loc[d["prev_vix"] >= min_vix, "date_only"])
        d = d[d["date_only"].isin(good)]

    if max_nifty_pct is not None:
        good = set(d.loc[d["nifty_range_pct"] <= max_nifty_pct, "date_only"])
        d = d[d["date_only"].isin(good)]

    # Stock-level gates (drop rows; nlargest still applies within remaining)
    if max_tight is not None:
        d = d[d["tightness"] <= max_tight]

    if min_score is not None:
        d = d[d["score"] >= min_score]

    dates_used = sorted(d["date_only"].unique())
    idle_pct   = (N_TOTAL - len(dates_used)) / N_TOTAL * 100.0
    pnl, eq    = run_once_np(d, bar_cache, **BASE)
    full_stats = calc_stats(pnl, eq)
    is_s, oos_s = split_stats(pnl, dates_used)
    return full_stats, is_s, oos_s, idle_pct


# ── Baseline (no filter) ───────────────────────────────────────────────────────
print("\nRunning baseline (rf=0.15, no filters) ...", flush=True)
bl_full, bl_is, bl_oos, _ = run_filtered(daily)
print(f"  Full  Sharpe={bl_full['sharpe']}  K={bl_full['kfactor']}  CAGR={bl_full['cagr_pct']}%")
print(f"  IS    Sharpe={bl_is['sharpe']}   K={bl_is['kfactor']}")
print(f"  OOS   Sharpe={bl_oos['sharpe']}  K={bl_oos['kfactor']}", flush=True)

# ── Sweep ──────────────────────────────────────────────────────────────────────
records = []
sweep_results = {}   # dim_name → list of (val, is_s, oos_s, idle_pct)

for dim_name, param, grid, description in GRIDS:
    print(f"\n── {dim_name} ──", flush=True)
    dim_rows = []
    for val in grid:
        kw = {param: val}
        full_s, is_s, oos_s, idle_pct = run_filtered(daily, **kw)
        print(f"  {param}={val:5.2f}  IS Sharpe={is_s['sharpe']:.2f}  "
              f"OOS Sharpe={oos_s['sharpe']:.2f}  OOS K={oos_s['kfactor']:.3f}  "
              f"Idle={idle_pct:.1f}%", flush=True)
        records.append({
            "dim": dim_name, "param": param, "value": val,
            "is_sharpe":  is_s["sharpe"],  "is_k":  is_s["kfactor"],
            "is_trades":  is_s["n_trades"],
            "oos_sharpe": oos_s["sharpe"], "oos_k": oos_s["kfactor"],
            "oos_trades": oos_s["n_trades"], "idle_pct": round(idle_pct, 1),
        })
        dim_rows.append((val, is_s, oos_s, idle_pct))
    sweep_results[dim_name] = dim_rows

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = RESULTS_DIR / "selectivity_results.csv"
pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nCSV → {csv_path}", flush=True)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle(
    "Selectivity Filter Sweep  (baseline = rf=0.15 | IS 2021-22 | OOS 2023-26)",
    fontsize=13, fontweight="bold"
)

col_titles = ["IS Sharpe", "OOS Sharpe", "OOS K-Ratio", "Idle Days %"]
bl_refs    = [bl_is["sharpe"], bl_oos["sharpe"], bl_oos["kfactor"], 0.0]
col_colors = ["steelblue", "seagreen", "mediumpurple", "darkorange"]

for ri, (dim_name, param, grid, _) in enumerate(GRIDS):
    rows = sweep_results[dim_name]
    vals      = [r[0] for r in rows]
    is_sharpe = [r[1]["sharpe"]   for r in rows]
    oos_sharpe= [r[2]["sharpe"]   for r in rows]
    oos_k     = [r[2]["kfactor"]  for r in rows]
    idle      = [r[3]             for r in rows]

    series = [is_sharpe, oos_sharpe, oos_k, idle]

    for ci, (data, title, ref, col) in enumerate(zip(series, col_titles, bl_refs, col_colors)):
        ax = axes[ri][ci]
        ax.plot(vals, data, "o-", color=col, lw=2, ms=6)
        ax.axhline(ref, color="gray", ls="--", lw=1.2, alpha=0.7, label=f"baseline={ref:.2f}")
        ax.set_xlabel(param, fontsize=8)
        ax.set_title(f"{dim_name}\n{title}", fontsize=8, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
        # Shade OOS Sharpe < baseline red
        if ci == 1:
            for x, y in zip(vals, data):
                if y < ref:
                    ax.axvspan(x - (vals[1]-vals[0])*0.4 if len(vals) > 1 else x - 0.5,
                               x + (vals[1]-vals[0])*0.4 if len(vals) > 1 else x + 0.5,
                               alpha=0.12, color="red")

plt.tight_layout()
out = RESULTS_DIR / "selectivity_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot → {out}", flush=True)
