#!/usr/bin/env python3
"""
4-config filter comparison vs rf=0.15 baseline.
Single plot: equity curves + drawdown + stats table.
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
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from paths import PICKLE_PATH, BAR_CACHE_PATH, RESULTS_DIR, DB_HOST, DB_PORT, DB_NAME, DB_USER
from sensitivity_backtest import run_once_np, calc_stats, START_CAP

IS_END    = date(2022, 12, 31)
OOS_START = date(2023,  1,  1)
BASE      = dict(atr_mult=1.5, rel_vol=1.8, r_factor=0.15, min_turn=25e7, top_k=8)
ORB_S     = 9 * 3600 + 20 * 60
ORB_E     = 9 * 3600 + 35 * 60

CONFIGS = [
    ("Baseline",                    dict()),
    ("VIX ≥ 16",                    dict(min_vix=16)),
    ("Score ≥ 2.0",                 dict(min_score=2.0)),
    ("VIX ≥ 16  +  Score ≥ 2.0",   dict(min_vix=16, min_score=2.0)),
]
COLORS = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
STYLES = ["--", "-", "-", "-"]
WIDTHS = [1.5, 2.0, 2.0, 2.2]

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading caches ...", flush=True)
with open(PICKLE_PATH,    "rb") as f: daily     = pickle.load(f)
with open(BAR_CACHE_PATH, "rb") as f: bar_cache = pickle.load(f)

print("Loading VIX ...", flush=True)
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER)
cur  = conn.cursor()
cur.execute("SELECT date, close FROM eod_prices WHERE ticker='INDIAVIX' ORDER BY date")
vix_df = pd.DataFrame(cur.fetchall(), columns=["date", "vix"])
conn.close()
vix_df["date"]     = pd.to_datetime(vix_df["date"]).dt.date
vix_df["prev_vix"] = vix_df["vix"].shift(1)
vix_map = dict(zip(vix_df["date"], vix_df["prev_vix"]))

daily = daily.copy()
daily["prev_vix"] = daily["date_only"].map(vix_map)

all_dates = sorted(daily["date_only"].unique())
N_TOTAL   = len(all_dates)

# ── Filter + run ───────────────────────────────────────────────────────────────
def run_config(min_vix=None, min_score=None):
    d = daily
    if min_vix is not None:
        good = set(d.loc[d["prev_vix"] >= min_vix, "date_only"])
        d = d[d["date_only"].isin(good)]
    if min_score is not None:
        d = d[d["score"] >= min_score]
    dates  = sorted(d["date_only"].unique())
    idle   = (N_TOTAL - len(dates)) / N_TOTAL * 100
    pnl, eq = run_once_np(d, bar_cache, **BASE)
    s = calc_stats(pnl, eq)
    eq_ser  = pd.Series(eq, index=pd.to_datetime(dates))
    return s, eq_ser, idle, pnl

# ── Run all 4 ──────────────────────────────────────────────────────────────────
results = []
for label, kw in CONFIGS:
    print(f"  {label} ...", end=" ", flush=True)
    s, eq_ser, idle, pnl = run_config(**kw)
    results.append((label, s, eq_ser, idle, pnl))
    print(f"Sharpe={s['sharpe']}  K={s['kfactor']}  CAGR={s['cagr_pct']}%  "
          f"MDD={s['mdd_pct']}%  Idle={idle:.1f}%", flush=True)

# ── IS / OOS Sharpe ────────────────────────────────────────────────────────────
def period_sharpe(pnl_list, start, end):
    df = pd.DataFrame(pnl_list)
    if df.empty: return 0.0
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty: return 0.0
    dp = df.groupby("date")["net_pnl"].sum() / START_CAP
    return round(dp.mean() / dp.std() * 252**0.5, 2) if dp.std() > 0 else 0.0

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = GridSpec(2, 2, figure=fig, height_ratios=[3, 1.6], hspace=0.42, wspace=0.32)
ax_eq = fig.add_subplot(gs[0, :])
ax_dd = fig.add_subplot(gs[1, 0])
ax_tb = fig.add_subplot(gs[1, 1])

fig.suptitle("Selectivity Filter Comparison  (baseline = rf=0.15, atr_mult=1.5)",
             fontsize=12, fontweight="bold")

# Shade IS / OOS
is_end_ts  = pd.Timestamp("2022-12-31")
oos_start_ts = pd.Timestamp("2023-01-01")

for i, (label, s, eq_ser, idle, pnl) in enumerate(results):
    ax_eq.plot(eq_ser.index, eq_ser.values / 1000,
               color=COLORS[i], ls=STYLES[i], lw=WIDTHS[i], label=label, zorder=3+i)
    peak = eq_ser.cummax()
    dd   = (eq_ser - peak) / peak * 100
    ax_dd.plot(dd.index, dd.values,
               color=COLORS[i], ls=STYLES[i], lw=WIDTHS[i]*0.8, label=label)

ax_eq.axvspan(eq_ser.index[0], is_end_ts,   alpha=0.05, color="orange", label="IS 2021-22")
ax_eq.axvspan(oos_start_ts, eq_ser.index[-1], alpha=0.05, color="green",  label="OOS 2023-26")
ax_eq.axhline(START_CAP / 1000, color="gray", ls=":", lw=0.8, alpha=0.5)
ax_eq.set_ylabel("Equity (INR '000)")
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_eq.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax_eq.legend(fontsize=8.5, ncol=2); ax_eq.grid(True, alpha=0.25)

ax_dd.axhline(0, color="gray", lw=0.6)
ax_dd.axvspan(eq_ser.index[0], is_end_ts,    alpha=0.05, color="orange")
ax_dd.axvspan(oos_start_ts, eq_ser.index[-1], alpha=0.05, color="green")
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown Comparison", fontsize=9, fontweight="bold")
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_dd.legend(fontsize=7); ax_dd.grid(True, alpha=0.25)

# Stats table
ax_tb.axis("off")
col_labels = ["Config", "Sharpe", "K-Ratio", "CAGR%", "MDD%", "IS Sh", "OOS Sh", "Idle%"]
rows = []
for label, s, eq_ser, idle, pnl in results:
    is_sh  = period_sharpe(pnl, date(2021, 1, 1), IS_END)
    oos_sh = period_sharpe(pnl, OOS_START, date(2026, 5, 30))
    rows.append([label, f"{s['sharpe']:.2f}", f"{s['kfactor']:.3f}",
                 f"{s['cagr_pct']:.1f}", f"{s['mdd_pct']:.1f}",
                 f"{is_sh:.2f}", f"{oos_sh:.2f}", f"{idle:.1f}"])

tbl = ax_tb.table(cellText=rows, colLabels=col_labels,
                  cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(COLORS[row-1] + "22")
    cell.set_edgecolor("#cccccc")
ax_tb.set_title("Stats Summary", fontsize=9, fontweight="bold", pad=4)

out = RESULTS_DIR / "filter_compare.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nPlot → {out}", flush=True)
