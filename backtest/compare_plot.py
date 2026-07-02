#!/usr/bin/env python3
"""
Multi-config equity curve comparison.

Overlays equity curves for key ATR_Multiplier values + r_factor=0.15 (dual-criterion winner).
Run from orb3_package root:
    python scripts/compare_plot.py
"""
import sys, pickle, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import PICKLE_PATH, BAR_CACHE_PATH, RESULTS_DIR
from sensitivity_backtest import run_once_np, calc_stats, START_CAP, _calc_kfactor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

CONFIGS = [
    ("atr×0.50",  dict(atr_mult=0.50, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)),
    ("atr×0.75",  dict(atr_mult=0.75, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)),
    ("atr×1.00",  dict(atr_mult=1.00, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)),
    ("atr×1.25",  dict(atr_mult=1.25, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)),
    ("atr×1.50 (base)", dict(atr_mult=1.50, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)),
    ("rf=0.15 *",  dict(atr_mult=1.50, rel_vol=1.80, r_factor=0.15, min_turn=25e7, top_k=8)),
]

COLORS = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f", "#d62728"]
STYLES = ["-", "-", "-", "-", "--", ":"]

print("Loading caches ...")
with open(PICKLE_PATH, "rb") as f: daily = pickle.load(f)
with open(BAR_CACHE_PATH, "rb") as f: bar_cache = pickle.load(f)
dates = [pd.Timestamp(d) for d in sorted(daily["date_only"].unique())]

results = []
eq_series = []
for label, params in CONFIGS:
    print(f"  {label} ...", end=" ", flush=True)
    pnl, eq = run_once_np(daily, bar_cache, **params)
    s = calc_stats(pnl, eq)
    eq_series.append(pd.Series(eq, index=dates, dtype=float))
    results.append((label, s))
    print(f"Sharpe={s['sharpe']}  K={s['kfactor']}  CAGR={s['cagr_pct']}%  MDD={s['mdd_pct']}%")

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = GridSpec(2, 2, figure=fig, height_ratios=[3, 1.8], hspace=0.45, wspace=0.35)
ax_eq = fig.add_subplot(gs[0, :])
ax_dd = fig.add_subplot(gs[1, 0])
ax_tb = fig.add_subplot(gs[1, 1])

# Equity overlay
for i, ((label, s), eq_ser) in enumerate(zip(results, eq_series)):
    lw = 2.2 if i in (1, 5) else 1.5
    ax_eq.plot(eq_ser.index, eq_ser.values / 1000,
               color=COLORS[i], ls=STYLES[i], lw=lw, label=label, zorder=3 + i)

ax_eq.axhline(START_CAP / 1000, color="gray", ls="--", lw=0.8, alpha=0.5)
ax_eq.axvspan(dates[0], pd.Timestamp("2022-12-31"), alpha=0.06, color="orange", label="IS 2021-22")
ax_eq.axvspan(pd.Timestamp("2023-01-01"), dates[-1], alpha=0.06, color="green",  label="OOS 2023-26")
ax_eq.set_ylabel("Equity (INR '000)")
ax_eq.set_title("ATR_TRAIL — Multi-Config Equity Comparison  (* = dual-criterion winner)",
                fontsize=11, fontweight="bold", pad=8)
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_eq.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax_eq.legend(fontsize=8.5, ncol=2); ax_eq.grid(True, alpha=0.25)

# Drawdown overlay
for i, ((label, s), eq_ser) in enumerate(zip(results, eq_series)):
    peak  = eq_ser.cummax()
    dd    = (eq_ser - peak) / peak * 100.0
    lw = 2.0 if i in (1, 5) else 1.2
    ax_dd.plot(dd.index, dd.values, color=COLORS[i], ls=STYLES[i], lw=lw, label=label)
ax_dd.axhline(0, color="gray", lw=0.6)
ax_dd.fill_between(dates, 0, 0, alpha=0)  # force y includes 0
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("Drawdown Comparison", fontsize=9, fontweight="bold")
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_dd.legend(fontsize=7); ax_dd.grid(True, alpha=0.25)

# Stats table
ax_tb.axis("off")
col_labels = ["Config", "Sharpe", "K-Ratio", "R²", "CAGR%", "MDD%"]
rows = []
for label, s in results:
    rows.append([label, f"{s['sharpe']:.2f}", f"{s['kfactor']:.3f}",
                 f"{s['r2']:.3f}", f"{s['cagr_pct']:.1f}", f"{s['mdd_pct']:.1f}"])
tbl = ax_tb.table(cellText=rows, colLabels=col_labels,
                  cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
    elif row in (2, 6):  # atr×0.75 (row 2) and rf=0.15 (row 6) highlighted
        cell.set_facecolor("#e8f4f8")
    else:
        cell.set_facecolor("#f9f9f9" if row % 2 else "white")
    cell.set_edgecolor("#cccccc")
ax_tb.set_title("Stats Summary", fontsize=9, fontweight="bold", pad=4)

out = RESULTS_DIR / "compare_configs.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nPlot -> {out}")
