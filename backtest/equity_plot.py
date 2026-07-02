#!/usr/bin/env python3
"""
Equity curve for ATR_TRAIL at recommended config: atr_mult=0.75
Reuses bar_cache.pkl and daily_metrics.pkl already built by sensitivity_backtest.py
"""
import sys, pickle, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── reuse helpers from sensitivity_backtest ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sensitivity_backtest import run_once_np, calc_stats, START_CAP
from paths import PICKLE_PATH, BAR_CACHE_PATH, RESULTS_DIR as OUT_DIR

# ── CONFIG ────────────────────────────────────────────────────────────────
PARAMS = dict(atr_mult=0.75, rel_vol=1.80, r_factor=0.30, min_turn=25e7, top_k=8)

print("Loading daily metrics ...")
with open(PICKLE_PATH, "rb") as f:
    daily = pickle.load(f)

print("Loading bar cache ...")
with open(BAR_CACHE_PATH, "rb") as f:
    bar_cache = pickle.load(f)

print("Running simulation (atr_mult=0.75) ...")
pnl_list, eq_list = run_once_np(daily, bar_cache, **PARAMS)
stats = calc_stats(pnl_list, eq_list)
print(f"Sharpe={stats['sharpe']}  CAGR={stats['cagr_pct']}%  "
      f"MDD={stats['mdd_pct']}%  WR={stats['win_rate']}%  N={stats['n_trades']}")

# ── build time-indexed equity series ─────────────────────────────────────
dates  = sorted(daily["date_only"].unique())
eq_ser = pd.Series(eq_list, index=pd.to_datetime(dates), dtype=float)
peak   = eq_ser.cummax()
dd_ser = (eq_ser - peak) / peak * 100.0

# daily P&L for returns
pnl_df  = pd.DataFrame(pnl_list)
daily_pnl = pnl_df.groupby("date")["net_pnl"].sum().reindex(
    pd.to_datetime(dates), fill_value=0.0)
daily_ret = daily_pnl / START_CAP * 100.0   # % return per day

# year-by-year returns
eq_ser.index = pd.to_datetime(eq_ser.index)
yoy = {}
years = sorted(eq_ser.index.year.unique())
for yr in years:
    yr_eq = eq_ser[eq_ser.index.year == yr]
    if len(yr_eq) < 2:
        continue
    # start value = end of prior year (or START_CAP)
    prior = eq_ser[eq_ser.index.year < yr]
    start = float(prior.iloc[-1]) if not prior.empty else START_CAP
    end   = float(yr_eq.iloc[-1])
    yoy[yr] = (end / start - 1) * 100.0

# ── PLOT ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(3, 2, figure=fig,
               height_ratios=[3, 1.5, 1.2],
               hspace=0.45, wspace=0.35)

ax_eq  = fig.add_subplot(gs[0, :])   # equity curve (full width)
ax_dd  = fig.add_subplot(gs[1, :])   # drawdown (full width)
ax_yoy = fig.add_subplot(gs[2, 0])   # year-by-year bar
ax_ret = fig.add_subplot(gs[2, 1])   # daily return distribution

# ── (1) Equity curve ──────────────────────────────────────────────────────
ax_eq.plot(eq_ser.index, eq_ser.values / 1000, color="#1f77b4", lw=1.8, label="Equity")
ax_eq.fill_between(eq_ser.index, START_CAP / 1000, eq_ser.values / 1000,
                   where=eq_ser.values >= START_CAP,
                   alpha=0.12, color="#1f77b4")
ax_eq.axhline(START_CAP / 1000, color="gray", ls="--", lw=0.8, alpha=0.6)

# shade IS vs OOS
is_end  = pd.Timestamp("2022-12-31")
oos_start = pd.Timestamp("2023-01-01")
ax_eq.axvspan(eq_ser.index[0], is_end,
              alpha=0.06, color="orange", label="IS (2021-22)")
ax_eq.axvspan(oos_start, eq_ser.index[-1],
              alpha=0.06, color="green",  label="OOS (2023-26)")

# annotate key stats
txt = (f"Sharpe {stats['sharpe']}   CAGR {stats['cagr_pct']}%   "
       f"MDD {stats['mdd_pct']}%   WR {stats['win_rate']}%   "
       f"N={stats['n_trades']} trades\n"
       f"ATR_TRAIL  atr_mult=0.75  r_factor=0.30  rel_vol=1.8  top_k=8")
ax_eq.set_title(txt, fontsize=9.5, pad=6)
ax_eq.set_ylabel("Equity (INR '000)", fontsize=9)
ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_eq.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_eq.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax_eq.legend(fontsize=8, loc="upper left")
ax_eq.grid(True, alpha=0.25)

# ── (2) Drawdown ──────────────────────────────────────────────────────────
ax_dd.fill_between(dd_ser.index, dd_ser.values, 0,
                   color="#d62728", alpha=0.55, label="Drawdown")
ax_dd.plot(dd_ser.index, dd_ser.values, color="#d62728", lw=0.8)
ax_dd.axhline(0, color="gray", lw=0.6)
ax_dd.axhline(stats["mdd_pct"], color="#d62728", ls=":", lw=1,
              alpha=0.7, label=f"Max DD {stats['mdd_pct']}%")
ax_dd.axvspan(eq_ser.index[0], is_end, alpha=0.06, color="orange")
ax_dd.axvspan(oos_start, eq_ser.index[-1], alpha=0.06, color="green")
ax_dd.set_ylabel("Drawdown (%)", fontsize=9)
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7.5)
ax_dd.legend(fontsize=8, loc="lower left")
ax_dd.grid(True, alpha=0.25)

# ── (3) Year-by-year returns ───────────────────────────────────────────────
yr_labels = [str(y) for y in yoy.keys()]
yr_vals   = list(yoy.values())
colors    = ["#2ca02c" if v >= 0 else "#d62728" for v in yr_vals]
bars = ax_yoy.bar(yr_labels, yr_vals, color=colors, edgecolor="white", width=0.6)
for bar, val in zip(bars, yr_vals):
    ax_yoy.text(bar.get_x() + bar.get_width() / 2,
                val + (1.5 if val >= 0 else -3.5),
                f"{val:+.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax_yoy.axhline(0, color="gray", lw=0.8)
ax_yoy.set_title("Year-by-Year Return", fontsize=9, fontweight="bold")
ax_yoy.set_ylabel("Return (%)", fontsize=8)
ax_yoy.tick_params(axis="x", labelsize=8)
ax_yoy.grid(True, alpha=0.25, axis="y")

# ── (4) Daily return distribution ─────────────────────────────────────────
non_zero = daily_ret[daily_ret != 0]
ax_ret.hist(non_zero, bins=50, color="#1f77b4", edgecolor="white",
            alpha=0.75, density=True)
ax_ret.axvline(0, color="gray", lw=0.8)
ax_ret.axvline(non_zero.mean(), color="orange", lw=1.4, ls="--",
               label=f"Mean {non_zero.mean():.2f}%")
ax_ret.set_title("Daily Return Distribution (active days)", fontsize=9, fontweight="bold")
ax_ret.set_xlabel("Daily Return (%)", fontsize=8)
ax_ret.set_ylabel("Density", fontsize=8)
ax_ret.legend(fontsize=7.5)
ax_ret.grid(True, alpha=0.25)

fig.suptitle("ATR_TRAIL Strategy — Recommended Configuration (atr_mult=0.75)",
             fontsize=12, fontweight="bold", y=0.98)

out = OUT_DIR / "equity_curve_atr075.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved -> {out}")
